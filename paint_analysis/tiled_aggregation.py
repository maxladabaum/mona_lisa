"""Incremental tiled results: keep metadata in memory and numeric data on disk."""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
import tempfile
import hashlib
from crash_tracking import record as record_diagnostic

import numpy as np
import pandas as pd

from origami_analysis import OrigamiPickResult


SEQUENCE_PARAMS = (
    'classification_observed_digital_states', 'classification_exact_digital_match',
    'classification_lattice_precision', 'classification_empty_cell_fraction',
    'classification_bright_cell_probability', 'classification_cell_pattern_correlation',
    'classification_cell_agreement', 'classification_monte_carlo_posterior',
    'classification_monte_carlo_log_bayes_factor', 'classification_scores',
    'classification_dispositions', 'classification_qc_eligible',
    'classification_lookup_eligible', '_alignment_accepted_mask',
    'classification_winner_margins', 'classification_runner_up_templates',
    'classification_template_probabilities', 'classification_raw_template_probabilities',
    'classification_template_probability_vectors', 'classification_raw_template_probability_vectors',
    'classification_crop_retained_fraction', 'classification_logical_bit_probabilities',
    'digital_group_localization_evidence', 'digital_group_prominences', 'digital_pixel_probabilities',
)


# These dense candidate-by-template matrices dominated retained Python metadata.
# Keep full float64 precision, but store their values on disk, not as Python floats.
DISK_SEQUENCE_PARAMS = {
    'classification_template_probability_vectors',
    'classification_raw_template_probability_vectors',
}


@dataclass
class _ArrayRef:
    offset: int
    shape: tuple
    dtype: np.dtype


@dataclass
class _ObjectArrayRef:
    shape: tuple
    values: list


@dataclass
class _LiteralTuple:
    """Immutable scalar metadata needs neither decoding nor a second copy."""
    values: tuple

    def __iter__(self):
        return iter(self.values)


class _ArrayStore:
    """One shared file for variable-size region, history, and parameter arrays."""
    def __init__(self, directory):
        self.path = directory / 'arrays.bin'
        self.path.touch()
        self.size = 0
        self.buffer = None
        self.writer = None
        self.retained = []

    @contextmanager
    def writing(self):
        # One buffered stream per tile. ndarray.tofile bypasses Python buffering
        # and performs seek/flush work for each tiny array on external drives.
        with self.path.open('ab', buffering=1024 * 1024) as handle:
            self.writer = handle
            try:
                yield
            finally:
                self.writer = None
                self.retained.clear()

    def put(self, value, memo):
        if isinstance(value, np.ndarray):
            key = id(value)
            if key in memo:
                return memo[key]
            # Classification parameters also expose shared arrays through fresh
            # row views. Recognize the same memory layout, not just object IDs.
            layout = ('layout', value.__array_interface__['data'][0],
                      value.shape, value.strides, value.dtype)
            if not value.dtype.hasobject and layout in memo:
                return memo[layout]
            self.retained.append(value)  # Prevent temporary-array ID reuse within this tile.
            if value.dtype.hasobject:
                # Object values are metadata, never raw Python pointers on disk.
                result = _ObjectArrayRef(value.shape, [self.put(item, memo) for item in value.flat])
                memo[key] = result
                return result
            offset = (self.size + 63) // 64 * 64
            if self.writer is None:
                raise RuntimeError('Array storage requires an active tile write.')
            self.writer.write(b'\0' * (offset - self.size))
            if value.nbytes:
                self.writer.write(memoryview(np.ascontiguousarray(value)).cast('B'))
            self.size = offset + value.nbytes
            result = _ArrayRef(offset, value.shape, value.dtype)
            memo[key] = result
            memo[layout] = result
            return result
        if isinstance(value, (dict, tuple, list)):
            key = id(value)
            if key in memo:
                return memo[key]
            # Shared classification vectors and template metadata can be nested
            # Python containers, too. Copying them for every template grows RAM
            # throughout a saved run even though numeric arrays live on disk.
            self.retained.append(value)
            if isinstance(value, tuple) and all(
                    item is None or isinstance(item, (bool, int, float, complex, str, bytes, np.generic))
                    and not (isinstance(item, np.generic) and item.dtype.hasobject)
                    for item in value):
                result = _LiteralTuple(value)
            elif isinstance(value, dict):
                result = {key: self.put(item, memo) for key, item in value.items()}
            else:
                result = type(value)(self.put(item, memo) for item in value)
            memo[key] = result
            return result
        return value

    def get(self, value, memo=None):
        if isinstance(value, _LiteralTuple):
            return value.values
        if memo is None:
            memo = {}
        if not isinstance(value, (_ObjectArrayRef, _ArrayRef, dict, tuple, list)):
            return value
        key = id(value)
        if key in memo:
            return memo[key]
        if isinstance(value, _ObjectArrayRef):
            output = np.empty(value.shape, dtype=object)
            for index, item in enumerate(value.values):
                output.flat[index] = self.get(item, memo)
        elif isinstance(value, _ArrayRef):
            if not np.prod(value.shape, dtype=np.int64):
                output = np.empty(value.shape, dtype=value.dtype)
            else:
                if self.buffer is None:
                    self.buffer = np.memmap(self.path, mode='r', dtype=np.uint8)
                output = np.ndarray(value.shape, dtype=value.dtype, buffer=self.buffer, offset=value.offset)
        elif isinstance(value, dict):
            output = {key: self.get(item, memo) for key, item in value.items()}
        else:
            output = type(value)(self.get(item, memo) for item in value)
        memo[key] = output
        return output


class _WriterPool:
    """Bound buffered writers and descriptors while reusing files across tiles."""
    def __init__(self):
        self.limit = 512
        try:
            import resource
            soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft > 0:
                self.limit = min(self.limit, max(16, (soft - 64) // 2))
        except (ImportError, ValueError):
            self.limit = 64
        self.handles = OrderedDict()

    def get(self, path):
        handle = self.handles.pop(path, None)
        if handle is None:
            if len(self.handles) >= self.limit:
                _, oldest = self.handles.popitem(last=False)
                oldest.close()
            handle = path.open('ab', buffering=64 * 1024)
        self.handles[path] = handle
        return handle

    def close_file(self, path):
        handle = self.handles.pop(path, None)
        if handle is not None:
            handle.close()

    def close(self):
        error = None
        while self.handles:
            _, handle = self.handles.popitem()
            try:
                handle.close()
            except OSError as exc:
                error = error or exc
        if error is not None:
            raise error


class _Rows:
    """Append rows without allocating the final array or keeping prior tiles."""
    def __init__(self, path, pool=None):
        self.path = path
        self.pool = pool
        self.shape = None
        self.dtype = None
        self.count = 0
        self.empty = None

    def append(self, value):
        value = np.asarray(value)
        if self.empty is None:
            self.empty = np.empty((0, *value.shape[1:]), dtype=value.dtype)
        if not len(value):
            return
        if value.dtype.hasobject:
            raise ValueError('A numeric tiled result unexpectedly contains object values.')
        if self.shape is None:
            self.shape, self.dtype = value.shape[1:], value.dtype
        if value.shape[1:] != self.shape or value.dtype != self.dtype:
            raise ValueError('Tile array shape or dtype changed during disk aggregation.')
        if value.nbytes:
            data = memoryview(np.ascontiguousarray(value)).cast('B')
            if self.pool is not None:
                self.pool.get(self.path).write(data)
            else:
                with self.path.open('ab') as handle:
                    handle.write(data)
        self.count += len(value)

    def finish(self):
        if self.pool is not None:
            self.pool.close_file(self.path)
        if not self.count:
            return self.empty
        shape = (self.count, *self.shape)
        if not np.prod(shape, dtype=np.int64):
            return np.empty(shape, dtype=self.dtype)
        return np.memmap(self.path, mode='r', dtype=self.dtype, shape=shape)


class _SharedRows:
    """Store each tile array once, then materialize each distinct row history once.

    Templates may share a field on some tiles but diverge on others. Referencing
    every chunk independently preserves that distinction without copying the
    shared prefix or retaining source arrays in RAM.
    """
    def __init__(self, path, store, finished):
        self.path, self.store, self.finished = path, store, finished
        self.chunks = []
        self.shape = self.dtype = None
        self.empty = None

    def append(self, value, memo):
        if value.dtype.hasobject:
            raise ValueError('A numeric tiled result unexpectedly contains object values.')
        if not len(value):
            # No detections may produce (0,) instead of (0, height, width).
            # Zero rows neither constrain nor change the populated row schema.
            if self.empty is None or value.ndim > self.empty.ndim:
                self.empty = np.empty(value.shape, dtype=value.dtype)
            return
        if self.shape is None:
            self.shape, self.dtype = value.shape[1:], value.dtype
        if value.shape[1:] != self.shape or value.dtype != self.dtype:
            raise ValueError(f'Tile field {self.path.stem}: expected rows {self.shape} / {self.dtype}, '
                             f'got {value.shape[1:]} / {value.dtype} during disk aggregation.')
        self.chunks.append(self.store.put(value, memo))

    def finish(self, progress=None):
        if not self.chunks:
            return self.empty
        key = tuple((chunk.offset, chunk.shape, chunk.dtype) for chunk in self.chunks)
        if key not in self.finished:
            rows = _Rows(self.path)
            total = sum(int(np.prod(chunk.shape)) * chunk.dtype.itemsize for chunk in self.chunks)
            done = 0
            for chunk in self.chunks:
                value = self.store.get(chunk)
                # Bound each copy so a large tile cannot hide minutes of I/O
                # behind a single progress update. Slices remain disk-backed.
                row_bytes = max(1, int(np.prod(value.shape[1:])) * value.dtype.itemsize)
                step = max(1, (4 * 1024 * 1024) // row_bytes)
                if not len(value):
                    rows.append(value)
                for start in range(0, len(value), step):
                    block = value[start:start + step]
                    if progress is not None:
                        progress(done, total)
                    rows.append(block)
                    done += block.nbytes
            self.finished[key] = rows.finish()
            if progress is not None:
                progress(done, total)
        return self.finished[key]


class _Picks:
    _lists = {'regions', 'aligned_regions', 'alignment_pose_history'}
    _density = {'density_image', 'density_contrast', 'density_component_labels'}
    _static_arrays = {'template_points_nm', 'alignment_reference_image'}

    def __init__(self, directory, store, pool, finished):
        self.directory, self.store, self.pool = directory, store, pool
        self.values, self.rows = {}, {}
        self.finished = finished
        self.extent = None
        self.grid = None

    def append(self, picks, memo):
        grid = picks.template_points_nm
        if self.grid is None:
            self.grid = np.array(grid, copy=True)
        elif grid.shape != self.grid.shape or not np.allclose(grid, self.grid, rtol=0, atol=1e-9):
            raise ValueError('Tiled origami results must use the same theoretical template.')
        extent = picks.density_extent_nm
        if self.extent is None:
            self.extent = tuple(extent)
        else:
            old = self.extent
            self.extent = (min(old[0], extent[0]), max(old[1], extent[1]),
                           min(old[2], extent[2]), max(old[3], extent[3]))
        for field in fields(OrigamiPickResult):
            name = field.name
            value = getattr(picks, name)
            if name == 'density_extent_nm':
                continue
            if name in self._lists:
                if name == 'alignment_pose_history' and not value:
                    value = [[] for _ in picks.regions]
                self.values.setdefault(name, []).extend(self.store.put(value, memo))
            elif name in self._density:
                self.values[name] = np.empty((0, 0), dtype=np.asarray(value).dtype)
            elif isinstance(value, np.ndarray) and name not in self._static_arrays:
                if name in {'orientation_scores', 'orientation_correlations', 'orientation_flipped', 'orientation_ambiguous'} and len(value) != len(picks.regions):
                    paired = name in {'orientation_scores', 'orientation_correlations'}
                    value = np.full((len(picks.regions), 2) if paired else (len(picks.regions),),
                                    np.nan if paired else False, dtype=float if paired else bool)
                if name == 'footprint_overlap_fraction' and len(value) != len(picks.regions):
                    value = np.zeros(len(picks.regions))
                if name not in self.rows:
                    self.rows[name] = _SharedRows(self.directory / f'{name}.bin', self.store, self.finished)
                self.rows[name].append(value, memo)
            elif (name not in self.values or
                  (name in self._static_arrays and isinstance(value, np.ndarray) and value.size
                   and isinstance(self.values[name], _ArrayRef)
                   and not np.prod(self.values[name].shape))):
                # An initial no-detection tile has no alignment reference image.
                # Take the first available reference from a later populated tile.
                self.values[name] = self.store.put(value, memo)

    def finish(self, status=None, memo=None):
        if status is not None:
            status('Opening shared regions and alignment history')
        values = self.store.get(self.values, memo)
        for name, rows in self.rows.items():
            if status is not None:
                status(f'Opening {name}')
            def copying(done, total, field=name):
                if status is not None:
                    status(f'Copying {field}: {done / 1048576:.1f}/{total / 1048576:.1f} MiB')
            values[name] = rows.finish(progress=copying if status is not None else None)
        values['density_extent_nm'] = self.extent
        return OrigamiPickResult(**values)


class StreamingTileResults:
    """Append one tile, discard its payload, then expose combined disk-backed arrays.

    Cache files are disposable. The atomic per-tile archives remain the recovery
    authority, so interrupted builders are never mistaken for complete results.
    """
    def __init__(self, parent, multi):
        parent = Path(parent)
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix='combined-', dir=parent))
        self.store = _ArrayStore(self.directory)
        self.pool = _WriterPool()
        self.multi = multi
        self.templates = []
        self.finished_rows = {}
        self.names = None
        self.counts = None
        self.unclassified = self.suppressed = 0
        self.details = []
        self.centers = _Rows(self.directory / 'unclassified-centers.bin', self.pool)
        self.tiles = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.pool.close()
        except OSError:
            if exc_type is None:
                raise

    def append(self, payload):
        with self.store.writing():
            self._append(payload)

    def _append(self, payload):
        items = payload['templates'] if self.multi else [dict(name='single', picks=payload['picks'], params={})]
        names = [item['name'] for item in items]
        if self.names is None:
            self.names = names
            for index, item in enumerate(items):
                directory = self.directory / str(index)
                directory.mkdir()
                self.templates.append(dict(picks=_Picks(directory, self.store, self.pool, self.finished_rows), params=None, sequences={}, numeric_sequences={}))
        if names != self.names:
            raise ValueError('Template order changed while aggregating tiled classifications.')
        # IDs are only meaningful within this tile; never retain its arrays.
        memo = {}
        for item, target in zip(items, self.templates):
            target['picks'].append(item['picks'], memo)
            if target['params'] is None:
                target['params'] = self.store.put({k: v for k, v in item['params'].items() if k not in SEQUENCE_PARAMS}, memo)
            for name in SEQUENCE_PARAMS:
                if name in item['params']:
                    sequence = item['params'][name]
                    if name in DISK_SEQUENCE_PARAMS:
                        array = np.ascontiguousarray(sequence, dtype=np.float64)
                        # Independently decoded templates often have equal, but
                        # not identical, probability matrices. Deduplicate those
                        # within this tile before writing any bytes.
                        signature = ('probability-matrix', array.shape,
                                     hashlib.sha256(array.tobytes()).digest())
                        shared = memo.get(signature)
                        if shared is None:
                            memo[signature] = array
                        else:
                            array = shared
                        if name not in target['numeric_sequences']:
                            target['numeric_sequences'][name] = _SharedRows(
                                target['picks'].directory / f'{name}.bin', self.store, self.finished_rows)
                        target['numeric_sequences'][name].append(array, memo)
                        continue
                    # Arrays need row views here; lists/tuples already provide
                    # the sequence and must retain their shared identity.
                    if isinstance(sequence, np.ndarray):
                        sequence = tuple(sequence)
                    # Keep tile chunks shared instead of duplicating all candidate
                    # references in every template's growing list.
                    target['sequences'].setdefault(name, []).append(self.store.put(sequence, memo))
        if self.multi:
            counts = np.asarray(payload['counts'], dtype=int)
            self.counts = counts.copy() if self.counts is None else self.counts + counts
            self.unclassified += int(payload['unclassified_count'])
            self.suppressed += int(payload.get('suppressed_duplicate_count', 0))
            self.centers.append(np.asarray(payload.get('unclassified_centers_nm', np.empty((0, 2))), dtype=float).reshape(-1, 2))
            self.details.extend(self.store.put(payload.get('unclassified_details', []), memo))
        self.tiles += 1

    def finish(self, progress=None, status=None):
        if not self.tiles:
            raise ValueError('No completed tiles to combine.')
        templates = []
        memo = {}
        finished_sequences = {}
        for index, (name, target) in enumerate(zip(self.names, self.templates)):
            def detail(message, current=index):
                record_diagnostic('aggregation_finalize', template=name, index=current, total=len(self.names), stage=message)
                if status is not None:
                    status(current, len(self.names), message)
            detail('Opening classification parameters')
            params = self.store.get(target['params'], memo)
            for key, chunks in target['sequences'].items():
                detail(f'Opening {key}')
                # Chunk identities encode the complete history, including tiles
                # on which templates diverged. Share only identical histories.
                history = tuple(id(chunk) for chunk in chunks)
                if history not in finished_sequences:
                    finished_sequences[history] = tuple(
                        item for chunk in chunks for item in self.store.get(chunk, memo))
                params[key] = finished_sequences[history]
            for key, rows in target['numeric_sequences'].items():
                detail(f'Opening disk-backed {key}')
                params[key] = rows.finish()
            templates.append(dict(name=name, picks=target['picks'].finish(status=detail, memo=memo), params=params))
            if progress is not None:
                progress(len(templates), len(self.names))
        if not self.multi:
            return dict(picks=templates[0]['picks'])
        return dict(templates=templates, counts=self.counts, unclassified_count=self.unclassified,
                    suppressed_duplicate_count=self.suppressed, unclassified_centers_nm=self.centers.finish(),
                    unclassified_details=self.store.get(self.details, memo))

    def source_table(self, locs, index_chunks, progress=None):
        """Gather selected rows column-by-column without copying the full table.

        Numeric columns and the index remain disk-backed. Object/extension
        columns preserve their pandas dtype, but necessarily use memory.
        """
        def gather(series, name):
            values = series.to_numpy(copy=False)
            numeric = not values.dtype.hasobject and not isinstance(series.dtype, pd.api.extensions.ExtensionDtype)
            rows = _Rows(self.directory / name) if numeric else None
            parts = []
            for indices in index_chunks:
                for start in range(0, len(indices), 100_000):
                    take = indices[start:start + 100_000]
                    if numeric:
                        rows.append(values[take])
                    else:
                        parts.append(series.iloc[take].reset_index(drop=True))
            if numeric:
                if rows.empty is None:
                    rows.append(values[:0])
                return np.asarray(rows.finish())
            return pd.concat(parts, ignore_index=True) if parts else series.iloc[:0].reset_index(drop=True)

        columns = {}
        for index, name in enumerate(locs.columns):
            columns[index] = gather(locs[name], f'source-column-{index}.bin')
            if progress is not None:
                progress(index + 1, len(locs.columns) + 1)
        result = pd.DataFrame(columns, copy=False)
        result.columns = locs.columns
        result.index = pd.Index(gather(pd.Series(locs.index, copy=False), 'source-index.bin'), name=locs.index.name)
        if progress is not None:
            progress(len(locs.columns) + 1, len(locs.columns) + 1)
        return result

    def tile_index(self, locs, pixelsize, lattice, progress=None, chunk_size=100_000):
        """Bucket rows with bounded buffers, then assemble the index sequentially."""
        xs = np.unique([tile[0] for tile in lattice])
        ys = np.unique([tile[2] for tile in lattice])
        xmax = max(tile[1] for tile in lattice)
        ymax = max(tile[3] for tile in lattice)
        count = len(xs) * len(ys)

        def chunks():
            for start in range(0, len(locs), chunk_size):
                block = locs.iloc[start:start + chunk_size]
                x = block['x'].to_numpy(dtype=float) * pixelsize
                y = block['y'].to_numpy(dtype=float) * pixelsize
                tx = np.searchsorted(xs, x, side='right') - 1
                ty = np.searchsorted(ys, y, side='right') - 1
                valid = (np.isfinite(x) & np.isfinite(y) & (tx >= 0) & (ty >= 0)
                         & (x < xmax) & (y < ymax))
                yield start, ty[valid] * len(xs) + tx[valid], np.flatnonzero(valid) + start

        counts = np.zeros(count, dtype=np.int64)
        # Scatter writes through a large writable mmap cause page faults and
        # seeks on external disks. Small append-only buckets bound both memory
        # and open descriptors, then become one sequential, read-only index.
        buffer_bytes = max(8, min(256 * 1024, (32 * 1024 * 1024) // count))
        buffers = [bytearray() for _ in range(count)]
        path = self.directory / 'tile-source-index.bin'
        with tempfile.TemporaryDirectory(prefix='tile-index-', dir=self.directory) as temporary:
            paths = [Path(temporary) / f'{tile}.bin' for tile in range(count)]

            def flush(tile):
                if buffers[tile]:
                    with paths[tile].open('ab') as stream:
                        stream.write(buffers[tile])
                    buffers[tile].clear()

            for start, ids, positions in chunks():
                order = np.argsort(ids, kind='stable')
                ids, positions = ids[order], positions[order]
                boundaries = np.r_[0, np.flatnonzero(np.diff(ids)) + 1, len(ids)]
                for left, right in zip(boundaries[:-1], boundaries[1:]):
                    if right == left:
                        continue
                    tile = ids[left]
                    buffers[tile].extend(positions[left:right].astype(np.int64, copy=False).tobytes())
                    counts[tile] += right - left
                    if len(buffers[tile]) >= buffer_bytes:
                        flush(tile)
                if progress is not None:
                    progress(min(start + chunk_size, len(locs)), 2 * len(locs))
            offsets = np.concatenate(([0], np.cumsum(counts)))
            copied = 0
            with path.open('wb') as output:
                for tile in range(count):
                    flush(tile)
                    if not counts[tile]:
                        continue
                    with paths[tile].open('rb') as source:
                        while block := source.read(4 * 1024 * 1024):
                            output.write(block)
                            copied += len(block) // 8
                            if progress is not None:
                                progress(len(locs) + copied, 2 * len(locs))
                    paths[tile].unlink()
        if progress is not None:
            progress(2 * len(locs), 2 * len(locs))
        if not offsets[-1]:
            return np.empty(0, dtype=np.int64), offsets
        return np.memmap(path, mode='r', dtype=np.int64, shape=(int(offsets[-1]),)), offsets

    def source_points(self, locs, pixelsize, progress=None):
        rows = _Rows(self.directory / 'source-points.bin', self.pool)
        for start in range(0, len(locs), 100_000):
            rows.append(locs.iloc[start:start + 100_000][['x', 'y']].to_numpy(dtype=float) * pixelsize)
            if progress is not None:
                progress(min(start + 100_000, len(locs)), len(locs))
        return rows.finish()
