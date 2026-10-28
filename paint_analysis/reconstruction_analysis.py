"""Selective reconstruction reader for analysis archives and tiled run snapshots.

Navigate serialized references before decoding: ignored branches are never opened.
Numeric arrays reuse the normal analysis cache without loading DataFrames or pick
records. Only accepted poses and the small geometry parameters survive decoding.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np

from origami_composite import _progress_heartbeat


class ReconstructionArchive:
    def __init__(self, path, progress=None, preload=False):
        self.path = Path(path)
        self.progress = progress or (lambda text: None)
        self._buffer = None
        self.archive = None
        try:
            # Small tiles contain hundreds of tiny pose arrays scattered through
            # the ZIP. One sequential read avoids a seek for each on HDDs/USB.
            size = self.path.stat().st_size
            if preload and size <= 64 * 1024 * 1024:
                self._buffer = io.BytesIO()
                self.progress(f'Reading tile sequentially: {self.path.name} (0/{size:,} bytes)')
                with self.path.open('rb') as stream:
                    while block := stream.read(1024 * 1024):
                        self._buffer.write(block)
                        self.progress(f'Reading tile sequentially: {self.path.name} '
                                      f'({self._buffer.tell():,}/{size:,} bytes)')
                self._buffer.seek(0)
            self.progress(f'Opening archive directory: {self.path.name}…')
            self.archive = ZipFile(self._buffer if self._buffer is not None else path)
            self.progress(f'Decoding reconstruction metadata: {self.path.name}…')
            manifest = json.loads(self.archive.read('manifest.json'))
            if manifest.get('format') != 'paint-analysis-session' or manifest.get('version') != 1:
                raise ValueError('Unsupported analysis file format or version.')
            self.nodes, self.root = manifest['nodes'], manifest['root']
            stat = self.path.stat()
            identity = hashlib.sha256(f'{self.path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()[:24]
            self.directory = self.path.parent / '.analysis-cache' / identity
            self.cache = {}
            self.active = set()
        except Exception:
            if self.archive is not None:
                self.archive.close()
            if self._buffer is not None:
                self._buffer.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.archive.close()
        if self._buffer is not None:
            self._buffer.close()
        self.cache.clear()
        self.nodes.clear()

    def node(self, ref):
        if not isinstance(ref, dict) or set(ref) != {'ref'}:
            raise ValueError('Invalid analysis reference.')
        index = ref['ref']
        if not isinstance(index, int) or index < 0 or index >= len(self.nodes):
            raise ValueError('Invalid analysis node index.')
        return self.nodes[index]

    def fields(self, ref):
        node = self.node(ref)
        if node['kind'] == 'dict':
            return {self.decode(key): value for key, value in node['values']}
        if node['kind'] == 'record':
            return node['values']
        raise ValueError('Expected an analysis dictionary or record.')

    def items(self, ref):
        node = self.node(ref)
        if node['kind'] not in ('list', 'tuple'):
            raise ValueError('Expected an analysis sequence.')
        return node['values']

    def value(self, fields, key, default=None):
        return self.decode(fields[key]) if key in fields else default

    def decode(self, ref):
        if not isinstance(ref, dict):
            return ref
        index = ref['ref']
        if index in self.cache:
            return self.cache[index]
        if index in self.active:
            raise ValueError('Cyclic analysis metadata is not supported.')
        self.active.add(index)
        try:
            node = self.node(ref)
            kind = node['kind']
            if kind == 'array':
                member = self.archive.getinfo(node['file'])
                if member.file_size < 1024 * 1024:
                    # Never create one cache file per tiny mask/geometry array.
                    # The per-reader memo still preserves sharing across templates.
                    result = np.load(io.BytesIO(self.archive.read(member)), allow_pickle=False)
                    self.cache[index] = result
                    return result
                self.directory.mkdir(parents=True, exist_ok=True)
                path = self.directory / f'array-{index}.npy'
                if not path.exists() or path.stat().st_size != member.file_size:
                    temporary = None
                    try:
                        with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as target:
                            temporary = Path(target.name)
                            with self.archive.open(member) as stream:
                                done, last_percent = 0, -1
                                while block := stream.read(1024 * 1024):
                                    target.write(block)
                                    done += len(block)
                                    percent = int(100 * done / member.file_size)
                                    if member.file_size >= 1024 * 1024 and percent != last_percent:
                                        self.progress(f'Reading required array from {self.path.name}: '
                                                      f'{done:,}/{member.file_size:,} bytes ({percent}%)')
                                        last_percent = percent
                        os.replace(temporary, path)
                    finally:
                        if temporary is not None:
                            temporary.unlink(missing_ok=True)
                elif member.file_size >= 1024 * 1024:
                    self.progress(f'Reusing cached reconstruction array from {self.path.name}…')
                result = np.load(path, mmap_mode='r', allow_pickle=False)
            elif kind == 'path':
                result = Path(node['value'])
            elif kind == 'dict':
                result = {self.decode(k): self.decode(v) for k, v in node['values']}
            elif kind in ('tuple', 'list', 'set'):
                result = {'tuple': tuple, 'list': list, 'set': set}[kind](self.decode(v) for v in node['values'])
            else:
                raise ValueError(f'Unexpected reconstruction metadata node: {kind}')
            self.cache[index] = result
            return result
        finally:
            self.active.remove(index)

    def params(self, ref):
        fields = self.fields(ref)
        keys = ('rows', 'columns', 'spacing_x_nm', 'spacing_y_nm', 'mirror_all_templates',
                'mirror_template_y', 'column_offsets_nm', 'alignment_template_overlay_points_nm',
                '_checkpoint_preview', '_reconstruction_grid_nm', '_reconstruction_exclusions_nm')
        params = {key: self.decode(fields[key]) for key in keys if key in fields}
        for key in ('logical_model', 'digital_pixel_model', 'shared_alignment_template'):
            if key not in fields or fields[key] is None:
                continue
            nested = self.fields(fields[key])
            allowed = ('column_offsets_nm', 'bit_physical_cells') if key != 'shared_alignment_template' else (
                'rows', 'columns', 'spacing_x_nm', 'spacing_y_nm', 'column_offsets_nm',
                'overlay_points_nm', 'rectangle_margin_nm', 'template_pixel_size_x_nm', 'template_pixel_size_y_nm')
            params[key] = {name: self.decode(nested[name]) for name in allowed if name in nested}
            if (key == 'shared_alignment_template' and 'image' in nested
                    and not np.asarray(params.get('alignment_template_overlay_points_nm', ())).size
                    and not np.asarray(params[key].get('overlay_points_nm', ())).size):
                # Old snapshots may need the small Step 2 image to recover marker sites.
                params[key]['image'] = self.decode(nested['image'])
        return params

    def picks(self, ref, offset=0):
        fields = self.fields(ref)
        mask = np.asarray(self.decode(fields['accepted_mask']))
        corners = self.decode(fields['rectangle_corners_nm'])
        if mask.ndim != 1 or mask.dtype.kind != 'b' or corners.shape != (len(mask), 4, 2):
            raise ValueError('Inconsistent saved origami poses or acceptance mask.')
        accepted = np.flatnonzero(mask)
        selected = np.asarray(corners[accepted], dtype=float)
        if not np.all(np.isfinite(selected)):
            raise ValueError('Nonfinite accepted origami poses.')
        indices = (np.asarray(self.decode(fields['candidate_indices']))[accepted]
                   if 'candidate_indices' in fields else accepted + offset)
        return SimpleNamespace(accepted_mask=np.ones(len(accepted), dtype=bool),
                               rectangle_corners_nm=selected, candidate_indices=indices), len(mask)


def _owner(path, source, results, configuration, unclassified, source_path):
    if not results:
        raise ValueError('No saved barcode classifications found. Run Step 4 before loading reconstruction.')
    if not isinstance(source, np.ndarray) or source.ndim != 2 or source.shape[1] != 2 or source.dtype.kind not in 'fiu':
        raise ValueError('Invalid saved source coordinates.')
    return SimpleNamespace(origami_source_points_nm=source, origami_multi_template_results=results,
                           origami_custom_templates=[], origami_composite_configuration=configuration,
                           origami_multi_template_unclassified_count=unclassified,
                           origami_loaded_source_path=str(source_path), origami_identification_running=False,
                           origami_identification_baseline=None, reconstruction_path=str(Path(path).resolve()),
                           reconstruction_region=None)


def load_reconstruction_from_analysis(path, progress=None, filter_series=None):
    """Read just reconstruction inputs; never install/restore a full analysis."""
    path = Path(path)
    with _progress_heartbeat(progress) as report, ReconstructionArchive(path, report) as reader:
        root = reader.fields(reader.root)
        if 'checkpoint_version' in root:
            return _load_run(reader, root, report, filter_series)
        if {'run_id', 'index', 'bounds', 'payload'}.issubset(root):
            raise ValueError('Choose run.paintanalysis from this folder, not an individual tile file.')
        if 'state' not in root:
            raise ValueError('This archive does not contain a saved analysis or tiled run.')
        state = reader.fields(root['state'])
        result_refs = reader.fields(state['origami_multi_template_results']) if 'origami_multi_template_results' in state else {}
        results = {}
        for number, (name, ref) in enumerate(result_refs.items(), 1):
            if report:
                report(f'Loading reconstruction poses: {name} ({number}/{len(result_refs)} templates)')
            fields = reader.fields(ref)
            picks, _ = reader.picks(fields['picks'])
            results[name] = dict(picks=picks, params=reader.params(fields['params']))
        if not results:
            raise ValueError('No saved barcode classifications found. Run Step 4 first.')
        if report:
            report('Opening source coordinates only; skipping localization tables and classification intermediates…')
        if 'origami_source_points_nm' not in state:
            raise ValueError('Analysis is missing the source coordinates required for reconstruction.')
        source = reader.decode(state['origami_source_points_nm'])
        return _owner(path, source, results, reader.value(state, 'origami_composite_configuration'),
                      reader.value(state, 'origami_multi_template_unclassified_count', 0),
                      reader.value(state, 'origami_loaded_source_path', path))


def _load_run(reader, root, report, filter_series):
    if reader.value(root, 'checkpoint_version') != 1:
        raise ValueError('Unsupported tiled checkpoint version.')
    run_id = reader.value(root, 'run_id')
    context = reader.fields(root['context'])
    lattice = reader.value(context, 'tile_lattice')
    planned = np.asarray(reader.value(context, 'tile_indices'), dtype=int)
    if planned.ndim != 1 or len(set(planned.tolist())) != len(planned):
        raise ValueError('Invalid or duplicate planned tile indices.')
    results, offsets, chunks, completed = {}, {}, {}, []
    unclassified = 0
    for sequence, index in enumerate(planned):
        if index < 0 or index >= len(lattice):
            raise ValueError('Invalid planned tile index.')
        path = reader.path.parent / f'tile-{index:08d}.paintanalysis'
        if not path.exists():
            continue
        if report:
            report(f'Loading reconstruction from completed tile {sequence + 1}/{len(planned)}…')
        with ReconstructionArchive(path, report, preload=True) as tile:
            fields = tile.fields(tile.root)
            if (tile.value(fields, 'run_id') != run_id or tile.value(fields, 'index') != index
                    or tuple(tile.value(fields, 'bounds')) != tuple(lattice[index])):
                raise ValueError(f'{path.name} does not match the selected run or tile geometry.')
            if tile.value(fields, 'mode') != 'multi':
                raise ValueError('Reconstruction requires saved multi-template barcode classifications (Step 4).')
            payload = tile.fields(fields['payload'])
            expected_names = set(results)
            current_names = []
            template_refs = tile.items(payload['templates'])
            for template_index, ref in enumerate(template_refs, 1):
                if report and (template_index == 1 or template_index % 20 == 0 or template_index == len(template_refs)):
                    report(f'Reading accepted poses: {path.name}, template {template_index}/{len(template_refs)}')
                item = tile.fields(ref)
                name = tile.value(item, 'name')
                current_names.append(name)
                picks, count = tile.picks(item['picks'], offsets.get(name, 0))
                offsets[name] = offsets.get(name, 0) + count
                params = tile.params(item['params'])
                if name not in results:
                    results[name] = dict(params=params)
                    chunks[name] = []
                # The run uses identical template geometry for all tiles.
                elif not _same_geometry(results[name]['params'], params):
                    raise ValueError(f'Template geometry changed between saved tiles: {name}.')
                chunks[name].append(picks)
            if len(current_names) != len(set(current_names)) or (completed and set(current_names) != expected_names):
                raise ValueError('Template identities changed between saved tiles.')
            unclassified += int(tile.value(payload, 'unclassified_count', 0))
        completed.append(int(index))
    if not completed:
        raise ValueError('No completed barcode-classification tiles are available in this run folder.')
    if report:
        report('Combining accepted reconstruction poses…')
    for name, result in results.items():
        parts = chunks[name]
        corners = np.concatenate([p.rectangle_corners_nm for p in parts])
        indices = np.concatenate([p.candidate_indices for p in parts])
        result['picks'] = SimpleNamespace(rectangle_corners_nm=corners, candidate_indices=indices,
                                          accepted_mask=np.ones(len(corners), dtype=bool))
        result['params']['_checkpoint_preview'] = dict(completed=len(completed), planned=len(planned))
    source = _run_source(reader, context, lattice, planned, completed, report, filter_series)
    analysis = reader.fields(root['analysis'])
    state = reader.fields(analysis['state'])
    return _owner(reader.path, source, results, reader.value(state, 'origami_composite_configuration'), unclassified,
                  reader.value(state, 'origami_loaded_source_path', reader.path))


def _same_geometry(left, right):
    # Compare small metadata directly; do not expand a legacy alignment image
    # into millions of Python scalars for every template in every tile.
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) - {'_checkpoint_preview'}
        return keys == set(right) - {'_checkpoint_preview'} and all(_same_geometry(left[k], right[k]) for k in keys)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same_geometry(a, b) for a, b in zip(left, right))
    return left == right


def _run_source(reader, context, lattice, planned, completed, report, filter_series):
    """Extract XY plus columns needed to honor the original source filters."""
    fingerprint = hashlib.sha256(json.dumps(completed).encode()).hexdigest()[:16]
    destination = reader.directory / f'reconstruction-xy-v1-{fingerprint}.npy'
    if destination.exists():
        if report:
            report('Reusing cached source coordinates for the same completed tiles…')
        source = np.load(destination, mmap_mode='r', allow_pickle=False)
        if source.ndim != 2 or source.shape[1] != 2 or source.dtype != np.dtype(float):
            raise ValueError('Invalid reconstruction coordinate cache; remove it and retry.')
        return source
    params = reader.value(context, 'source_params')
    pixelsize = float(reader.value(context, 'pixelsize'))
    if not np.isfinite(pixelsize) or pixelsize <= 0:
        raise ValueError('Invalid saved localization pixel size.')
    table = reader.node(context['source_locs'])
    if table['kind'] != 'dataframe':
        raise ValueError('Missing saved source localization columns.')
    columns = dict(zip(reader.decode(table['columns']), table['values']))
    filters = params.get('active_filters', ()) if str(params.get('source', '')).lower().startswith('filtered') else ()
    needed = {'x', 'y'}
    dependencies = {'precision_radial_nm': {'lpx', 'lpy'}, 'lpx_nm': {'lpx'}, 'lpy_nm': {'lpy'},
                    'localizations_per_frame': {'frame'}, 'nearest_neighbor_nm': {'x', 'y'}}
    for mode, _ in filters:
        if mode.startswith('event_') and mode not in columns:
            needed.update({'frame', 'x', 'y'})
            if 'photons' in columns:
                needed.add('photons')
        else:
            needed.update(dependencies.get(mode, {mode}))
    if needed - columns.keys():
        raise ValueError('Missing saved source columns: ' + ', '.join(sorted(needed - columns.keys())))
    arrays = {key: reader.decode(columns[key]) for key in sorted(needed)}
    n = len(arrays['x'])
    if any(value.shape != (n,) for value in arrays.values()):
        raise ValueError('Inconsistent source coordinate columns.')
    keep = None
    if filters:
        if filter_series is None:
            raise ValueError('Source filters require the application filter evaluator.')
        import pandas as pd
        frame = pd.DataFrame(arrays, copy=False)
        keep = np.ones(n, dtype=bool)
        for mode, (left, right) in filters:
            if report:
                report(f'Applying saved source filter: {mode}…')
            values, _ = filter_series(frame, mode, pixelsize, float(params['exposure_ms']),
                                      float(params['link_radius_nm']), int(params['max_gap_frames']))
            values = np.asarray(values, dtype=float)
            keep &= np.isfinite(values) & (values >= min(left, right)) & (values <= max(left, right))
    lattice = np.asarray(lattice, dtype=float)
    x_starts, y_starts = np.unique(lattice[:, 0]), np.unique(lattice[:, 2])
    if len(x_starts) * len(y_starts) != len(lattice):
        raise ValueError('Unsupported irregular tile lattice.')
    if not np.array_equal(lattice[:, [0, 2]], np.array([(x, y) for y in y_starts for x in x_starts])):
        raise ValueError('Unexpected tile ordering.')
    available = np.zeros(len(lattice), dtype=bool)
    available[completed] = True
    nonempty = np.zeros(len(lattice), dtype=bool)

    def block(start):
        stop = min(start + 100_000, n)
        xy = np.column_stack((arrays['x'][start:stop], arrays['y'][start:stop])).astype(float) * pixelsize
        tx = np.searchsorted(x_starts, xy[:, 0], side='right') - 1
        ty = np.searchsorted(y_starts, xy[:, 1], side='right') - 1
        valid = (np.all(np.isfinite(xy), axis=1) & (tx >= 0) & (ty >= 0)
                 & (xy[:, 0] < lattice[:, 1].max()) & (xy[:, 1] < lattice[:, 3].max()))
        selected = np.ones(len(xy), dtype=bool) if keep is None else keep[start:stop]
        ids = np.clip(ty * len(x_starts) + tx, 0, len(lattice) - 1)
        return xy, ids, valid & selected, selected

    count = 0
    for start in range(0, n, 100_000):
        xy, ids, valid, selected = block(start)
        nonempty[ids[valid]] = True
        count += np.count_nonzero(valid & available[ids])
        if report:
            report(f'Selecting source coordinates from completed tiles: {min(start + 100_000, n):,}/{n:,}')
    # Match the full-analysis preview: include full-field context only when all
    # lattice tiles are completed or empty; partial runs include completed tiles.
    full = len(planned) == len(lattice) and not np.any(nonempty & ~available)
    if full:
        count = n if keep is None else int(np.count_nonzero(keep))
    reader.directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=reader.directory, suffix='.npy')
    os.close(fd)
    try:
        out = np.lib.format.open_memmap(temporary, mode='w+', dtype=float, shape=(int(count), 2))
        cursor = 0
        for start in range(0, n, 100_000):
            xy, ids, valid, selected = block(start)
            selected = selected if full else valid & available[ids]
            points = xy[selected]
            out[cursor:cursor + len(points)] = points
            cursor += len(points)
            if report:
                report(f'Writing reconstruction source coordinates: {cursor:,}/{count:,}')
        out.flush()
        del out
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return np.load(destination, mmap_mode='r', allow_pickle=False)
