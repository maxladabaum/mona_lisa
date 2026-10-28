"""Standalone reconstruction archives for a fixed image extraction region.

ZIP_STORED keeps the source NPY contiguous: opening a session reads metadata and
accepted poses, never inflates or copies the acquisition. Source coordinates are
restricted to the union of accepted image-region circumscribed circles. No pickle payloads.
"""
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
from zipfile import ZipFile, ZIP_STORED

import numpy as np

from origami_composite import CompositeSettings, _progress_heartbeat, _reconstruction_source, reconstruction_mapping_issues

FORMAT = 'paint-composite-session'
VERSION = 1
REGION_KEYS = ('left_nm', 'top_nm', 'width_nm', 'height_nm', 'blend_width_nm', 'blend_height_nm')


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported reconstruction metadata: {type(value).__name__}')


def _validate_configuration(configuration, results):
    if configuration.get('format') != 'paint-composite-mappings-v1':
        raise ValueError('Invalid reconstruction mapping configuration.')
    CompositeSettings(**configuration['settings']).validate()
    if not (configuration.get('rgb_batch_manifest') or {}).get('designs'):
        raise ValueError('Load the saved RGB picklist batch before saving a reconstruction session.')
    issues = reconstruction_mapping_issues(results, configuration['mappings'])
    if issues:
        raise ValueError('Check reconstruction mappings: ' + '; '.join(f'{n}: {m}' for n, m in issues))


def save_reconstruction_session(path, source, results, configuration, exclusions, grids,
                                source_path='', unclassified_count=0, progress=None):
    """Atomically save fixed-region coordinates and the inputs required to rebuild."""
    _validate_configuration(configuration, results)
    if not results or not any(np.any(result['picks'].accepted_mask) for result in results.values()):
        raise ValueError('Classify and accept origami before saving a reconstruction session.')
    path = Path(path)
    # Replacing a mapped file is platform dependent; use a fresh destination.
    mapped_path = getattr(source, 'filename', None)
    if mapped_path and Path(mapped_path).resolve() == path.resolve():
        raise ValueError('Choose a new filename when saving an open reconstruction session.')
    source = np.asarray(source)
    if source.ndim != 2 or source.shape[1] != 2 or source.dtype.kind not in 'fiu':
        raise ValueError('Source coordinates must be a nonempty numeric N × 2 array.')
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        with _progress_heartbeat(progress) as report, ZipFile(temporary, 'w', compression=ZIP_STORED, allowZip64=True) as archive:
            settings = CompositeSettings(**configuration['settings'])
            source = _reconstruction_source(source, results, list(results), settings, report)
            records = []
            for number, (name, result) in enumerate(results.items()):
                if report:
                    report(f'Saving identified origami: {number + 1}/{len(results)} templates ({name})')
                picks = result['picks']
                accepted = np.flatnonzero(picks.accepted_mask)
                corners = np.asarray(picks.rectangle_corners_nm)[accepted]
                if corners.shape != (len(accepted), 4, 2) or not np.all(np.isfinite(corners)):
                    raise ValueError(f'Invalid saved alignment footprints for {name}.')
                indices = getattr(picks, 'candidate_indices', np.arange(len(picks.accepted_mask)))[accepted]
                for key, array in (('corners', corners), ('indices', indices)):
                    with archive.open(f'{number}_{key}.npy', 'w', force_zip64=True) as stream:
                        np.save(stream, array, allow_pickle=False)
                params = result.get('params', {})
                # Omit images, candidate crops, probability matrices and classification state.
                small_params = {key: params[key] for key in (
                    'rows', 'columns', 'spacing_x_nm', 'spacing_y_nm', 'mirror_all_templates',
                    'mirror_template_y', '_checkpoint_preview') if key in params}
                small_params['_reconstruction_grid_nm'] = np.asarray(grids[name]).tolist()
                small_params['_reconstruction_exclusions_nm'] = np.asarray(exclusions.get(name, ())).reshape(-1, 2).tolist()
                records.append(dict(name=name, params=small_params))
            if report:
                report(f'Saving source coordinates: 0/{len(source):,} (0%)')
            with archive.open('source.npy', 'w', force_zip64=True) as stream:
                np.lib.format.write_array_header_2_0(stream, dict(descr=np.lib.format.dtype_to_descr(source.dtype),
                                                                fortran_order=False, shape=source.shape))
                for start in range(0, len(source), 100_000):
                    stop = min(start + 100_000, len(source))
                    block = np.ascontiguousarray(source[start:stop])
                    if not np.all(np.isfinite(block)):
                        raise ValueError('Source coordinates must be finite.')
                    stream.write(block.tobytes())
                    if report:
                        report(f'Saving source coordinates: {stop:,}/{len(source):,} ({stop / len(source):.0%})')
            if report:
                report('Saving embedded RGB batch manifest and reconstruction settings…')
            document = dict(format=FORMAT, version=VERSION, templates=records, configuration=configuration,
                            source_path=str(source_path), unclassified_count=int(unclassified_count),
                            extraction_region={key: getattr(settings, key) for key in REGION_KEYS})
            archive.writestr('reconstruction.json', json.dumps(document, default=_json_value, allow_nan=False))
        os.replace(temporary, path)
        if progress:
            progress('Reconstruction session saved.')
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _mapped_source(path, info):
    """Locate a stored NPY payload using actual ZIP header lengths (including ZIP64)."""
    if info.compress_type != ZIP_STORED or info.flag_bits & 1:
        raise ValueError('Reconstruction source must be stored without compression or encryption.')
    with open(path, 'rb') as stream:
        stream.seek(info.header_offset)
        header = stream.read(30)
        signature, name_size, extra_size = struct.unpack('<4s22xHH', header)
        if signature != b'PK\x03\x04':
            raise ValueError('Invalid reconstruction source header.')
        stream.seek(name_size + extra_size, 1)
        start = stream.tell()
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError('Unsupported source array format.')
        offset = stream.tell()
        if len(shape) != 2 or shape[1] != 2 or shape[0] < 0 or dtype.kind not in 'fiu' or fortran:
            raise ValueError('Invalid reconstruction source array.')
        if offset - start + shape[0] * shape[1] * dtype.itemsize != info.file_size:
            raise ValueError('Truncated or inconsistent reconstruction source array.')
    if not shape[0]:
        return np.empty(shape, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape)


def load_reconstruction_session(path, progress=None):
    """Load a reconstruction-only data owner, leaving the full analysis untouched."""
    with _progress_heartbeat(progress) as report, ZipFile(path) as archive:
        if report:
            report('Loading reconstruction settings and embedded RGB batch manifest…')
        document = json.loads(archive.read('reconstruction.json'))
        if document.get('format') != FORMAT or document.get('version') != VERSION:
            raise ValueError('Unsupported reconstruction session format or version.')
        results = {}
        records = document['templates']
        for number, record in enumerate(records):
            name = record['name']
            if not isinstance(name, str) or name in results:
                raise ValueError('Invalid or duplicate template name in reconstruction session.')
            if report:
                report(f'Loading identified origami: {number + 1}/{len(records)} templates ({name})')
            with archive.open(f'{number}_corners.npy') as stream:
                corners = np.load(stream, allow_pickle=False)
            with archive.open(f'{number}_indices.npy') as stream:
                indices = np.load(stream, allow_pickle=False)
            if (corners.shape != (len(indices), 4, 2) or not np.all(np.isfinite(corners))
                    or indices.ndim != 1 or indices.dtype.kind not in 'iu' or np.any(indices < 0)):
                raise ValueError(f'Invalid identified origami for {name}.')
            params = record['params']
            for key in ('_reconstruction_grid_nm', '_reconstruction_exclusions_nm'):
                points = np.asarray(params[key], dtype=float)
                if points.size and (points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points))):
                    raise ValueError(f'Invalid saved template geometry for {name}.')
                if key == '_reconstruction_grid_nm' and not points.size:
                    raise ValueError(f'Missing saved template grid for {name}.')
            results[name] = dict(params=params, picks=SimpleNamespace(
                rectangle_corners_nm=corners, accepted_mask=np.ones(len(corners), dtype=bool), candidate_indices=indices))
        if not results:
            raise ValueError('Reconstruction session contains no identified templates.')
        configuration = document['configuration']
        _validate_configuration(configuration, results)
        region = document['extraction_region']
        saved_settings = CompositeSettings(**configuration['settings'])
        expected_region = {key: getattr(saved_settings, key) for key in REGION_KEYS}
        # Older sessions captured only the normal rectangle; never invent extra source coverage.
        region = dict(region)
        for key in ('blend_width_nm', 'blend_height_nm'):
            region.setdefault(key, 0.0)
        if region != expected_region:
            raise ValueError('Saved image region does not match reconstruction settings.')
        if report:
            report('Opening source coordinates without reading the full acquisition…')
        source = _mapped_source(path, archive.getinfo('source.npy'))
        owner = SimpleNamespace(origami_multi_template_results=results, origami_source_points_nm=source,
                                origami_custom_templates=[], origami_composite_configuration=configuration,
                                origami_multi_template_unclassified_count=document['unclassified_count'],
                                origami_loaded_source_path=document['source_path'], origami_identification_running=False,
                                origami_identification_baseline=None, reconstruction_path=str(Path(path).resolve()), reconstruction_region=region)
    if progress:
        progress('Reconstruction session loaded; saved RGB mappings restored.')
    return owner
