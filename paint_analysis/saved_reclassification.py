"""Classify a saved unclassified subset and merge it without changing old assignments."""
from dataclasses import fields, replace

import numpy as np

from tiled_aggregation import SEQUENCE_PARAMS


STATIC_PICK_FIELDS = {
    'template_points_nm', 'density_image', 'density_contrast',
    'density_component_labels', 'density_extent_nm', 'alignment_reference_image',
}


def unclassified_subset(picks, params, previous):
    if picks is None or not previous:
        raise ValueError('Load a completed classification before classifying unclassified particles.')
    n = len(picks.regions)
    dispositions = params.get('classification_dispositions', ())
    if len(dispositions) != n:
        raise ValueError('Saved unclassified labels are missing or stale. Load a completed analysis.')
    mask = np.array([str(value).startswith('unclassified:') for value in dispositions])
    for result in previous.values():
        old = result['picks']
        if len(old.regions) != n or not np.array_equal(old.bounds_nm, picks.bounds_nm):
            raise ValueError('Saved classes do not share candidate ordering; cannot safely update this analysis.')
        mask &= ~np.asarray(old.accepted_mask, dtype=bool)
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError('There are no previously unclassified particles to classify.')
    updates = {}
    for field in fields(picks):
        value = getattr(picks, field.name)
        if field.name in STATIC_PICK_FIELDS:
            continue
        if isinstance(value, np.ndarray) and value.ndim and len(value) == n:
            updates[field.name] = value[indices]
        elif isinstance(value, list) and len(value) == n:
            updates[field.name] = [value[i] for i in indices]
    subset = replace(picks, **updates)
    sliced = dict(params)
    for key in SEQUENCE_PARAMS:
        value = sliced.get(key)
        if value is not None and len(value) == n:
            sliced[key] = (value[indices] if isinstance(value, np.ndarray)
                           else tuple(value[i] for i in indices))
    return indices, subset, sliced


def merge_unclassified_results(payload, indices, original_picks, original_params, previous):
    """Scatter only newly classified rows into the saved full candidate collection."""
    n = len(original_picks.regions)
    merged = {name: {'name': name, 'picks': result['picks'], 'params': dict(result['params'])}
              for name, result in previous.items()}
    all_templates = {t['name']: t for t in original_params.get('custom_templates', ())}
    all_templates.update({t['name']: t for t in payload['params'].get('custom_templates', ())})
    dispositions = list(original_params['classification_dispositions'])
    new_dispositions = payload['templates'][0]['params']['classification_dispositions']
    for i, value in zip(indices, new_dispositions):
        dispositions[i] = value
    for result in payload['templates']:
        name = result['name']
        old = merged.get(name)
        params = dict(old['params'] if old else original_params)
        for key, value in result['params'].items():
            if key in SEQUENCE_PARAMS or key.startswith('_pre') or key == '_reuse_saved_measurements':
                continue
            params[key] = value
        for key in SEQUENCE_PARAMS:
            values = result['params'].get(key)
            if values is None or len(values) != len(indices):
                continue
            existing = params.get(key)
            if existing is not None and len(existing) == n and (old or not key.startswith('classification_')):
                expanded = list(existing)
            else:
                sample = values[0]
                if isinstance(sample, (tuple, list, np.ndarray)):
                    blank = tuple(0.0 for _ in sample)
                elif isinstance(sample, str):
                    blank = ''
                elif isinstance(sample, (bool, np.bool_)):
                    blank = False
                else:
                    blank = 0.0
                expanded = [blank for _ in range(n)]
            for i, value in zip(indices, values):
                expanded[i] = value
            params[key] = tuple(expanded)
        accepted = np.array(old['picks'].accepted_mask, copy=True) if old else np.zeros(n, dtype=bool)
        accepted[indices] = result['picks'].accepted_mask
        merged[name] = dict(name=name, picks=replace(original_picks, accepted_mask=accepted), params=params)
    for result in merged.values():
        result['params']['classification_dispositions'] = tuple(dispositions)
        result['params']['custom_templates'] = list(all_templates.values())
        result['params']['identification_generation'] = payload['params'].get('identification_generation')
        # Subset caches must never be serialized or reused with full candidate arrays.
        for key in ('_premeasured_picks', '_precomputed_candidates', '_prealigned_picks', '_reuse_saved_measurements'):
            result['params'].pop(key, None)
    output = dict(payload)
    output['templates'] = list(merged.values())
    output['counts'] = np.array([r['picks'].accepted_count for r in merged.values()])
    output['params'] = dict(next(iter(merged.values()))['params'])
    return output
