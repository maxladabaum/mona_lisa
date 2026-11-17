"""Visualize mixing separately from irreversible capture along one design path."""
import numpy as np


def canvas_build_steps(result, target_row):
    if result['model'] not in ('on_only', 'on_off'):
        return None
    recipe = result.get('recipe'); library = result.get('library_parameters')
    if recipe is None and library is None:
        return None
    assigned = int(np.flatnonzero(np.asarray(result['assignment']) == target_row)[0])
    bits = result['address_bits']
    address = int(result.get('address_indices', np.arange(len(result['assignment'])))[assigned])
    code = format(address, f'0{bits}b')
    independent = result.get('synthesis_mode') == 'independent_wells'
    destinations = [assigned] if independent else [2*r+int(bit) for r, bit in enumerate(code)]
    if recipe is not None:
        vectors = np.asarray(recipe['vector_concentrations_nm'])
        weights = np.asarray(recipe['branch_transfers_nl'])[destinations]
        contributions = weights[:, :, None, None]*vectors[None, ...]/recipe['settings']['branch_volume_nl']
        unit = 'Delivered concentration (nM)'
    else:
        vectors = np.asarray(library['basis_vectors'])
        if result['model'] == 'on_only':
            vectors = np.stack((vectors, np.zeros_like(vectors)), axis=-1)
        else:
            vectors = vectors[..., :2]
        weights = np.asarray(library['branch_coefficients'])[destinations]
        contributions = weights[:, :, None, None]*vectors[None, ...]
        unit = 'Conditional occupancy contribution'
    m = contributions.shape[2]
    on = np.zeros(m); off = np.zeros(m); vacant = np.ones(m)
    empty = np.zeros((m, 2))
    events = [dict(label='Initial empty canvas', phase='initial', round=0, vector=None, amount=0.,
                   before=empty, added=empty, mixture=empty, on_before=on, on_after=on, off_after=off)]
    def capture(mixture):
        if recipe is None:
            u, v = mixture[:, 0], mixture[:, 1]
        else:
            total = mixture.sum(axis=-1)
            q = (total > 0).astype(float) if recipe.get('response_model') == 'rgb_ratio' else -np.expm1(-np.log(2)*total/recipe['settings']['capture_c50_nm'])
            share = np.divide(mixture[:, 0], total, out=np.zeros_like(total), where=total>0)
            u, v = q*share, q*(1-share)
        return on+vacant*u, off+vacant*v, vacant*(1-u-v)
    for r, doses in enumerate(contributions):
        label = 'Destination well' if independent else f'Round {r+1} · branch {code[r]} · digital D{r+1}={code[r]} ({"ON" if code[r] == "1" else "OFF"}) · code {code[:r+1]}'
        mixture = np.zeros((m, 2))
        # Ascending K is a display order only. Capture occurs after ALL additions.
        for k in np.flatnonzero(weights[r] > 0):
            before = mixture.copy(); mixture = mixture+doses[k]
            expected_on, expected_off, _ = capture(mixture)
            amount = float(weights[r, k]); amount_label = f'{amount:g} nL' if recipe else f'weight {amount:.4g}'
            events.append(dict(label=f'{label}: add K{k+1} ({amount_label}) — mixing only', phase='addition', round=r+1,
                               vector=int(k+1), amount=amount, before=before, added=doses[k], mixture=mixture,
                               on_before=on, on_after=expected_on, off_after=expected_off))
        expected_on, expected_off, expected_vacant = capture(mixture)
        events.append(dict(label=f'{label}: capture complete', phase='capture', round=r+1, vector=None, amount=0.,
                           before=mixture, added=empty, mixture=mixture, on_before=on,
                           on_after=expected_on, off_after=expected_off))
        on, off, vacant = expected_on, expected_off, expected_vacant
    return dict(events=events, code=code, unit=unit, physical=recipe is not None, independent=independent,
                maximum=max(float(contributions.sum(axis=1).max()), 1e-12), final_on=on, final_off=off)
