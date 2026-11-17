"""Trace one assigned digital address through its actual branch mixtures."""
import numpy as np


def branch_trace(result, target_row):
    if result.get('synthesis_mode') == 'independent_wells' or result['model'] == 'pca' or result.get('library_parameters') is None:
        return None
    row = int(np.flatnonzero(np.asarray(result['assignment']) == target_row)[0])
    rounds = result['address_bits']
    address = int(result.get('address_indices', np.arange(len(result['assignment'])))[row])
    code = format(address, f'0{rounds}b')
    bits = np.array([int(b) for b in code])
    branches = 2*np.arange(rounds)+bits
    library = result['library_parameters']; recipe = result.get('recipe')
    if recipe is not None:
        amounts = np.asarray(recipe['branch_transfers_nl'])[branches]
        u = np.asarray(recipe['u'])[np.arange(rounds), bits]
        v = np.asarray(recipe['v'])[np.arange(rounds), bits]
        unit = 'Transfer volume (nL)'
        note = 'Actual 25 nL transfers from diluted vector wells; '+('each nonzero exposure saturates remaining sites using its ON/OFF ratio.' if recipe.get('response_model') == 'rgb_ratio' else 'capture uses the adjustable, uncalibrated C50 model.')
    elif result['model'] == 'additive':
        amounts = np.asarray(library['branch_coefficients'])*bits[:, None]
        changes = amounts @ library['basis_vectors']
        states = np.vstack((library['baseline'], library['baseline']+np.cumsum(changes, axis=0)))
        return dict(code=code, bits=bits, amounts=amounts, on=states, off=None, vacant=None,
                    unit='Signed coefficient', note='Additive reference: baseline plus signed contributions; not a physical staple recipe.')
    else:
        amounts = np.asarray(library['branch_coefficients'])[branches]
        basis = np.asarray(library['basis_vectors'])
        if result['model'] == 'on_only':
            u = amounts @ basis; v = np.zeros_like(u)
        else:
            u = amounts @ basis[..., 0]; v = amounts @ basis[..., 1]
        unit = 'Mixture weight'
        note = 'Ideal mixtures. Only unoccupied sites can become ON or OFF; occupied sites stay occupied.'
    on = [np.zeros(u.shape[1])]; off = [np.zeros(u.shape[1])]; vacant = [np.ones(u.shape[1])]
    for ur, vr in zip(u, v):
        on.append(on[-1]+vacant[-1]*ur)
        off.append(off[-1]+vacant[-1]*vr)
        vacant.append(vacant[-1]*(1-ur-vr))
    return dict(code=code, bits=bits, amounts=amounts, on=np.array(on), off=np.array(off),
                vacant=np.array(vacant), unit=unit, note=note)
