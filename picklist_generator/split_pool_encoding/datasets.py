"""Seeded benchmarks and strict CSV/NumPy input."""
from pathlib import Path
import numpy as np
from .models import addresses, transitions, validate_data

FAMILIES = ('random', 'low_rank_2', 'low_rank_4', 'low_rank_8', 'binary_cube', 'on_off', 'smooth', 'peaks', 'correlated', 'clusters', 'exact_k_2', 'exact_k_4', 'exact_k_8')


def generate(family, bits=8, positions=40, seed=0, *, recipe_settings=None,
             response_model='capture', extra_rounds=0):
    if family.startswith('exact_k_'):
        return generate_exact_k(int(family.rsplit('_', 1)[1]), bits, positions, seed,
                                recipe_settings, response_model, extra_rounds)
    rng = np.random.default_rng(seed); n = 2**bits
    truth = {}
    if family == 'random':
        d = rng.random((n, positions))
    elif family.startswith('low_rank_'):
        rank = int(family.rsplit('_', 1)[1])
        d = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, positions))
        d = .05+.9*(d-d.min())/max(np.ptp(d), 1e-12)
    elif family == 'binary_cube':
        w = rng.uniform(-1, 1, (bits, positions)); w *= .8/np.abs(w).sum(axis=0)
        c = .1-np.minimum(w, 0).sum(axis=0)
        d = c+addresses(bits)@w; truth = dict(baseline=c, basis=w)
    elif family == 'on_off':
        q = rng.uniform(.02, .5, (bits, 2, positions)); t = rng.random(q.shape)
        u, v = q*t, q*(1-t); d = transitions(u, v)[1]; truth = dict(u=u, v=v)
    elif family in ('smooth', 'peaks', 'correlated', 'clusters'):
        grid = np.linspace(0, 1, positions)
        if family == 'clusters':
            centers = rng.uniform(.15, .85, (min(8, n), positions))
            d = np.clip(centers[rng.integers(len(centers), size=n)]+rng.normal(0, .025, (n, positions)), 0, 1)
        elif family == 'peaks':
            centers = rng.random((n, 3, 1)); widths = rng.uniform(.04, .2, (n, 3, 1))
            d = (rng.random((n, 3, 1))*np.exp(-.5*((grid-centers)/widths)**2)).sum(axis=1)/3
        else:
            scale = .18 if family == 'smooth' else .05
            cov = np.exp(-.5*((grid[:, None]-grid)/scale)**2)+np.eye(positions)*1e-8
            raw = rng.normal(size=(n, positions))@np.linalg.cholesky(cov).T
            d = (np.tanh(raw)+1)/2
    else:
        raise ValueError(f'Unknown dataset: {family}')
    permutation = rng.permutation(n)
    return d[permutation], {**truth, 'source_address': permutation}


def load(path):
    path = Path(path)
    d = np.load(path, allow_pickle=False) if path.suffix.lower() == '.npy' else np.loadtxt(path, delimiter=',', ndmin=2)
    return validate_data(d)[0]


def from_designs(designs, color):
    selected = [d for d in designs if d['color'] == color]
    if not selected:
        raise ValueError('Load an image first, then select a populated color.')
    sites = [site['name'] for site in selected[0]['image_sites']]
    if any([site['name'] for site in d['image_sites']] != sites for d in selected):
        raise ValueError('Designs must share the same analog site layout.')
    data = [[d['site_fractions'][site] for site in sites] for d in selected]
    return validate_data(data, allow_partial=True)[0], dict(color=color, sites=sites, target_names=[d['name'] for d in selected],
                                     target_barcode_codes=[d['barcode_code'] for d in selected])


def generate_exact_k(k, bits=8, positions=40, seed=0, settings=None,
                     response_model='capture', extra_rounds=0):
    """Targets from an actual 25 nL recipe; K is stock count, not matrix rank.

    The stored construction is an existence proof, not a blind optimizer result.
    Exactness refers to expected ON occupancy under the same capture model.
    """
    import json
    from dataclasses import asdict
    from .quantization import ConcentrationLibrary, RecipeSettings
    settings = settings or RecipeSettings(20000, 20000, 100000, 300, 10000, 5000, 50)
    rounds = bits + extra_rounds
    if not 1 <= bits <= rounds <= 12 or positions < 1:
        raise ValueError('Use positive positions and 1–12 rounds, with nonnegative extra rounds.')
    rng = np.random.default_rng(seed)
    library = ConcentrationLibrary('on_off', rounds, positions, k, settings,
                                   response_model=response_model)
    recipe = library.recipe(4 * library.initial(rng))
    source_address = rng.choice(2**rounds, 2**bits, replace=False)
    targets = transitions(recipe['u'], recipe['v'])[1][source_address]
    if np.ptp(targets) < 1e-8:
        raise ValueError('These settings produce a degenerate exact-K dataset; increase stock/dose resolution.')
    truth = dict(k=k, rounds=rounds, source_address=source_address,
                 source_transfers_nl=recipe['source_transfers_nl'],
                 branch_transfers_nl=recipe['branch_transfers_nl'],
                 settings_json=json.dumps(asdict(settings)), response_model=response_model)
    return targets, truth


def replay_exact_k(truth):
    """Rebuild from exported transfer volumes, independently of generator transitions."""
    import json
    settings = json.loads(str(truth['settings_json']))
    source = np.asarray(truth['source_transfers_nl'])
    branch = np.asarray(truth['branch_transfers_nl'])
    stocks = source * [settings['on_stock_nm'], settings['off_stock_nm']] / settings['vector_volume_nl']
    dose = (branch @ stocks.reshape(len(source), -1) / settings['branch_volume_nl']).reshape(
        int(truth['rounds']), 2, source.shape[1], 2)
    codes = addresses(int(truth['rounds']))[truth['source_address']].astype(int)
    on = np.zeros((len(codes), source.shape[1])); empty = np.ones_like(on)
    for r in range(int(truth['rounds'])):
        exposure = dose[r, codes[:, r]]
        total = exposure.sum(axis=-1)
        capture = ((total > 0).astype(float) if str(truth['response_model']) == 'rgb_ratio'
                   else -np.expm1(-np.log(2)*total/settings['capture_c50_nm']))
        fraction = np.divide(exposure[..., 0], total, out=np.zeros_like(total), where=total > 0)
        on += empty*capture*fraction
        empty *= 1-capture
    return on
