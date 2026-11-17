"""Oracle-start checks distinguish a correct forward model from blind search recovery."""
import numpy as np
from .datasets import generate
from .optimization import fit


def known_solution_checks(bits=8, positions=40, seed=17):
    rows = []
    for model, family in (('additive', 'binary_cube'), ('on_off', 'on_off')):
        d, truth = generate(family, bits, positions, seed)
        if model == 'additive':
            w = truth['basis']; positive = np.maximum(w, 0); negative = np.maximum(-w, 0)
            baseline = truth['baseline']-negative.sum(axis=0)
            slack = 1-baseline-positive.sum(axis=0)-negative.sum(axis=0)
            weights = np.vstack((baseline, positive, negative, slack))
            initial = np.log(np.maximum(weights, 1e-15))
        else:
            q = truth['u']+truth['v']
            initial = np.stack((q, np.divide(truth['u'], q, out=np.zeros_like(q), where=q>0)), axis=-1)
        result = fit(d, model, initial=initial, restarts=1, alternations=2, maxiter=5)
        rows.append(dict(model=model, bits=bits, positions=positions, mse=result['mse'],
                         passed=result['mse'] < 1e-20, initialization='generator oracle; not blind recovery'))
    return rows
