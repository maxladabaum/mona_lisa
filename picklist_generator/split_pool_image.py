"""Finite-population RGB previews from fitted split–pool address probabilities."""
from copy import deepcopy
import numpy as np
from gaussian_reconstruction import render_gaussian_overlay


def image_cases(designs, color_fits, count=300, seed=42):
    """Restore target row order before applying independent binomial occupancies.

    Color views share one split–pool fit; independent wells may use separate libraries.
    N is per barcode per color.
    Binomial sampling is the marginal of N independent irreversible trajectories
    with a fixed branch history, not another stochastic optimizer run. PCA is
    a hypothetical reference: finite reconstructions are clipped to [0, 1]
    solely for rendering and binomial sampling; fitted arrays remain intact.
    """
    if isinstance(count, bool) or int(count) != count or count < 1:
        raise ValueError('Origami per barcode must be a positive integer.')
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('Example seed must be a nonnegative integer.')
    colors = list(dict.fromkeys(d['color'] for d in designs))
    if not colors or set(colors) != set(color_fits):
        raise ValueError('Fit every image color before rendering the RGB preview.')
    target, expected, sampled = (deepcopy(designs) for _ in range(3))
    rng = np.random.default_rng(seed); targets = []; predictions = []; observations = []; raw_predictions = []
    for color in colors:
        indices = [i for i, d in enumerate(designs) if d['color'] == color]
        result = color_fits[color]
        pi = np.asarray(result['assignment']); p = np.asarray(result['prediction'], float)
        sites = [s['name'] for s in designs[indices[0]]['image_sites']]
        if p.shape != (len(indices), len(sites)) or not np.array_equal(np.sort(pi), np.arange(len(indices))):
            raise ValueError('Fitted addresses must be a one-to-one mapping to image designs.')
        if not np.isfinite(p).all():
            raise ValueError('Image predictions must be finite.')
        if result.get('model') != 'pca' and np.any((p < -1e-12) | (p > 1+1e-12)):
            raise ValueError('Split–pool predictions must be probabilities in [0, 1].')
        raw = np.empty_like(p); raw[pi.astype(int)] = p
        ordered = np.clip(raw, 0, 1)
        measured = rng.binomial(int(count), ordered)/count
        for row, index in enumerate(indices):
            original = designs[index]
            if [s['name'] for s in original['image_sites']] != sites:
                raise ValueError('All designs in a color must share their analog site layout.')
            truth = np.array([original['site_fractions'][name] for name in sites])
            targets.extend(truth); predictions.extend(ordered[row]); observations.extend(measured[row])
            raw_predictions.extend(raw[row])
            for case, values in ((target, truth), (expected, ordered[row]), (sampled, measured[row])):
                for site, value in zip(case[index]['image_sites'], values):
                    site.pop('effective_brightness', None)
                    site['replacement_fraction'] = float(value)
                    case[index]['site_fractions'][site['name']] = float(value)
    d, p, measured = map(np.asarray, (targets, predictions, observations))
    raw = np.asarray(raw_predictions)
    errors = dict(raw_encoding_rmse=float(np.sqrt(np.mean((raw-d)**2))),
                  clipped_fraction=float(np.mean((raw < 0) | (raw > 1))),
                  encoding_rmse=float(np.sqrt(np.mean((p-d)**2))),
                  sampling_rmse=float(np.sqrt(np.mean((measured-p)**2))),
                  total_rmse=float(np.sqrt(np.mean((measured-d)**2))))
    return (target, expected, sampled), errors


def render_image_example(designs, color_fits, count=300, seed=42, render_options=None):
    cases, errors = image_cases(designs, color_fits, count, seed)
    images = [render_gaussian_overlay(case, **(render_options or {})).rgb for case in cases]
    return images, errors
