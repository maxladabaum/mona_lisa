"""Encoding metrics and repeated finite-population readout experiments."""
import numpy as np


def metrics(target, prediction):
    error = np.asarray(prediction)-target; absolute = np.abs(error)
    per_file = np.sqrt(np.mean(error**2, axis=1))
    return dict(mse=float(np.mean(error**2)), rmse=float(np.sqrt(np.mean(error**2))),
                max_absolute_error=float(absolute.max()), median_file_rmse=float(np.median(per_file)),
                p95_file_rmse=float(np.quantile(per_file, .95)),
                **{f'fraction_within_{tol}': float(np.mean(absolute <= tol)) for tol in (.01, .025, .05, .1)})


def sampling(target, prediction, counts=(50, 100, 200, 300, 500, 1000), repeats=100, seed=0, false_positive=0., false_negative=0.):
    if repeats < 1 or any(int(n) != n or n < 1 for n in counts) or not 0 <= false_positive <= 1 or not 0 <= false_negative <= 1:
        raise ValueError('Invalid readout settings.')
    p = np.asarray(prediction)
    if not np.isfinite(p).all() or np.any((p < -1e-12) | (p > 1+1e-12)):
        raise ValueError('Binomial readout requires physical probabilities in [0, 1].')
    # Remove floating-point accumulation error only (e.g. 1 + 2e-16).
    p = np.clip(p, 0, 1)
    rng = np.random.default_rng(seed); rows = []
    observed = p*(1-false_negative)+(1-p)*false_positive
    for n in counts:
        for repetition in range(repeats):
            measured = rng.binomial(n, observed)/n
            rows.append(dict(samples=int(n), repetition=repetition,
                             encoding_rmse=float(np.sqrt(np.mean((p-target)**2))),
                             sampling_rmse=float(np.sqrt(np.mean((measured-p)**2))),
                             total_rmse=float(np.sqrt(np.mean((measured-target)**2)))))
    return rows


def transfer_estimate(model, bits, positions):
    # One dispensing operation per site-specific reagent per branch environment;
    # plus one digital address reagent per branch. Pooling/washes not counted.
    if model == 'pca':
        return None
    return 2*bits*(positions*(2 if model in ('on_off', 'additive') else 1)+1)


def precision_sweep(target, parameters, model, bits, levels=(4, 16, 64, 256)):
    """Finite q/t programming precision; output error, not an information-capacity estimate."""
    from .models import predict_and_pullback
    if model not in ('on_only', 'on_off'):
        return []
    rows = []
    for count in levels:
        if count < 2 or int(count) != count:
            raise ValueError('At least two discrete programming levels are required.')
        quantized = np.rint(parameters*(count-1))/(count-1)
        p, _ = predict_and_pullback(quantized, model, bits, target.shape[1])
        rows.append(dict(levels=count, programming_bits_upper_bound=float(parameters.size*np.log2(count)),
                         **metrics(target, p)))
    return rows


def matrix_example(target, result, count=300, seed=42):
    """One finite-count realization, restored to original target-row order.

    Fits may contain realized quantized-recipe predictions; these are used
    unchanged. Only PCA is clipped for a hypothetical probability reference.
    """
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('Origami count must be a positive integer.')
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('Example seed must be a nonnegative integer.')
    target = np.asarray(target, float)
    p = np.asarray(result['prediction'], float); pi = np.asarray(result['assignment'])
    if target.ndim != 2 or min(target.shape) < 1 or p.shape != target.shape:
        raise ValueError('Predictions must match the target matrix.')
    if not np.isfinite(target).all() or np.any((target < 0) | (target > 1)):
        raise ValueError('Target values must be finite probabilities.')
    if not np.array_equal(np.sort(pi), np.arange(len(target))):
        raise ValueError('Every target row must have exactly one assigned prediction.')
    if not np.isfinite(p).all():
        raise ValueError('Predictions must be finite.')
    if result['model'] != 'pca' and np.any((p < -1e-12) | (p > 1+1e-12)):
        raise ValueError('Physical predictions must lie in [0, 1].')
    raw = np.empty_like(p); raw[pi.astype(int)] = p
    expected = np.clip(raw, 0, 1)
    measured = np.random.default_rng(seed).binomial(count, expected)/count
    errors = dict(raw_encoding_rmse=float(np.sqrt(np.mean((raw-target)**2))),
                  clipped_fraction=float(np.mean((raw < 0) | (raw > 1))),
                  encoding_rmse=float(np.sqrt(np.mean((expected-target)**2))),
                  sampling_rmse=float(np.sqrt(np.mean((measured-expected)**2))),
                  total_rmse=float(np.sqrt(np.mean((measured-target)**2))))
    return expected, measured, errors
