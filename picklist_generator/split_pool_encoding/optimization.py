"""Alternating Hungarian assignment and bounded analytic-gradient L-BFGS."""
from time import perf_counter
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial.distance import cdist
from .models import validate_data, shape, predict_and_pullback


def assignment(p, d, flexible=True):
    return linear_sum_assignment(cdist(p, d, metric='sqeuclidean'))[1] if flexible else np.arange(len(d))


def fit(d, model, seed=0, restarts=3, alternations=12, maxiter=150, flexible=True, initial=None, cancel=None, strategy="joint", allow_partial=False, basis_k=None, recipe_settings=None, recipe_probes=64, progress=None, address_bits=None, allowed_addresses=None, pca_seed=False, synthesis_mode='split_pool', response_model='capture', direct_source_nl=None):
    last_progress = 0.
    def report(fraction, detail):
        nonlocal last_progress
        last_progress = max(last_progress, min(1., float(fraction)))
        if progress is not None:
            progress(last_progress, detail)
    report(0, 'Preparing fit')
    if synthesis_mode not in ('split_pool', 'independent_wells'):
        raise ValueError('Unknown synthesis mode.')
    independent = synthesis_mode == 'independent_wells'
    if independent and model != 'pca' and (model not in ('on_only', 'on_off') or basis_k is None or recipe_settings is None):
        raise ValueError('Independent wells require finite K, ON-only or ON/OFF, and recipe settings.')
    if strategy not in ("joint", "per_site"):
        raise ValueError("Strategy must be joint or per_site.")
    d, bits = validate_data(d, bits=address_bits, allow_partial=allow_partial); m = d.shape[1]; start = perf_counter()
    eligible = np.arange(2**bits) if allowed_addresses is None else np.asarray(allowed_addresses)
    if (eligible.ndim != 1 or not np.issubdtype(eligible.dtype, np.integer)
            or len(eligible) < len(d) or len(np.unique(eligible)) != len(eligible)
            or np.any(eligible < 0) or np.any(eligible >= 2**bits)):
        raise ValueError('Allowed addresses must be unique valid integers with room for every target.')
    if recipe_settings is not None and (basis_k is None or model not in ('on_only', 'on_off')):
        raise ValueError('Quantized recipes require a finite K and an ON-only or ON/OFF model.')
    if model == 'pca':
        mean = d.mean(axis=0); u, s, vt = np.linalg.svd(d-mean, full_matrices=False)
        k = min(bits, len(s)); p = mean+(u[:, :k]*s[:k])@vt[:k]
        report(1, 'PCA complete')
        return dict(model=model, synthesis_mode=synthesis_mode, prediction=p, assignment=np.arange(len(d)), parameters=vt[:k],
                    mse=float(np.mean((p-d)**2)), runtime=perf_counter()-start, history=[], trials=[],
                    pca_mean=mean, pca_scores=u[:, :k]*s[:k],
                    address_indices=eligible[:len(d)].copy(), address_bits=bits, address_capacity=2**bits,
                    parameter_count=int(k*(len(d)+m)+m), converged=True)
    if model not in ('additive', 'on_only', 'on_off') or min(restarts, alternations, maxiter) < 1:
        raise ValueError('Unknown model or nonpositive optimization budget.')
    def match(prediction):
        if independent:
            return np.arange(len(d)), np.arange(len(d))
        if flexible:
            used, pi = linear_sum_assignment(cdist(prediction[eligible], d, metric='sqeuclidean'))
            return eligible[used], pi
        return eligible[:len(d)], np.arange(len(d))

    library = None
    if basis_k is not None:
        from .library import BasisLibrary
        if strategy != 'joint':
            raise ValueError('A shared library requires joint optimization across analog positions.')
        if recipe_settings is None:
            library = BasisLibrary(model, bits, m, basis_k)
        else:
            from .quantization import ConcentrationLibrary
            library = ConcentrationLibrary(model, bits, m, basis_k, recipe_settings, wells=len(d) if independent else None, response_model=response_model)
    def evaluate(x):
        return library.forward(x) if library else predict_and_pullback(x, model, bits, m)
    original_candidate = None; original_unavailable = None
    if independent and model == 'on_off' and direct_source_nl is not None and basis_k >= len(d):
        if cancel is not None and cancel.is_set():
            raise InterruptedError('Analysis cancelled.')
        try:
            source = np.zeros((basis_k, m, 2))
            original = np.asarray(direct_source_nl, float)
            if original.shape != (len(d), m, 2):
                raise ValueError('Original RGB transfer shape does not match the targets.')
            source[:len(d)] = original
            stocks = np.array([recipe_settings.on_stock_nm, recipe_settings.off_stock_nm])
            if np.any((original*stocks).sum(axis=-1) > recipe_settings.vector_max_nm*recipe_settings.vector_volume_nl+1e-6):
                raise ValueError('Original RGB staples exceed the configured per-site reference amount; increase the vector reference amount to preserve this recipe.')
            transfers = np.zeros((len(d), basis_k))
            transfers[np.arange(len(d)), np.arange(len(d))] = min(recipe_settings.vector_volume_nl, recipe_settings.branch_budget_nl)
            recipe = library.complete_recipe(source, transfers)
            prediction = recipe['u']; loss = float(np.mean((prediction-d)**2))
            original_candidate = dict(model=model, synthesis_mode=synthesis_mode, response_model=response_model,
                initialization='original_rgb_recipe', recipe=recipe, prediction=prediction, mse=loss, continuous_mse=loss,
                parameters=None, library_parameters=None, history=[], assignment=np.arange(len(d)),
                address_indices=eligible[:len(d)].copy(), address_bits=bits, address_capacity=2**bits,
                basis_k=basis_k, parameter_count=library.size, converged=True, quantized=True, retained_initial_recipe=False)
            if loss <= 1e-24:
                original_candidate.update(original_recipe_mse=loss, runtime=perf_counter()-start, pca_seed_tested=False,
                    exact_recipe_shortcut=True, trials=[dict(restart=0, initialization='original_rgb_recipe', mse=loss,
                    converged=True, optimizer_message='Original recipe matches targets to numerical precision; optimization skipped.')])
                report(1, 'Original RGB recipe matches targets; optimization skipped')
                return original_candidate
        except ValueError as exc:
            original_unavailable = str(exc)
    rng = np.random.default_rng(seed); best = None; trials = []
    stages = alternations + (1 if recipe_settings is not None else 0)
    starts = ['random']*restarts
    if pca_seed and library is not None:
        starts.append('pca')
    if independent and basis_k >= len(d):
        starts.append('target')
    total_restarts = len(starts)
    for trial in range(total_restarts):
        if cancel is not None and cancel.is_set():
            raise InterruptedError('Analysis cancelled.')
        report(trial/total_restarts, f'Restart {trial+1}/{total_restarts}')
        size = (library.size,) if library else shape(model, bits, m)
        x = rng.normal(0, .8, size) if model == 'additive' else rng.uniform(.02, .4, size)
        if model == 'on_off' and not library:
            x[..., 1] = rng.uniform(.1, .9, size[:-1])
        if library:
            x = library.initial(rng)
        initialization = starts[trial]
        if initialization == 'pca':
            x = library.pca_initial(d, np.random.default_rng(seed))
            report(trial/total_restarts, 'PCA-seeded library • constrained optimization')
        elif initialization == 'target':
            x = library.target_initial(d, np.random.default_rng(seed))
            report(trial/total_restarts, 'One-vector-per-design initialization • constrained optimization')
        if trial == 0 and initial is not None:
            initialization = 'provided'
            x = np.asarray(initial).reshape(size).copy()
        x = x.ravel(); history = []; previous = np.inf; converged = False
        initial_x = x.copy()
        initial_recipe = None
        if recipe_settings is not None:
            initial_recipe, initial_scored, _ = library.refine_recipe(x, d, False if independent else flexible,
                cancel, probes=0, allowed_addresses=np.arange(len(d)) if independent else eligible)
            history.append(dict(restart=trial, step=0, phase='initial_25nl_recipe', mse=initial_scored[0]))
        for step in range(alternations):
            iterations = 0
            report((trial*stages+step)/(total_restarts*stages), f'Restart {trial+1}/{total_restarts} • assignment {step+1}/{alternations}')
            p, _ = evaluate(x); used, pi = match(p)
            history.append(dict(restart=trial, step=step, phase='assignment', mse=float(np.mean((p[used]-d[pi])**2))))
            cached_z = None; cached_loss = None
            def objective(z):
                nonlocal cached_z, cached_loss
                if cancel is not None and cancel.is_set():
                    raise InterruptedError('Analysis cancelled.')
                pred, pullback = evaluate(z)
                residual = pred[used]-d[pi]
                gradient = np.zeros_like(pred)
                gradient[used] = 2*residual/residual.size
                loss_value = float(np.mean(residual**2)); derivative = pullback(gradient)
                cached_z = np.array(z, copy=True); cached_loss = loss_value
                return loss_value, derivative
            def callback(z):
                nonlocal iterations
                iterations += 1
                report((trial*stages+step+min(iterations/maxiter, 1))/(total_restarts*stages),
                       f'Restart {trial+1}/{total_restarts} • cycle {step+1}/{alternations} • iteration {iterations}/{maxiter}')
                logged_loss = cached_loss if cached_z is not None and np.array_equal(z, cached_z) else float(objective(z)[0])
                history.append(dict(restart=trial, step=step, phase='continuous', mse=logged_loss))
            options = dict(maxiter=maxiter, ftol=1e-12, gtol=1e-8, maxcor=15)
            if strategy == 'joint':
                result = minimize(objective, x, jac=True, method='L-BFGS-B', callback=callback,
                                  bounds=None if library or model == 'additive' else [(0, 1)]*len(x), options=options)
                x = result.x
            else:
                shaped = x.reshape(size).copy()
                all_success = True
                for j in range(m):
                    local = shaped[:, j:j+1] if model == 'additive' else shaped[:, :, j:j+1, :]
                    def local_objective(z):
                        if cancel is not None and cancel.is_set():
                            raise InterruptedError('Analysis cancelled.')
                        pred, pullback = predict_and_pullback(z, model, bits, 1)
                        residual = pred[used]-d[pi, j:j+1]
                        gradient = np.zeros_like(pred)
                        gradient[used] = 2*residual/residual.size
                        return np.mean(residual**2), pullback(gradient)
                    result = minimize(local_objective, local.ravel(), jac=True, method='L-BFGS-B',
                                      bounds=None if model == 'additive' else [(0, 1)]*local.size, options=options)
                    all_success = all_success and result.success
                    local[...] = result.x.reshape(local.shape)
                    report((trial*stages+step+(j+1)/m)/(total_restarts*stages),
                           f'Restart {trial+1}/{total_restarts} • cycle {step+1}/{alternations} • site {j+1}/{m}')
                result.success = all_success
                x = shaped.ravel(); callback(x)
            p, _ = evaluate(x); new_used, new_pi = match(p)
            loss = float(np.mean((p[new_used]-d[new_pi])**2))
            history.append(dict(restart=trial, step=step+1, phase='assignment', mse=loss))
            report((trial*stages+step+1)/(total_restarts*stages), f'Restart {trial+1}/{total_restarts} • cycle {step+1}/{alternations} complete')
            if abs(previous-loss) < 1e-10 and np.array_equal(pi, new_pi) and np.array_equal(used, new_used):
                converged = bool(result.success); break
            previous = loss
        continuous_loss = loss
        recipe = None
        retained_initial_recipe = False
        if recipe_settings is not None:
            recipe, scored, projected_loss = library.refine_recipe(x, d, False if independent else flexible, cancel, probes=recipe_probes, allowed_addresses=np.arange(len(d)) if independent else eligible,
                progress=lambda fraction: report((trial*stages+alternations+fraction)/(total_restarts*stages),
                    f'Restart {trial+1}/{total_restarts} • 25 nL refinement {int(fraction*100)}%'))
            loss, p, new_used, new_pi = scored
            history.extend([dict(restart=trial, step=step+1, phase='25nl_projection', mse=projected_loss),
                            dict(restart=trial, step=step+1, phase='25nl_refinement', mse=loss)])
            if initial_scored[0] < loss:
                # A better continuous fit can round to a worse physical recipe.
                # Retain the already feasible initial recipe in that case.
                x = initial_x; recipe = initial_recipe
                loss, p, new_used, new_pi = initial_scored
                continuous, _ = evaluate(x); used_continuous, pi_continuous = match(continuous)
                continuous_loss = float(np.mean((continuous[used_continuous]-d[pi_continuous])**2))
                retained_initial_recipe = True; converged = False
                history.append(dict(restart=trial, step=step+1, phase='retain_initial_25nl_recipe', mse=loss))
        trials.append(dict(restart=trial, initialization=initialization, mse=loss, retained_initial_recipe=retained_initial_recipe, converged=converged, optimizer_message=str(result.message)))
        candidate = dict(model=model, synthesis_mode=synthesis_mode, response_model=response_model, initialization=initialization, prediction=p[new_used], assignment=new_pi,
                         address_indices=eligible[new_used] if independent else new_used, address_bits=bits, address_capacity=2**bits, parameters=x.reshape(size), mse=loss,
                         history=history, parameter_count=(bits+1)*m if model == 'additive' else bits*2*m*(2 if model == 'on_off' else 1), converged=converged)
        if library:
            candidate.update(basis_k=basis_k, library_parameters=library.describe(x), parameter_count=library.size)
        if recipe is not None:
            candidate.update(recipe=recipe, continuous_mse=continuous_loss, quantized=True, retained_initial_recipe=retained_initial_recipe)
        if best is None or loss < best['mse']:
            best = candidate
        report((trial+1)/total_restarts, f'Restart {trial+1}/{total_restarts} complete')
    if original_candidate is not None:
        trials.append(dict(restart=len(trials), initialization='original_rgb_recipe', mse=original_candidate['mse'],
                           converged=True, optimizer_message='Exact existing RGB transfers; one stock per design.'))
        if original_candidate['mse'] <= best['mse']:
            best = original_candidate
        best['original_recipe_mse'] = original_candidate['mse']
    if original_unavailable:
        best['original_recipe_unavailable'] = original_unavailable
    best.update(runtime=perf_counter()-start, trials=trials, pca_seed_tested=bool(pca_seed and library is not None))
    report(1, 'Fit complete')
    return best
