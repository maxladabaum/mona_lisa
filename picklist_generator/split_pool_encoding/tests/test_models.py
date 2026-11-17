import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from split_pool_encoding.models import transitions, predict_and_pullback, shape, validate_data, physical_parameters
from split_pool_encoding.optimization import fit, assignment
from split_pool_encoding.datasets import generate, FAMILIES, from_designs
from split_pool_encoding.evaluation import sampling
from split_pool_encoding.experiment import run_dataset


def test_irreversible_occupancy():
    u = np.full((2, 2, 3), .5); v = np.zeros_like(u)
    U, O, F = transitions(u, v)
    np.testing.assert_allclose(O, .75); np.testing.assert_allclose(U+O+F, 1)
    u[0] = 0; v[0] = 1
    U, O, F = transitions(u, v)
    np.testing.assert_allclose(O, 0); np.testing.assert_allclose(F, 1)


def test_probability_conservation_and_last_off_nonidentifiability():
    rng = np.random.default_rng(2); q = rng.random((8, 2, 40)); t = rng.random(q.shape)
    u, v = q*t, q*(1-t); U, O, F = transitions(u, v)
    assert min(U.min(), O.min(), F.min()) >= 0
    np.testing.assert_allclose(U+O+F, 1)
    v[-1] = 0; np.testing.assert_allclose(transitions(u, v)[1], O)


@pytest.mark.parametrize('model', ['additive', 'on_only', 'on_off'])
def test_gradients_and_physical_predictions(model):
    rng = np.random.default_rng(4); x = rng.uniform(.1, .6, shape(model, 3, 4)).ravel(); g = rng.normal(size=(8, 4))
    p, pullback = predict_and_pullback(x, model, 3, 4)
    numeric = approx_derivative(lambda z: np.sum(predict_and_pullback(z, model, 3, 4)[0]*g), x).ravel()
    np.testing.assert_allclose(pullback(g), numeric, atol=1e-8)
    assert p.min() >= 0 and p.max() <= 1
    params = physical_parameters(x.reshape(shape(model, 3, 4)), model)
    if model != 'additive':
        np.testing.assert_allclose(transitions(**params)[1], p)


@pytest.mark.parametrize('family', FAMILIES)
def test_datasets(family):
    d, truth = generate(family, 3, 7, 8)
    assert validate_data(d)[1] == 3
    np.testing.assert_equal(d, generate(family, 3, 7, 8)[0])


@pytest.mark.parametrize('model', ['additive', 'on_only', 'on_off'])
def test_known_small_solutions_and_free_assignment(model):
    rng = np.random.default_rng(5); x = rng.uniform(.1, .7, shape(model, 2, 3))
    d = predict_and_pullback(x, model, 2, 3)[0][[2, 0, 3, 1]]
    result = fit(d, model, seed=9, restarts=3, alternations=12, maxiter=200)
    assert result['mse'] < 1e-7
    assert sorted(result['assignment']) == list(range(4))
    warm = fit(d, model, initial=x, restarts=1, alternations=2, maxiter=50)
    assert warm['mse'] < 1e-16


def test_sampling_error_scaling():
    p = np.full((8, 4), .5)
    rows = sampling(p, p, counts=[50, 1000], repeats=200, seed=8)
    assert np.mean([r['sampling_rmse'] for r in rows if r['samples'] == 50]) > 3*np.mean([r['sampling_rmse'] for r in rows if r['samples'] == 1000])
    assert all(r['encoding_rmse'] == 0 for r in rows)


@pytest.mark.parametrize('d', [np.ones((3, 4)), [[np.nan], [0]], [[1.1], [0]]])
def test_bad_data(d):
    with pytest.raises(ValueError):
        validate_data(d)


def test_export_and_assignment_orientation(tmp_path):
    d, _ = generate('binary_cube', 2, 3)
    rows = run_dataset(d, tmp_path/'run', dict(restarts=1, alternations=2, maxiter=20, measurement_repeats=2, figures=True))
    assert len(rows) == 7
    saved = np.load(tmp_path/'run/on_off/model.npz')
    assert np.mean((saved['prediction']-d[saved['assignment']])**2) == pytest.approx(rows[5]['mse'])
    assert (tmp_path/'run/on_off/heatmaps.pdf').exists()
    assert not (tmp_path/'run/pca/sampling.csv').exists()
    pca = np.load(tmp_path/'run/pca/model.npz')
    np.testing.assert_allclose(pca['mean']+pca['scores']@pca['parameters'], pca['prediction'])


def test_rgb_targets_preserve_site_order():
    designs = [dict(color='Red', name=str(i), barcode_code=i, image_sites=[dict(name='b'), dict(name='a')], site_fractions={'a': .2, 'b': .8}) for i in range(4)]
    d, metadata = from_designs(designs, 'Red')
    np.testing.assert_equal(d, [[.8, .2]]*4)
    assert metadata['sites'] == ['b', 'a']


def test_per_site_solver_and_cancellation():
    from threading import Event
    d, _ = generate('binary_cube', 2, 3)
    result = fit(d, 'additive', restarts=2, alternations=8, maxiter=150, strategy='per_site')
    assert result['mse'] < 1e-7
    cancelled = Event(); cancelled.set()
    with pytest.raises(InterruptedError):
        fit(d, 'on_off', cancel=cancelled)


def test_sampling_roundoff_is_not_an_invalid_probability():
    p = np.array([[1+2e-16], [0.]])
    assert sampling(np.clip(p, 0, 1), p, counts=[300], repeats=1)[0]['total_rmse'] == 0
    with pytest.raises(ValueError):
        sampling(p, p+.01, counts=[300], repeats=1)


def test_full_size_oracle_recovery():
    from split_pool_encoding.validation import known_solution_checks
    assert all(row['passed'] for row in known_solution_checks())


def test_off_competition_can_encode_an_interaction_absent_from_on_only():
    u = np.zeros((2, 2, 1)); v = np.zeros_like(u)
    v[0, 0] = 1  # First bit 0 irreversibly blocks the second round.
    u[1, 1] = .6
    p = transitions(u, v)[1].ravel()
    np.testing.assert_allclose(p, [0, 0, 0, .6])
    # Every additive cube has zero rectangular contrast. ON-only has zero
    # contrast in hazard space; this counterexample violates both identities.
    assert p[0]-p[1]-p[2]+p[3] == pytest.approx(.6)
    h = -np.log1p(-p)
    assert h[0]-h[1]-h[2]+h[3] > .9


def test_finite_programming_precision():
    from split_pool_encoding.evaluation import precision_sweep
    x = np.full(shape('on_off', 2, 3), .5)
    p, _ = predict_and_pullback(x, 'on_off', 2, 3)
    rows = precision_sweep(p, x, 'on_off', 2, levels=(3, 256))
    assert rows[0]['mse'] == 0
    assert rows[1]['programming_bits_upper_bound'] == x.size*8


@pytest.mark.parametrize('model', ['additive', 'on_only', 'on_off', 'pca'])
@pytest.mark.parametrize('count', [1, 3, 21])
def test_partial_address_space_preserves_every_target(model, count):
    rng = np.random.default_rng(7)
    d = rng.uniform(.1, .9, (count, 3))
    result = fit(d, model, restarts=1, alternations=2, maxiter=20, allow_partial=True)
    assert sorted(result['assignment']) == list(range(count))
    assert len(set(result['address_indices'])) == count
    assert result['address_capacity'] == 2**max(1, (count-1).bit_length())
    assert result['prediction'].shape == d.shape
    assert result['mse'] == pytest.approx(np.mean((result['prediction']-d[result['assignment']])**2))
    if model != 'pca':
        full, _ = predict_and_pullback(result['parameters'], model, result['address_bits'], 3)
        np.testing.assert_allclose(full[result['address_indices']], result['prediction'])


@pytest.mark.parametrize('flexible', [True, False])
def test_partial_per_site_optimizer(flexible):
    d = np.array([[.1, .3], [.6, .2], [.7, .8]])
    result = fit(d, 'on_off', restarts=1, alternations=2, maxiter=30,
                 allow_partial=True, strategy='per_site', flexible=flexible)
    assert sorted(result['assignment']) == [0, 1, 2]
    assert result['mse'] == pytest.approx(np.mean((result['prediction']-d[result['assignment']])**2))
    if not flexible:
        np.testing.assert_array_equal(result['address_indices'], [0, 1, 2])
        np.testing.assert_array_equal(result['assignment'], [0, 1, 2])


def test_matrix_monte_carlo_uses_assigned_probabilities_and_realized_recipe():
    from split_pool_encoding.evaluation import matrix_example
    target = np.tile([.1, .5, .9], (64, 1))
    pi = np.random.default_rng(4).permutation(64)
    realized = np.tile([0., .4, 1.], (64, 1))
    result = dict(model='on_off', assignment=pi, prediction=realized[pi], recipe={'realized': True})
    expected, measured, errors = matrix_example(target, result, 30, 5)
    np.testing.assert_equal(expected, realized)
    assert np.all(measured[:, 0] == 0) and np.all(measured[:, 2] == 1)
    np.testing.assert_array_equal(measured, matrix_example(target, result, 30, 5)[1])
    assert not np.array_equal(measured, matrix_example(target, result, 30, 6)[1])
    assert matrix_example(target, result, 3000, 5)[2]['sampling_rmse'] < errors['sampling_rmse']/5
    assert errors['encoding_rmse'] == pytest.approx(np.sqrt(np.mean((realized-target)**2)))


def test_matrix_pca_clips_only_preview_and_restores_target_row_order():
    from split_pool_encoding.evaluation import matrix_example
    target = np.array([[.1, .5, .9], [.2, .4, .8]])
    raw = np.array([[-.2, .5, 1.2], [.1, .3, .9]])
    result = dict(model='pca', assignment=np.array([1, 0]), prediction=raw[::-1].copy())
    before = result['prediction'].copy()
    expected, measured, errors = matrix_example(target, result, 300, 4)
    np.testing.assert_array_equal(expected, np.clip(raw, 0, 1))
    np.testing.assert_array_equal(result['prediction'], before)
    assert errors['clipped_fraction'] == pytest.approx(2/6)
    assert errors['raw_encoding_rmse'] > errors['encoding_rmse']
    result['model'] = 'on_off'
    with pytest.raises(ValueError, match='Physical predictions'):
        matrix_example(target, result)


@pytest.mark.parametrize('model', ['pca', 'additive', 'on_only', 'on_off'])
def test_fit_progress_is_monotonic_and_reports_completion(model):
    data = np.random.default_rng(7).random((4, 3))
    updates = []
    fit(data, model, restarts=2, alternations=2, maxiter=15,
        progress=lambda fraction, detail: updates.append((fraction, detail)))
    fractions = [u[0] for u in updates]
    assert fractions[0] == 0 and fractions[-1] == 1
    assert all(0 <= a <= b <= 1 for a, b in zip(fractions, fractions[1:]))
    if model != 'pca':
        assert any(0 < value < 1 for value in fractions)
        assert any('iteration' in detail for _, detail in updates)
