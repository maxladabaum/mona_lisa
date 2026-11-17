from dataclasses import replace
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from split_pool_encoding.quantization import RecipeSettings, ConcentrationLibrary, quantize_volumes
from split_pool_encoding.optimization import fit
from split_pool_encoding.models import transitions


def settings():
    return RecipeSettings(on_stock_nm=20000, off_stock_nm=200000,
        vector_volume_nl=10000, vector_max_nm=1000, branch_volume_nl=2000,
        branch_budget_nl=1000, capture_c50_nm=50)


@pytest.mark.parametrize('model', ['on_only', 'on_off'])
def test_concentration_gradients_and_two_stage_mass_balance(model):
    rng = np.random.default_rng(5); lib = ConcentrationLibrary(model, 2, 3, 2, settings())
    x = lib.initial(rng); p, back = lib.forward(x); g = rng.normal(size=p.shape)
    numeric = approx_derivative(lambda z: np.sum(lib.forward(z)[0]*g), x).ravel()
    np.testing.assert_allclose(back(g), numeric, atol=1e-8)
    r = lib.recipe(x); s = settings()
    for key in ('source_transfers_nl', 'branch_transfers_nl', 'vector_buffer_nl', 'branch_remaining_nl'):
        assert np.all(r[key] >= 0)
        np.testing.assert_allclose(r[key] % 25, 0)
    np.testing.assert_allclose(r['source_transfers_nl'].sum(axis=(1, 2))+r['vector_buffer_nl'], s.vector_volume_nl)
    np.testing.assert_allclose(r['branch_transfers_nl'].sum(axis=1)+r['branch_remaining_nl'], s.branch_volume_nl)
    np.testing.assert_allclose(r['vector_concentrations_nm'], r['source_transfers_nl']*[s.on_stock_nm, s.off_stock_nm]/s.vector_volume_nl)
    np.testing.assert_allclose(r['branch_concentrations_nm'].reshape(4, -1),
                              r['branch_transfers_nl']@r['vector_concentrations_nm'].reshape(2, -1)/s.branch_volume_nl)
    assert np.all(r['vector_remaining_nl'] >= 0)
    if model == 'on_only':
        assert not np.any(r['source_transfers_nl'][..., 1])


def test_dilution_changes_concentrations_not_dispensed_staple_amount():
    s = settings(); a = ConcentrationLibrary('on_off', 2, 3, 2, s)
    # Fixed full-site amount: doubling volume halves the stock concentration scale.
    b = ConcentrationLibrary('on_off', 2, 3, 2, replace(s, vector_volume_nl=20000, vector_max_nm=500))
    x = a.initial(np.random.default_rng(8)); ra, rb = a.recipe(x), b.recipe(x)
    np.testing.assert_array_equal(ra['source_transfers_nl'], rb['source_transfers_nl'])
    np.testing.assert_array_equal(ra['branch_transfers_nl'], rb['branch_transfers_nl'])
    np.testing.assert_allclose(rb['vector_concentrations_nm'], ra['vector_concentrations_nm']/2)
    np.testing.assert_allclose(rb['branch_concentrations_nm'], ra['branch_concentrations_nm']/2)
    assert np.any(rb['u']+rb['v'] < ra['u']+ra['v'])


def test_known_c50_and_unequal_stock_concentrations():
    s = replace(settings(), vector_max_nm=50, capture_c50_nm=50)
    lib = ConcentrationLibrary('on_off', 1, 1, 1, s)
    # Equal 25 nL volumes from 20/200 uM sources do NOT give 50/50 occupancy.
    r = lib.complete_recipe(np.array([[[25., 25.]]]), np.array([[1000.], [1000.]]))
    assert (r['v']/r['u'])[0, 0, 0] == pytest.approx(10.)
    total = r['branch_concentrations_nm'].sum(axis=-1)
    np.testing.assert_allclose(r['u']+r['v'], 1-2**(-total/50))
    half = lib.complete_recipe(np.array([[[50., 0.]]]), np.array([[1000.], [1000.]]))
    np.testing.assert_allclose(half['u'], .5)
    np.testing.assert_allclose(half['v'], 0.)


def test_budgets_and_vector_stock_availability():
    np.testing.assert_array_equal(quantize_volumes([100, 100], 25), [25, 0])
    s = replace(settings(), vector_volume_nl=100, vector_max_nm=10000)
    lib = ConcentrationLibrary('on_off', 3, 4, 1, s)
    r = lib.recipe(np.zeros(lib.size))
    assert r['source_transfers_nl'].sum() <= 100
    assert r['branch_transfers_nl'].sum() <= 100
    assert r['vector_remaining_nl'][0] >= 0


def test_fit_reports_realized_recipe_error_and_refines_only_legal_transfers():
    d = np.random.default_rng(2).uniform(.1, .8, (4, 3))
    result = fit(d, 'on_off', basis_k=2, recipe_settings=settings(), restarts=1,
                 alternations=2, maxiter=30, recipe_probes=40)
    r = result['recipe']; p = transitions(r['u'], r['v'])[1]
    np.testing.assert_allclose(result['prediction'], p[result['address_indices']])
    assert result['mse'] == pytest.approx(np.mean((result['prediction']-d[result['assignment']])**2))
    assert result['history'][-1]['mse'] <= result['history'][-2]['mse']
    for key in ('source_transfers_nl', 'branch_transfers_nl'):
        assert np.all(r[key] >= 0); assert not np.any(r[key] % 25)
    assert np.all(r['vector_remaining_nl'] >= 0)


@pytest.mark.parametrize('changes', [dict(capture_c50_nm=0), dict(on_stock_nm=float('nan')),
                                    dict(vector_volume_nl=101), dict(branch_budget_nl=3000)])
def test_bad_recipe_settings(changes):
    with pytest.raises(ValueError):
        replace(settings(), **changes).validate()


def test_recipe_rejects_fractional_transfers_and_overdrawn_source():
    lib = ConcentrationLibrary('on_off', 1, 1, 1, settings())
    with pytest.raises(ValueError, match='25 nL'):
        lib.complete_recipe(np.array([[[12.5, 0]]]), np.array([[25.], [25.]]))
    tiny = ConcentrationLibrary('on_off', 1, 1, 1, replace(settings(), vector_volume_nl=25))
    with pytest.raises(ValueError, match='volume budget'):
        tiny.complete_recipe(np.array([[[25., 0]]]), np.array([[25.], [25.]]))


def test_progress_includes_discrete_refinement():
    updates = []
    fit(np.full((4, 3), .4), 'on_off', basis_k=2, recipe_settings=settings(), restarts=1,
        alternations=1, maxiter=5, recipe_probes=8, progress=lambda f, d: updates.append((f, d)))
    assert updates[-1][0] == 1
    assert any('25 nL refinement' in detail and 0 < fraction < 1 for fraction, detail in updates)
    assert all(a[0] <= b[0] for a, b in zip(updates, updates[1:]))


@pytest.mark.parametrize('model', ['pca', 'additive', 'on_only', 'on_off'])
@pytest.mark.parametrize('flexible', [False, True])
def test_global_address_restrictions_survive_fitting_and_quantization(model, flexible):
    d = np.random.default_rng(4).uniform(size=(3, 3))
    allowed = np.array([2, 4, 6, 7])
    result = fit(d, model, allow_partial=True, address_bits=3, allowed_addresses=allowed,
                 flexible=flexible, basis_k=2, restarts=1, alternations=1, maxiter=5,
                 recipe_settings=settings() if model in ('on_only', 'on_off') else None,
                 recipe_probes=8)
    assert result['address_bits'] == 3
    assert result['address_capacity'] == 8
    assert len(set(result['address_indices'])) == 3
    assert set(result['address_indices']) <= set(allowed)
    if not flexible or model == 'pca':
        np.testing.assert_array_equal(result['address_indices'], allowed[:3])
    if 'recipe' in result:
        recipe = result['recipe']
        np.testing.assert_allclose(result['prediction'], transitions(recipe['u'], recipe['v'])[1][result['address_indices']])
