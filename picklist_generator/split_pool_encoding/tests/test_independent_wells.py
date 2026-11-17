import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from split_pool_encoding.quantization import RecipeSettings, ConcentrationLibrary
from split_pool_encoding.optimization import fit


def settings():
    return RecipeSettings(20000, 200000, 10000, 1000, 2000, 1000, 50)


@pytest.mark.parametrize('model', ['on_only', 'on_off'])
def test_independent_well_gradients(model):
    lib = ConcentrationLibrary(model, 3, 4, 3, settings(), wells=5)
    rng = np.random.default_rng(8); x = lib.initial(rng)
    prediction, back = lib.forward(x); g = rng.normal(size=prediction.shape)
    assert prediction.shape == (5, 4)
    numeric = approx_derivative(lambda z: (lib.forward(z)[0]*g).sum(), x).ravel()
    np.testing.assert_allclose(back(g), numeric, atol=1e-8)


def test_mix_concentrations_then_capture_once_and_preserve_well_independence():
    lib = ConcentrationLibrary('on_off', 2, 1, 2, settings(), wells=3)
    source = np.array([[[100, 0]], [[0, 100]]], float)
    transfer = np.array([[100, 100], [200, 0], [0, 0]], float)
    recipe = lib.complete_recipe(source, transfer)
    c = recipe['branch_concentrations_nm']; total = c.sum(axis=-1)
    expected = -np.expm1(-np.log(2)*total/50)*np.divide(c[..., 0], total, out=np.zeros_like(total), where=total>0)
    np.testing.assert_allclose(recipe['u'], expected)
    np.testing.assert_allclose(recipe['u']+recipe['v'], -np.expm1(-np.log(2)*total/50))
    changed = transfer.copy(); changed[0] = 0
    np.testing.assert_array_equal(lib.complete_recipe(source, changed)['u'][1:], recipe['u'][1:])
    swapped = lib.complete_recipe(source[::-1], transfer[:, ::-1])
    np.testing.assert_allclose(swapped['u'], recipe['u'])
    assert recipe['u'][2, 0] == 0


@pytest.mark.parametrize('model', ['on_only', 'on_off'])
def test_fit_independent_recipes_use_unique_identifiers_and_realized_prediction(model):
    data = np.array([[.1, .2], [.3, .7], [.8, .4]])
    result = fit(data, model, synthesis_mode='independent_wells', allow_partial=True,
        address_bits=3, allowed_addresses=np.array([2, 4, 5, 7]), basis_k=4,
        recipe_settings=settings(), restarts=1, alternations=2, maxiter=15, pca_seed=True, recipe_probes=8)
    np.testing.assert_array_equal(result['assignment'], np.arange(3))
    np.testing.assert_array_equal(result['address_indices'], [2, 4, 5])
    assert [t['initialization'] for t in result['trials']] == ['random', 'pca', 'target']
    recipe = result['recipe']
    np.testing.assert_allclose(result['prediction'], recipe['u'])
    assert recipe['branch_transfers_nl'].shape == (3, 4)
    assert np.all(recipe['vector_remaining_nl'] >= 0)
    for key in ('source_transfers_nl', 'branch_transfers_nl'):
        assert not np.any(recipe[key] % 25)


def test_one_vector_per_design_is_exact_in_achievable_continuous_limit():
    data = np.array([[.1, .2], [.3, .7], [.8, .4]])
    lib = ConcentrationLibrary('on_only', 2, 2, 3, settings(), wells=3)
    x = lib.target_initial(data, np.random.default_rng(4))
    np.testing.assert_allclose(lib.forward(x)[0], data, atol=1e-7)


def test_independent_wells_require_physical_recipe():
    with pytest.raises(ValueError, match='require'):
        fit(np.ones((4, 2)), 'on_off', synthesis_mode='independent_wells', basis_k=2)


@pytest.mark.parametrize('model', ['on_only', 'on_off'])
def test_rgb_ratio_gradients(model):
    lib = ConcentrationLibrary(model, 2, 3, 4, settings(), wells=3, response_model='rgb_ratio')
    rng = np.random.default_rng(8); x = lib.initial(rng)
    p, backward = lib.forward(x); g = rng.normal(size=p.shape)
    numerical = approx_derivative(lambda z: (lib.forward(z)[0]*g).sum(), x).ravel()
    np.testing.assert_allclose(backward(g), numerical, atol=1e-8)


def test_same_rgb_recipe_same_ratio_even_after_stock_dilution(monkeypatch):
    from unittest.mock import Mock
    import split_pool_encoding.optimization as optimization
    optimizer = Mock(side_effect=AssertionError('An exact recipe needs no optimization'))
    monkeypatch.setattr(optimization, 'minimize', optimizer)
    source = np.array([[[25, 0], [0, 25], [25, 25]], [[50, 25], [100, 25], [25, 25]]], float)
    molar = source*np.array([settings().on_stock_nm, settings().off_stock_nm])
    target = molar[..., 0]/molar.sum(axis=-1)
    result = fit(target, 'on_off', synthesis_mode='independent_wells', response_model='rgb_ratio',
        basis_k=128, recipe_settings=settings(), direct_source_nl=source,
        restarts=1, alternations=1, maxiter=2, recipe_probes=0)
    assert result['initialization'] == 'original_rgb_recipe'
    assert result['mse'] < 1e-25
    np.testing.assert_allclose(result['prediction'], target, atol=1e-14)
    np.testing.assert_array_equal(result['recipe']['source_transfers_nl'][:2], source)
    np.testing.assert_array_equal(result['recipe']['branch_transfers_nl'][:, :2], np.eye(2)*1000)
    assert result['parameters'] is None
    assert result['exact_recipe_shortcut']
    optimizer.assert_not_called()


def test_capture_default_limits_brightness_while_rgb_ratio_does_not():
    # Defaults tie maximum stock concentration and C50 to the same target.
    from dataclasses import replace
    s = replace(settings(), vector_max_nm=50, branch_budget_nl=2000, capture_c50_nm=50)
    lib = ConcentrationLibrary('on_off', 2, 3, 4, s, wells=3)
    x = lib.target_initial(np.full((3, 3), .9), np.random.default_rng(4))
    assert np.max(lib.forward(x)[0]) <= .5
    ratio_lib = ConcentrationLibrary('on_off', 2, 3, 4, s, wells=3, response_model='rgb_ratio')
    seed = ratio_lib.target_initial(np.full((3, 3), .9), np.random.default_rng(4))
    np.testing.assert_allclose(ratio_lib.forward(seed)[0], .9, atol=1e-6)
