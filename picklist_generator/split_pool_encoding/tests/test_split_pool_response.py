import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from split_pool_encoding.quantization import ConcentrationLibrary, RecipeSettings
from split_pool_encoding.models import transitions
from split_pool_encoding.optimization import fit


def settings():
    return RecipeSettings(20000, 200000, 10000, 1000, 2000, 1000, 50)


@pytest.mark.parametrize('model', ['on_only', 'on_off'])
def test_saturating_split_pool_gradient(model):
    lib = ConcentrationLibrary(model, 3, 3, 4, settings(), response_model='rgb_ratio')
    rng = np.random.default_rng(3); x = lib.initial(rng)
    p, back = lib.forward(x); g = rng.normal(size=p.shape)
    numerical = approx_derivative(lambda z: (lib.forward(z)[0]*g).sum(), x).ravel()
    np.testing.assert_allclose(back(g), numerical, atol=1e-8)


def test_complete_occupancy_preserves_off_blocking_and_empty_rounds():
    lib = ConcentrationLibrary('on_off', 2, 2, 2, settings(), response_model='rgb_ratio')
    source = np.array([[[25, 0], [25, 0]], [[0, 25], [0, 0]]])
    volumes = np.array([[0, 25], [0, 0], [25, 0], [25, 0]])
    recipe = lib.complete_recipe(source, volumes)
    vacant, on, off = transitions(recipe['u'], recipe['v'])
    np.testing.assert_array_equal(on, [[0, 1], [0, 1], [1, 1], [1, 1]])
    np.testing.assert_array_equal(off[:, 0], [1, 1, 0, 0])
    np.testing.assert_allclose(vacant+on+off, 1)
    np.testing.assert_array_equal(recipe['u'][0, 1], [0, 0])


def test_fitter_preserves_better_initial_quantized_recipe(monkeypatch):
    from types import SimpleNamespace
    import split_pool_encoding.optimization as optimization
    lib = ConcentrationLibrary('on_off', 2, 3, 2, settings())
    x = lib.initial(np.random.default_rng(9))
    recipe = lib.recipe(x)
    data = transitions(recipe['u'], recipe['v'])[1]
    worse = x.copy(); lib.unpack(worse)[1][...] = [-20, -20, 20]
    monkeypatch.setattr(optimization, 'minimize', lambda *a, **k: SimpleNamespace(x=worse, success=True, message='Test candidate'))
    result = fit(data, 'on_off', initial=x, basis_k=2, recipe_settings=settings(),
                 restarts=1, alternations=1, maxiter=1, flexible=False, recipe_probes=0)
    assert result['retained_initial_recipe']
    assert result['mse'] == 0
    np.testing.assert_array_equal(result['parameters'], x)
    np.testing.assert_array_equal(result['recipe']['source_transfers_nl'], recipe['source_transfers_nl'])
    np.testing.assert_array_equal(result['prediction'], data)
