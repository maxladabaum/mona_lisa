import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from split_pool_encoding.library import BasisLibrary
from split_pool_encoding.models import addresses, transitions
from split_pool_encoding.optimization import fit


@pytest.mark.parametrize('model', ['additive', 'on_only', 'on_off'])
def test_learned_library_gradients_and_mixture_reconstruction(model):
    rng = np.random.default_rng(3)
    library = BasisLibrary(model, 3, 5, 2); x = library.initial(rng)
    p, back = library.forward(x); g = rng.normal(size=p.shape)
    numeric = approx_derivative(lambda z: (library.forward(z)[0]*g).sum(), x).ravel()
    np.testing.assert_allclose(back(g), numeric, atol=1e-8)
    assert p.min() >= 0 and p.max() <= 1
    params = library.describe(x); b = params['basis_vectors']; w = params['branch_coefficients']
    if model == 'additive':
        rebuilt = params['baseline']+addresses(3)@w@b
    elif model == 'on_only':
        u = (w@b).reshape(3, 2, 5)
        rebuilt = transitions(u, np.zeros_like(u))[1]
    else:
        mixture = (w@b[..., :2].reshape(2, -1)).reshape(3, 2, 5, 2)
        rebuilt = transitions(mixture[..., 0], mixture[..., 1])[1]
    np.testing.assert_allclose(rebuilt, p, atol=1e-12)


@pytest.mark.parametrize('model', ['additive', 'on_only', 'on_off'])
def test_basis_is_trained_for_targets(model):
    rng = np.random.default_rng(19); library = BasisLibrary(model, 2, 4, 2)
    truth = library.initial(rng); d = library.forward(truth)[0]
    initial = library.initial(rng)
    before = np.mean((library.forward(initial)[0]-d)**2)
    result = fit(d, model, basis_k=2, initial=initial, flexible=False,
                 restarts=1, alternations=3, maxiter=100)
    assert result['mse'] < before*.05
    assert not np.allclose(library.describe(initial)['basis_vectors'], result['library_parameters']['basis_vectors'])
    assert result['basis_k'] == 2
    assert result['address_capacity'] == 4
    np.testing.assert_allclose(library.forward(result['parameters'])[0][result['address_indices']], result['prediction'])


@pytest.mark.parametrize('k', [0, -1, 1.5, 257, True])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        BasisLibrary('on_off', 2, 4, k)


def test_optimizer_logging_reuses_evaluated_objective(monkeypatch):
    from types import SimpleNamespace
    import split_pool_encoding.optimization as optimization
    calls = []
    original = BasisLibrary.forward
    def tracked(self, x):
        calls.append(1)
        return original(self, x)
    monkeypatch.setattr(BasisLibrary, 'forward', tracked)
    def minimize(fun, x, **kwargs):
        fun(x)
        count = len(calls)
        kwargs['callback'](x.copy())
        assert len(calls) == count
        return SimpleNamespace(x=x, success=True, message='test')
    monkeypatch.setattr(optimization, 'minimize', minimize)
    fit(np.random.default_rng(3).uniform(size=(4, 3)), 'on_off', basis_k=2,
        restarts=1, alternations=1, maxiter=1)
