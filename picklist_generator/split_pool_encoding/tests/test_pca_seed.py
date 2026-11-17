import numpy as np
import pytest
from split_pool_encoding.library import BasisLibrary
from split_pool_encoding.optimization import fit
from split_pool_encoding.quantization import RecipeSettings


@pytest.mark.parametrize('model', ['additive', 'on_only', 'on_off'])
@pytest.mark.parametrize('k', [1, 4, 128])
def test_pca_seed_valid_for_constant_and_variable_targets(model, k):
    rng = np.random.default_rng(3)
    library = BasisLibrary(model, 3, 5, k)
    for data in (np.zeros((8, 5)), np.ones((8, 5)), rng.uniform(size=(8, 5))):
        initial = library.pca_initial(data, np.random.default_rng(7))
        assert initial.shape == (library.size,)
        assert np.isfinite(initial).all()
        prediction, _ = library.forward(initial)
        assert np.all((prediction >= 0) & (prediction <= 1))
        np.testing.assert_array_equal(initial, library.pca_initial(data, np.random.default_rng(7)))


@pytest.mark.parametrize('physical', [False, True])
def test_pca_extra_candidate_keeps_original_random_baseline_and_address_constraints(physical):
    data = np.random.default_rng(5).uniform(.1, .8, size=(3, 4))
    settings = RecipeSettings(20000, 200000, 10000, 1000, 2000, 1000, 50)
    kwargs = dict(basis_k=4, restarts=2, alternations=2, maxiter=10, seed=4,
                  allow_partial=True, address_bits=3, allowed_addresses=np.array([1, 3, 4, 7]),
                  recipe_settings=settings if physical else None, recipe_probes=8)
    baseline = fit(data, 'on_off', **kwargs)
    progress = []
    seeded = fit(data, 'on_off', **kwargs, pca_seed=True, progress=lambda f, _: progress.append(f))
    assert len(seeded['trials']) == 3
    assert [t['initialization'] for t in seeded['trials']] == ['random', 'random', 'pca']
    np.testing.assert_allclose([t['mse'] for t in seeded['trials'][:2]], [t['mse'] for t in baseline['trials']])
    assert seeded['mse'] <= baseline['mse']
    assert seeded['mse'] == min(t['mse'] for t in seeded['trials'])
    assert set(seeded['address_indices']) <= {1, 3, 4, 7}
    assert np.all(np.diff(progress) >= 0) and progress[-1] == 1
    if physical:
        assert not np.any(seeded['recipe']['source_transfers_nl'] % 25)
        assert not np.any(seeded['recipe']['branch_transfers_nl'] % 25)
