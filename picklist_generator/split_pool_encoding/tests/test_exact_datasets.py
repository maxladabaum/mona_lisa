import json
import numpy as np
import pytest
from split_pool_encoding.datasets import generate, replay_exact_k
from split_pool_encoding.quantization import RecipeSettings, ConcentrationLibrary
from split_pool_encoding.models import transitions


@pytest.mark.parametrize('k', [2, 4, 8])
@pytest.mark.parametrize('response', ['capture', 'rgb_ratio'])
def test_exact_recipe_reconstructs_shuffled_targets(k, response, tmp_path):
    targets, truth = generate(f'exact_k_{k}', bits=4, positions=12, seed=17,
                              response_model=response, extra_rounds=1)
    settings = RecipeSettings(**json.loads(truth['settings_json']))
    library = ConcentrationLibrary('on_off', 5, 12, k, settings, response_model=response)
    recipe = library.complete_recipe(truth['source_transfers_nl'], truth['branch_transfers_nl'])
    expected = transitions(recipe['u'], recipe['v'])[1][truth['source_address']]
    np.testing.assert_array_equal(targets, expected)
    path = tmp_path/'truth.npz'
    np.savez(path, **truth)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_allclose(targets, replay_exact_k(saved), atol=1e-15, rtol=0)
    assert np.ptp(targets) > .1
    assert np.all(truth['source_transfers_nl'] % 25 == 0)
    assert np.all(truth['branch_transfers_nl'].sum(axis=0) > 0)
    again, _ = generate(f'exact_k_{k}', bits=4, positions=12, seed=17,
                        response_model=response, extra_rounds=1)
    np.testing.assert_array_equal(targets, again)


def test_exact_recipe_uses_requested_settings():
    settings = RecipeSettings(10000, 20000, 50000, 200, 5000, 2500, 75)
    target, truth = generate('exact_k_4', recipe_settings=settings)
    assert json.loads(truth['settings_json'])['capture_c50_nm'] == 75
    np.testing.assert_allclose(target, replay_exact_k(truth), atol=1e-15, rtol=0)
