import numpy as np
import pytest
from split_pool_canvas_build import canvas_build_steps
from split_pool_encoding.quantization import ConcentrationLibrary, RecipeSettings
from split_pool_encoding.library import BasisLibrary
from split_pool_encoding.models import transitions


@pytest.mark.parametrize('response', ['capture', 'rgb_ratio'])
@pytest.mark.parametrize('independent', [False, True])
def test_construction_uses_real_doses_and_matches_final_prediction(response, independent):
    settings = RecipeSettings(20000, 200000, 10000, 1000, 2000, 1000, 50)
    lib = ConcentrationLibrary('on_off', 3, 5, 3, settings, wells=3 if independent else None, response_model=response)
    recipe = lib.recipe(lib.initial(np.random.default_rng(8)))
    addresses = np.array([1, 4, 7]); assignment = np.array([2, 0, 1])
    prediction = recipe['u'] if independent else transitions(recipe['u'], recipe['v'])[1][addresses]
    result = dict(model='on_off', recipe=recipe, synthesis_mode='independent_wells' if independent else 'split_pool',
                  address_bits=3, address_indices=addresses, assignment=assignment, prediction=prediction)
    plan = canvas_build_steps(result, 0)
    np.testing.assert_allclose(plan['final_on'], prediction[1])
    assert plan['code'] == '100'
    captures = [event for event in plan['events'] if event['phase'] == 'capture']
    assert len(captures) == (1 if independent else 3)
    before = np.zeros(5)
    for event in plan['events'][1:]:
        np.testing.assert_allclose(event['before']+event['added'], event['mixture'])
        np.testing.assert_array_equal(event['on_before'], before)
        if event['phase'] == 'addition':
            assert event['amount'] % 25 == 0
            np.testing.assert_allclose(event['added'], recipe['vector_concentrations_nm'][event['vector']-1]*event['amount']/2000)
        else:
            before = event['on_after']


@pytest.mark.parametrize('model', ['on_only', 'on_off'])
def test_ideal_library_construction_matches_fitted_forward_model(model):
    lib = BasisLibrary(model, 2, 4, 3); x = lib.initial(np.random.default_rng(2))
    prediction, _ = lib.forward(x)
    result = dict(model=model, library_parameters=lib.describe(x), address_bits=2,
                  assignment=np.arange(4), prediction=prediction)
    plan = canvas_build_steps(result, 3)
    np.testing.assert_allclose(plan['final_on'], prediction[3])
    assert not plan['physical']


def test_off_blocking_survives_visualized_additions():
    settings = RecipeSettings(20000, 200000, 10000, 1000, 2000, 1000, 50)
    lib = ConcentrationLibrary('on_off', 2, 1, 2, settings, response_model='rgb_ratio')
    recipe = lib.complete_recipe(np.array([[[25, 0]], [[0, 25]]]), np.array([[0, 25], [0, 0], [25, 0], [25, 0]]))
    result = dict(model='on_off', recipe=recipe, address_bits=2, assignment=np.arange(4))
    plan = canvas_build_steps(result, 0)
    for event in plan['events']:
        np.testing.assert_array_equal(event['on_after'], [0])
    np.testing.assert_array_equal(plan['final_off'], [1])


def test_nonphysical_references_have_no_fake_addition_sequence():
    assert canvas_build_steps({'model': 'pca'}, 0) is None
    assert canvas_build_steps({'model': 'additive'}, 0) is None
