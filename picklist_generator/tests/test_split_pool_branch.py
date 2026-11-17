import numpy as np
import pytest
from split_pool_branch import branch_trace
from split_pool_encoding.library import BasisLibrary


@pytest.mark.parametrize('model', ['additive', 'on_only', 'on_off'])
def test_trace_reconstructs_assigned_address(model):
    lib = BasisLibrary(model, 3, 5, 4)
    x = lib.initial(np.random.default_rng(7))
    prediction, _ = lib.forward(x)
    result = dict(model=model, address_bits=3, assignment=np.array([2, 0, 1]),
                  address_indices=np.array([1, 4, 7]), library_parameters=lib.describe(x))
    for target_row, address in [(2, 1), (0, 4), (1, 7)]:
        trace = branch_trace(result, target_row)
        assert trace['code'] == format(address, '03b')
        np.testing.assert_allclose(trace['on'][-1], prediction[address])
        if model != 'additive':
            np.testing.assert_allclose(trace['on']+trace['off']+trace['vacant'], 1)


def test_actual_recipe_overrides_ideal_and_off_blocks_later_on():
    u = np.zeros((2, 2, 1)); v = np.zeros_like(u)
    v[0, 1] = .75; u[1, 0] = 1
    transfers = np.array([[0, 0], [25, 50], [75, 25], [0, 0]])
    result = dict(model='on_off', address_bits=2, assignment=np.array([0]), address_indices=np.array([2]),
                  library_parameters={}, recipe=dict(u=u, v=v, branch_transfers_nl=transfers))
    trace = branch_trace(result, 0)
    assert trace['code'] == '10'
    np.testing.assert_array_equal(trace['amounts'], [[25, 50], [75, 25]])
    np.testing.assert_allclose(trace['on'][:, 0], [0, 0, .25])
    np.testing.assert_allclose(trace['off'][:, 0], [0, .75, .75])


def test_pca_has_no_physical_branch_trace():
    assert branch_trace({'model': 'pca'}, 0) is None
