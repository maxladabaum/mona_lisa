import numpy as np
import pytest
from split_pool_encoding.digital_affine import analyze, factor, transform, inverse_transform, unpack, demos


@pytest.mark.parametrize('name,values', list(demos()))
def test_exact_transform_storage_and_feasibility(name, values):
    report, blob, model = analyze(values)
    expected_rank = {'structured_rank_6': 6, 'structured_rank_8': 8, 'arbitrary_random': 63}[name]
    assert report['affine_dimension'] == expected_rank
    assert report['fits_independent_binary_rounds'] == (expected_rank <= 8)
    np.testing.assert_array_equal(unpack(blob), values)
    encoded = transform(values, model)
    assert not encoded[:, expected_rank:].any()
    np.testing.assert_array_equal(inverse_transform(encoded, model), values)
    # Full transform is invertible on unseen values as well; residuals must be kept.
    arbitrary = np.random.default_rng(3).integers(0, 2, (10, 128), dtype=np.uint8)
    np.testing.assert_array_equal(inverse_transform(transform(arbitrary, model), model), arbitrary)
    assert report['total_encoded_bytes'] == len(blob)
    assert (report['storage_saving_fraction'] > 0) == (expected_rank <= 8)


@pytest.mark.parametrize('values', [np.zeros((3, 9), dtype=np.uint8), np.ones((1, 1), dtype=np.uint8), np.eye(12, dtype=np.uint8)])
def test_edge_cases(values):
    report, blob, model = analyze(values)
    np.testing.assert_array_equal(unpack(blob), values)
    np.testing.assert_array_equal(inverse_transform(transform(values, model), model), values)


def test_order_duplicates_and_extra_products():
    _, values = list(demos())[1]
    values = values[[4, 2, 4, 1, 63, 0]]
    report, blob, _ = analyze(values)
    np.testing.assert_array_equal(unpack(blob), values)
    assert report['unique_files'] == 5
    assert report['ordered_selection_manifest_bits'] == 6*8
    with pytest.raises(ValueError):
        unpack(blob[:-1])


@pytest.mark.parametrize('values', [[[.5, 1]], [[0, 2]], [], [0, 1]])
def test_invalid_binary_input(values):
    with pytest.raises(ValueError):
        factor(values)
