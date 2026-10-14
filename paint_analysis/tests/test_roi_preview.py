import numpy as np
import pandas as pd
from unittest.mock import patch

from roi_preview import roi_preview


def test_exact_count_and_raw_coordinates_with_reordered_filtered_rows():
    raw = pd.DataFrame({'x': [20., 30., 40., 50.], 'y': [1., 2., 3., 4.]}, index=[9, 7, 5, 3])
    corrected = pd.DataFrame({'x': [2., 1., np.nan], 'y': [2., 1., 2.]}, index=[3, 7, 9])
    count, points = roi_preview(corrected, raw, (4, 2, 4, 2), 2, chunk_size=1)
    assert count == 2
    np.testing.assert_array_equal(points, [[60, 4], [100, 8]])


def test_large_selection_is_bounded_reproducible_and_does_not_copy_tables():
    n = 100_000
    raw = pd.DataFrame({'x': np.arange(n), 'y': np.zeros(n), 'unused': np.ones(n)})
    # Forbid whole-frame copies and label-based table extraction used by the old path.
    with patch.object(pd.DataFrame, 'copy', side_effect=AssertionError('table copy')):
        count, points = roi_preview(raw, raw, (0, n, -1, 1), 1, limit=100, chunk_size=1000)
    assert count == n
    assert points.shape == (100, 2)
    assert points[:, 0].min() < n/4
    assert points[:, 0].max() > 3*n/4
    _, again = roi_preview(raw, raw, (0, n, -1, 1), 1, limit=100, chunk_size=1000)
    np.testing.assert_array_equal(points, again)
    assert raw.index._cache.get('_engine') is None


def test_empty_and_missing_raw_matches():
    raw = pd.DataFrame({'x': [1.], 'y': [1.]}, index=[2])
    count, points = roi_preview(raw, raw, (10, 20, 10, 20), 1)
    assert count == 0 and points.shape == (0, 2)
    corrected = raw.set_axis([5])
    count, points = roi_preview(corrected, raw, (0, 2, 0, 2), 1)
    assert count == 1 and points.shape == (0, 2)
