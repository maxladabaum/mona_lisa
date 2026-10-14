import numpy as np
import pandas as pd
from unittest.mock import patch
from source_selection import select_source_roi, source_points_nm
from paint_analysis_gui import render_fast_density_map, render_classification_source_preview


def test_roi_selects_only_matching_rows_without_full_copy():
    frame = pd.DataFrame({'x': [1., 2., np.nan, 8.], 'y': [1., 2., 3., 8.], 'photons': [10,20,30,40]}, index=[9,7,5,3])
    with patch.object(pd.DataFrame, 'copy', side_effect=AssertionError('copy')):
        selected = select_source_roi(frame, (25,5,25,5), 10, chunk_size=1)
    assert selected.index.tolist() == [9,7]
    assert selected.photons.tolist() == [10,20]
    points = source_points_nm(selected, 10, chunk_size=1)
    np.testing.assert_array_equal(points, [[10,10],[20,20]])
    assert points.flags.c_contiguous


def test_full_selection_shares_storage_and_empty_roi_works():
    frame = pd.DataFrame({'x': [1.,2.], 'y': [3.,4.]})
    for roi in (None, (0,10,0,10)):
        selected = select_source_roi(frame, roi, 1)
        assert np.shares_memory(selected.x.to_numpy(), frame.x.to_numpy())
    assert select_source_roi(frame, (20,30,20,30), 1).empty


def test_chunked_density_matches_histogram_including_upper_edges():
    frame = pd.DataFrame({'x': [0,1,3,4,np.nan,5], 'y': [0,2,3,4,2,5]})
    with patch('paint_analysis_gui.CHUNK_ROWS', 2):
        result = render_fast_density_map(frame, [{'Pixelsize':1,'Width':4,'Height':4}], 1)
    expected, _, _ = np.histogram2d([0,2,3,4], [0,1,3,4], bins=(4,4), range=((0,4),(0,4)))
    np.testing.assert_array_equal(result['image'], expected)
    assert result['n_rendered'] == 4


def test_large_source_preview_avoids_picasso_and_bounds_image():
    frame = pd.DataFrame({'x':np.zeros(1_000_001), 'y':np.zeros(1_000_001)})
    with patch('paint_analysis_gui.render_picasso_map', side_effect=AssertionError('unbounded render')):
        result = render_classification_source_preview(frame, [{'Pixelsize':1,'Width':10000,'Height':10000}], .1, 'gaussian', 1, None)
    assert result['image'].size <= 4_000_000
    assert result['image'].sum() == len(frame)
    assert 'no blur' in result['preview_note']
