from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial import cKDTree

from origami_analysis import (_compare_half_turn_poses, _align_regions_by_image_correlation,
                             _render_candidate_image, identify_origami_regions,
                             concatenate_origami_pick_results)
from paint_analysis_gui import PaintAnalysisApp, ORIGAMI_ALIGNMENT_CACHE_KEYS, origami_orientation_text


def fiducials():
    # Uploaded pattern: upper-left L, lower-left triple, right-hand pairs.
    return np.array([[-55,17.5],[-45,17.5],[-45,12.5],[-55,-7.5],[-55,-12.5],[-55,-17.5],
                     [55,17.5],[55,12.5],[55,-12.5],[55,-17.5]], dtype=float)


def signal(sites=None):
    rng = np.random.default_rng(808)
    return np.vstack([rng.normal(site, .35, (16, 2)) for site in (fiducials() if sites is None else sites)])


def compare(points, sites=None, **kwargs):
    return _compare_half_turn_poses(points, fiducials() if sites is None else sites,
                                    site_radius_nm=3, minimum_site_localizations=5, **kwargs)


def test_wrong_half_turn_is_corrected_and_correct_pose_is_kept():
    for sign in (1, -1):
        original = sign * signal()
        rotation, offset, scores, flipped, ambiguous, clouds = compare(original)
        assert flipped == (sign == -1)
        assert not ambiguous
        assert scores[int(flipped)] - scores[1-int(flipped)] > .05
        np.testing.assert_allclose(original @ rotation.T + offset, clouds[int(flipped)])
        assert cKDTree(fiducials()).query(clouds[int(flipped)])[0].mean() < .6


def test_independent_small_rotation_and_translation_refinement():
    angle = np.deg2rad(2)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = -signal() @ rotation.T + [.7, -.6]
    _, _, _, flipped, ambiguous, clouds = compare(points)
    assert flipped and not ambiguous
    assert cKDTree(fiducials()).query(clouds[1])[0].mean() < .7


def test_symmetric_template_and_missing_directional_signal_are_ambiguous():
    sites = fiducials()
    common = sites[cKDTree(sites).query(-sites)[0] < .01]
    for template in (sites, common):
        points = np.repeat(common, 8, axis=0)
        _, _, _, flipped, ambiguous, _ = compare(points, template)
        assert ambiguous and not flipped


def test_score_gap_is_configurable_and_symmetric_cannot_be_forced_with_zero_gap():
    _, _, _, flipped, ambiguous, _ = compare(-signal(), minimum_score_gap=1)
    assert ambiguous and not flipped
    sites = np.array([[-20, 0], [20, 0]])
    assert compare(np.repeat(sites, 8, axis=0), sites, minimum_score_gap=0)[4]
    for gap in (-.1, 1.1, float('nan')):
        with pytest.raises(ValueError):
            compare(signal(), minimum_score_gap=gap)


def test_bright_shared_corner_cannot_overwhelm_orientation():
    points = np.vstack((-signal(), np.repeat([[55,17.5]], 3000, axis=0)))
    assert compare(points)[3:5] == (True, False)


def test_alignment_updates_world_pose_and_keeps_paired_metadata():
    sites = fiducials()
    # Force the initial search to the wrong basin; let the paired check recover it.
    region = -signal() + [200, 300]
    template = _render_candidate_image(sites, np.zeros(2), 150, 1, 1)
    diagnostics = []
    pixel = max(1., 2 * np.hypot(130, 55) / 256)
    shift = (np.array([200, 300]) - np.median(region, axis=0)) / pixel
    with (patch('origami_analysis._full_pose_rotation_candidates', return_value=np.array([0.])),
          patch('origami_analysis._translation_hypotheses_to_reference', return_value=[(shift[1], shift[0])])):
        aligned, _, corners, angles, corr, _, _ = _align_regions_by_image_correlation(
            [region], rectangle_width_nm=130, rectangle_height_nm=55, requested_pixel_nm=1,
            max_patch_pixels=256, iterations=1, template_points_nm=sites, template_image=template,
            template_pixel_size_x_nm=1, template_pixel_size_y_nm=1,
            sparse_pose_site_count=0, sparse_site_radius_nm=3, sparse_min_site_localizations=5,
            orientation_check=True, orientation_min_score_gap=.05, orientation_output=diagnostics)
    assert len(diagnostics) == 1
    assert diagnostics[0]['flipped'] and not diagnostics[0]['ambiguous']
    assert abs((angles[0] - 180 + 180) % 360 - 180) < 1
    basis = np.column_stack((corners[0,1]-corners[0,0], corners[0,3]-corners[0,0]))
    basis /= np.linalg.norm(basis, axis=0)
    np.testing.assert_allclose((region-corners[0].mean(axis=0)) @ basis, aligned[0], atol=1e-8)
    assert cKDTree(sites).query(aligned[0])[0].mean() < .8
    assert corr[0] == diagnostics[0]['correlations'][1]


@dataclass
class GatePicks:
    point_counts: np.ndarray
    rectangle_confidence: np.ndarray
    accepted_mask: np.ndarray
    orientation_ambiguous: np.ndarray
    supported_site_count: np.ndarray
    supported_row_count: np.ndarray
    supported_column_count: np.ndarray
    site_spacing_max_error_nm: np.ndarray


def test_ambiguous_pose_stays_rejected_in_both_alignment_and_classification():
    picks = GatePicks(np.array([100,100]), np.ones(2), np.ones(2, bool), np.array([True,False]),
                      np.ones(2),np.ones(2),np.ones(2),np.zeros(2))
    params = dict(orientation_check=True, use_correlation_gate=False)
    for method in (PaintAnalysisApp._apply_origami_alignment_filters, PaintAnalysisApp._apply_origami_acceptance_filters):
        np.testing.assert_array_equal(method(picks, params).accepted_mask, [False, True])
    assert {'orientation_check', 'orientation_min_score_gap'} <= set(ORIGAMI_ALIGNMENT_CACHE_KEYS)


def test_diagnostics_text_reports_both_scores():
    picks=SimpleNamespace(regions=[[]], orientation_scores=np.array([[.2,.8]]),
                          orientation_flipped=np.array([True]), orientation_ambiguous=np.array([False]))
    assert '1 corrected' in origami_orientation_text(picks)
    assert '0.200/0.800' in origami_orientation_text(picks, 0)


def test_identification_and_tile_merge_preserve_orientation_rejection():
    from scipy.ndimage import gaussian_filter
    sites = fiducials()
    common = sites[cKDTree(sites).query(-sites)[0] < .01]
    image = np.zeros((121, 281))
    for x, y in sites:
        image[int(y / .5 + 60), int(x / .5 + 140)] = 1
    image = gaussian_filter(image, 1)
    regions = [signal() + [200, 300], np.repeat(common, 16, axis=0) + [500, 300]]
    points = np.concatenate(regions)
    picks = identify_origami_regions(
        points, pick_bin_size_nm=5, connect_distance_nm=15, density_threshold=.1,
        min_candidate_points=1, max_candidate_points=10000, rows=8, columns=12,
        spacing_x_nm=10, spacing_y_nm=5, rectangle_margin_nm=10,
        alignment_template_image=image, template_pixel_size_x_nm=.5, template_pixel_size_y_nm=.5,
        use_correlation_gate=False, orientation_check=True, site_mask_radius_nm=3,
        min_site_localizations=5, alignment_iterations=1, max_footprint_overlap_fraction=1,
        measure_sites=False, compute_grid_blob_bic=False,
        precomputed_candidates=(regions, np.zeros((1,1)), np.zeros((1,1)), (0,700,0,500), np.zeros((1,1), int)))
    assert picks.orientation_scores.shape == (2, 2)
    np.testing.assert_array_equal(picks.orientation_ambiguous, [False, True])
    np.testing.assert_array_equal(picks.accepted_mask, [True, False])
    merged = concatenate_origami_pick_results([picks, picks])
    np.testing.assert_array_equal(merged.orientation_scores, np.concatenate([picks.orientation_scores]*2))
    np.testing.assert_array_equal(merged.orientation_ambiguous, [False, True, False, True])
    legacy = replace(picks, orientation_scores=np.empty((0,2)), orientation_correlations=np.empty((0,2)),
                     orientation_flipped=np.empty(0, bool), orientation_ambiguous=np.empty(0, bool))
    mixed = concatenate_origami_pick_results([legacy, picks])
    assert np.isnan(mixed.orientation_scores[:2]).all()
    np.testing.assert_array_equal(mixed.orientation_ambiguous, [False, False, False, True])
