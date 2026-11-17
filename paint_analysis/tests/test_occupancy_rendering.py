"""Forward-rendering tests, including numerical parity with the picklist code."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from binary_occupancy import OccupancySettings, manifest_layout
from occupancy_rendering import (gaussian_raster, rendering_settings, _sum_kernels,
                                 _missing_mask, gaussian_rgb)
from origami_composite import CompositeSettings, reorient_composite
from test_binary_occupancy import experiment, build


@pytest.mark.parametrize('angle', [0, 90, 180, 270])
@pytest.mark.parametrize('normalize,linear_light', [(False, False), (True, True)])
def test_matches_picklist_gaussian_renderer(monkeypatch, angle, normalize, linear_light):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'picklist_generator'))
    from gaussian_reconstruction import render_gaussian_overlay
    from image_encoding import geometry
    # Unequal site weights and multiple emitters test superposition, not just shape.
    designs = [dict(channel=c, color=('R1 R', 'R1 G', 'R1 B')[c],
                    encoding_scheme='fixed_4x4', origami_rotation_degrees_clockwise=angle,
                    quadrant_normalization_weight=.75, redundancy=2,
                    image_sites=[dict(x_nm=24.3375, y_nm=18.9375, replacement_fraction=.7/(c+1)),
                                 dict(x_nm=34.2375, y_nm=24.0375, replacement_fraction=.2)]) for c in range(3)]
    reference = render_gaussian_overlay(designs, fwhm_nm=7.5, exposure=.8,
        normalize_designs=normalize, compensate_area=True, linear_light=linear_light)
    info = geometry('fixed_4x4', angle)
    manifest = dict(designs=designs, geometry=info, gaussian_reconstruction=reference.metadata)
    rendering = rendering_settings(manifest)
    width, height = info['field_width_nm'], info['field_height_nm']
    offset = np.array([info['pixel_pitch_x_nm'], info['pixel_pitch_y_nm']]) / 2
    populations = {}
    for design in designs:
        values = np.array([s['replacement_fraction'] for s in design['image_sites']])
        populations[design['channel']] = dict(ids=['a','b'], channel=design['channel'],
            gaussian_centers=np.array([[s['x_nm'], s['y_nm']] for s in design['image_sites']]) + offset,
            gaussian_weight=.75 if normalize else .5, occupancy=values, targets=values,
            cells=np.array([[[0, 0], [width, height]]]*2))
    settings = CompositeSettings(0, height, width, height, pixel_nm=.15, rotation_deg=angle)
    actual = gaussian_raster(populations, settings, rendering)
    np.testing.assert_allclose(actual['gaussian_density'], reference.density, atol=1e-16, rtol=1e-12)
    np.testing.assert_array_equal(actual['rgb'], np.asarray(reference.rgb))
    np.testing.assert_allclose(actual['gaussian_target_signal'], actual['gaussian_signal'])
    assert actual['metadata']['linear_rgb_gain'] == reference.metadata['linear_rgb_gain']


def test_circular_unit_integral_fwhm_and_additivity():
    sigma = 7.5 / np.sqrt(8*np.log(2))
    xs = np.arange(-30, 30.001, .15)
    density = _sum_kernels(xs, xs, np.array([[0., 0.]]), np.array([.7]), sigma)
    assert density.sum() * .15**2 == pytest.approx(.7, abs=1e-12)
    np.testing.assert_allclose(density, density.T, atol=1e-16)
    center = len(xs)//2
    assert density[center, center+25] / density[center, center] == pytest.approx(.5, abs=1e-12)
    double = _sum_kernels(xs, xs, np.array([[0., 0.], [0., 0.]]), np.array([.7,.7]), sigma)
    np.testing.assert_allclose(double, 2*density)
    wide = _sum_kernels(xs, xs, np.array([[0., 0.]]), np.array([.7]), sigma*1.2)
    assert wide.max() < density.max()
    assert wide.sum()*.15**2 == pytest.approx(.7, abs=1e-12)


def test_smooth_image_preserves_site_counts_and_saved_gain():
    args = list(experiment(20))
    args[4]['gaussian_reconstruction'] = dict(fwhm_nm=7.5, linear_rgb_gain=40.392, linear_light=False)
    normal = build(args, compare=True)
    wider = build(args, classifier=OccupancySettings(gaussian_fwhm_nm=10))
    for key in ('counts','states','occupancy','valid'):
        np.testing.assert_array_equal(normal['populations']['Red'][key], wider['populations']['Red'][key])
    np.testing.assert_array_equal(normal['density'], wider['density'])
    assert wider['gaussian_density'].max() < normal['gaussian_density'].max()
    assert normal['report']['gaussian_rendering']['width_source'] == 'manifest'
    assert wider['report']['gaussian_rendering']['width_source'] == 'override'
    plane = normal['gaussian_density'][:, :, 0]
    assert len(np.unique(plane)) > 100  # Not flat square cells or a blurred cell raster.
    np.testing.assert_allclose(normal['gaussian_signal'], normal['gaussian_density']*40.392)
    np.testing.assert_allclose(normal['gaussian_difference'], normal['gaussian_signal']-normal['gaussian_overlay_sampled'])
    args[4]['gaussian_reconstruction']['linear_rgb_gain'] *= 2
    exposed = build(args)
    np.testing.assert_array_equal(exposed['gaussian_density'], normal['gaussian_density'])
    np.testing.assert_allclose(exposed['gaussian_signal'], normal['gaussian_signal']*2)


def test_site_centers_use_saved_nm_positions_and_section_transforms():
    args = list(experiment(10))
    # Physical emitter coordinates, not square-cell centers, are authoritative.
    geometry = args[4]['geometry']
    geometry.update(pixel_pitch_x_nm=2.475, pixel_pitch_y_nm=1.275)
    for design in args[4]['designs']:
        for site in design['image_sites']:
            site['x_nm'] = site['pixel_x']*9.9
            site['y_nm'] = site['pixel_y']*5.1
    layout = manifest_layout(args[4], args[1], args[2], args[5], args[3])
    np.testing.assert_allclose(layout['Red']['gaussian_centers'][0], [1.2375, .6375])
    args[2]['Red'].update(rotation_deg=90, mirror_x=True, x_shift_nm=10, y_shift_nm=20)
    transformed = manifest_layout(args[4], args[1], args[2], args[5], args[3])
    np.testing.assert_allclose(transformed['Red']['gaussian_centers'][0], [10.6375, 21.2375])


def test_missing_sites_mask_neighboring_spot_support_and_roundtrip_orientation():
    args = experiment(10)
    normal = build(args)
    uncertain = build(args, classifier=OccupancySettings(manual_uncertain={
        'Red': {str(i): ['H01-06'] for i in range(10)}}))
    assert np.isnan(uncertain['gaussian_density'][0, 0, 0])
    assert np.isnan(uncertain['gaussian_density'][5, 5, 0])
    assert np.isfinite(uncertain['gaussian_density'][-1, -1, 0])
    assert np.isfinite(uncertain['gaussian_density'][:, :, 1:]).all()
    assert normal['populations']['Red']['occupancy'][0] == .7
    assert np.isnan(uncertain['populations']['Red']['occupancy'][0])
    rotated = reorient_composite(uncertain, 90, True, True)
    restored = reorient_composite(rotated, 0)
    for key in ('rgb','site_rgb','density','gaussian_density','gaussian_signal'):
        np.testing.assert_array_equal(restored[key], uncertain[key])


def test_edge_crop_does_not_renormalize_or_wrap():
    sigma = 1.
    xs = np.arange(.005, 10, .01)
    values = _sum_kernels(xs, xs, np.array([[0.,0.]]), np.array([1.]), sigma)
    assert values.sum() * .01**2 == pytest.approx(.25, abs=1e-10)
    assert values[-1, -1] < 1e-40


def test_invalid_and_unresolved_widths_fail_clearly():
    args = experiment(10)
    for fwhm in (-1, 0, np.nan, np.inf):
        with pytest.raises(ValueError, match='FWHM'):
            build(args, classifier=OccupancySettings(gaussian_fwhm_nm=fwhm))
    with pytest.raises(ValueError, match='too coarse'):
        build(args, classifier=OccupancySettings(gaussian_fwhm_nm=1))
    manifest = deepcopy(args[4])
    manifest['gaussian_reconstruction'] = dict(linear_rgb_gain=-1)
    with pytest.raises(ValueError, match='gains'):
        rendering_settings(manifest)


def test_template_heatmaps_separate_barcode_and_localization_multiplicity():
    from occupancy_rendering import template_heatmap_data
    args = list(experiment(10))
    baseline = build(args)
    # Add binding events only in Red: this changes the overlay, not molecular votes.
    red_points = args[0][args[0][:, 1] < 100]
    args[0] = np.concatenate([args[0], red_points, red_points])
    increased = build(args)
    rendering = baseline['report']['gaussian_rendering']
    red = template_heatmap_data(baseline['populations']['Red'], args[3], rendering)
    red_more = template_heatmap_data(increased['populations']['Red'], args[3], rendering)
    green = template_heatmap_data(increased['populations']['Green'], args[3], rendering)
    np.testing.assert_allclose(red_more['overlay'], red['overlay']*3)
    np.testing.assert_array_equal(red_more['occupancy'], red['occupancy'])
    green_before = template_heatmap_data(baseline['populations']['Green'], args[3], rendering)
    np.testing.assert_array_equal(green['overlay'], green_before['overlay'])
    np.testing.assert_array_equal(green['occupancy'], red['occupancy'])
    assert red['grid'].shape == (96, 2)
    assert red['overlay'].shape == red['occupancy'].shape


@pytest.mark.parametrize('mirror_x,mirror_y', [(False,False), (True,False), (False,True), (True,True)])
def test_template_heatmaps_share_aligned_site_coordinates(mirror_x, mirror_y):
    from occupancy_rendering import template_heatmap_data
    args = experiment(10, mirror_x=mirror_x, mirror_y=mirror_y)
    result = build(args)
    p = result['populations']['Red']
    data = template_heatmap_data(p, args[3], result['report']['gaussian_rendering'])
    for field in ('overlay', 'occupancy'):
        y, x = np.unravel_index(np.nanargmax(data[field]), data[field].shape)
        np.testing.assert_allclose([data['xs'][x], data['ys'][y]], p['sites'][0], atol=.6)
    assert p['occupancy'][0] == .7
    # Decoded placement and display rotation do not rotate an aligned-molecule view.
    rotated = template_heatmap_data(p, replace(args[3], rotation_deg=90, mirror_x=True), result['report']['gaussian_rendering'])
    np.testing.assert_array_equal(rotated['occupancy'], data['occupancy'])


def test_template_heatmaps_missing_data_and_grid_overlays():
    from occupancy_rendering import template_heatmap_data
    from binary_occupancy import occupancy_figure
    args = experiment(10)
    args[1]['Blue']['picks'].accepted_mask[:] = False
    result = build(args)
    missing = template_heatmap_data(result['populations']['Blue'], args[3], result['report']['gaussian_rendering'])
    assert np.isnan(missing['overlay']).all()
    assert np.isnan(missing['occupancy']).all()
    fig = occupancy_figure(result, 'Template heatmaps', barcode='Green')
    axes = [ax for ax in fig.axes if ax.images]
    assert len(axes) == 2
    assert 'Green' in fig._suptitle.get_text()
    for ax in axes:
        assert len(ax.collections) == 2  # full lattice and highlighted analog sites
        assert len(ax.collections[0].get_offsets()) == 96
        np.testing.assert_allclose(ax.get_xlim(), result['populations']['Green']['overlay_extent'][:2])
        assert len(ax.patches) == 1  # extraction boundary
    assert len(axes[1].texts) == 40
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert max(ax.title.get_window_extent(renderer).y1 for ax in axes) < fig._suptitle.get_window_extent(renderer).y0
    fig.clear()
    fig = occupancy_figure(result, 'Template heatmaps', barcode='Blue', show_grid=False, show_values=False, full_grid=True)
    axes = [ax for ax in fig.axes if ax.images]
    assert all(not ax.collections and not ax.texts for ax in axes)
    assert axes[0].get_xlim()[0] < result['populations']['Blue']['overlay_extent'][0]
    fig.clear()


def test_gui_template_heatmaps_action_uses_selected_template():
    from types import SimpleNamespace
    from test_composite_export_progress import window_method
    from test_composite_session import Variable
    calls = []
    window = SimpleNamespace(result={'populations': {}}, view=Variable('Composite'),
        status=Variable(''), plot_result=lambda: calls.append('plot'), reveal_preview=lambda: calls.append('reveal'))
    action = window_method('show_template_heatmaps', {})
    action(window)
    assert window.view.get() == 'Template heatmaps'
    assert calls == ['plot','reveal']
    window.result = None
    calls.clear()
    action(window)
    assert not calls
    assert 'Build Binary Occupancy or Compare Both' in window.status.get()
