from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from binary_occupancy import (OccupancySettings, classify_sites, summarize_states,
    reconstruct_occupancy, manifest_layout, export_occupancy, occupancy_figure,
    subsample_occupancy, export_subsampling, control_rates, display_rgb)
from origami_analysis import ideal_grid_points
from origami_composite import (CompositeSettings, reconstruct_composite,
    reorient_composite, image_region_defaults)
from composite_session import save_reconstruction_session, load_reconstruction_session
from test_origami_composite import footprint, payload


def experiment(n=200, random=False, mirror_x=False, mirror_y=False):
    sx, sy = 9.9, 5.1
    grid = ideal_grid_points(8, 12, sx, sy, [-5]*6 + [5]*6)
    geometry = dict(site_pitch_x_nm=sx, site_pitch_y_nm=sy,
        image_columns_left_to_right=['06','05','04','03','02'],
        image_origin_from_C12_H01_nm=[6*sx+10, 0],
        pixel_edges_x_nm=(np.arange(6)*sx).tolist(), pixel_edges_y_nm=(np.arange(9)*sy).tolist())
    sites = [dict(name=f'H{2*r+1:02d}-{6-c:02d}', row=f'H{2*r+1:02d}', column=f'{6-c:02d}',
                  pixel_x=c, pixel_y=r, replacement_fraction=.7 if c == 0 and r == 0 else 0)
             for r in range(8) for c in range(5)]
    designs = [dict(name=color, bits=[i//2, i%2], color=f'R1 {color[0]}',
                    x_shift_nm=0, y_shift_nm=0, image_sites=[dict(site) for site in sites])
               for i, color in enumerate(('Red','Green','Blue'))]
    manifest = dict(geometry=geometry, designs=designs, runs=[{}])
    params = dict(mirror_all_templates=mirror_x, mirror_template_y=mirror_y,
                  rows=8, columns=12, spacing_x_nm=sx, spacing_y_nm=sy)
    settings = CompositeSettings(**image_region_defaults(geometry, grid, params), pixel_nm=1, sigma_nm=0)
    results, mappings, grids, source = {}, {}, {}, []
    rng = np.random.default_rng(809)
    expected = {}
    for k, color in enumerate(('Red', 'Green', 'Blue')):
        centers = np.column_stack((np.arange(n)*200, np.full(n, k*200)))
        results[color] = payload([footprint(center) for center in centers])
        results[color]['params'] = params.copy()
        mappings[color] = dict(code=f'{k:02b}', color=color, x_shift_nm=0, y_shift_nm=0, include=True)
        grids[color] = grid
        occupied = rng.random(n) < .7 if random else np.arange(n) < int(n*.7)
        expected[color] = occupied.mean() if n else np.nan
        site = np.array([grid[:, 0].min()+6*sx+10, grid[:, 1].max()]) * [-1 if mirror_x else 1, -1 if mirror_y else 1]
        for i in np.flatnonzero(occupied):
            # Molecules generate wildly different counts, but each has one vote.
            count = 3 + (i % 17)*10
            source.extend(site + centers[i] + rng.normal(0, .04, (count, 2)))
    return np.asarray(source).reshape(-1, 2), results, mappings, settings, manifest, grids, expected


def build(args, **kwargs):
    return reconstruct_occupancy(*args[:6], **kwargs)


def test_exact_140_of_200_equal_weight_and_separate_barcodes():
    args = experiment()
    # Only green has half as many accepted molecules (first 100 are all ON).
    args[1]['Green']['picks'].accepted_mask[100:] = False
    result = build(args)
    red = result['populations']['Red']
    assert red['states'].shape == (200, 40)
    assert red['on'][0] == 140 and red['valid'][0] == 200
    assert red['occupancy'][0] == .7
    assert np.all(red['occupancy'][1:] == 0)
    assert result['populations']['Green']['occupancy'][0] == 1
    assert result['populations']['Green']['rejected_candidates'] == 100
    np.testing.assert_array_equal(result['site_rgb'][0, 0], [179, 255, 179])
    assert red['counts'][:, 0].max() > 100
    assert red['states'][:, 0].max() == 1


def test_bernoulli_synthetic_probabilities():
    args = experiment(random=True)
    result = build(args)
    for name, expected in args[-1].items():
        assert abs(expected - .7) < .1
        assert result['populations'][name]['occupancy'][0] == expected


def test_no_double_counting_overlapping_circular_rois():
    counts, assigned = classify_sites([[1, 0]]*20 + [[0, 0], [20, 20]], [[0, 0], [2, 0]], OccupancySettings(radius_nm=2))
    assert counts.sum() == 21
    assert np.count_nonzero(assigned == -1) == 1
    assert sorted(counts.tolist()) in ([0, 21], [1, 20])


def test_uncertain_denominator_and_zero_valid():
    summary = summarize_states([[1, -1], [0, -1], [-1, -1]])
    np.testing.assert_array_equal(summary['valid'], [2, 0])
    np.testing.assert_array_equal(summary['excluded'], [1, 3])
    assert summary['occupancy'][0] == .5 and np.isnan(summary['occupancy'][1])
    args = experiment(10)
    classifier = OccupancySettings(per_position_thresholds={'H01-06': 50}, uncertainty_margin=1,
        manual_uncertain={'Red': {'0': ['H01-06']}})
    result = build(args, classifier=classifier)
    p = result['populations']['Red']
    assert p['states'][0, 0] == -1 and p['reasons'][0, 0] == 'manual'
    assert p['valid'][0] == 9
    assert p['on'][0] == 2  # counts 53 and 63 pass threshold 50
    with pytest.raises(ValueError, match='Unknown per-position'):
        build(args, classifier=OccupancySettings(per_position_thresholds={'typo': 3}))
    with pytest.raises(ValueError, match='unavailable candidate'):
        build(args, classifier=OccupancySettings(manual_uncertain={'Red': {'99': ['H01-06']}}))


def test_missing_barcode_even_without_classification_template():
    args = list(experiment(10))
    del args[1]['Blue']; del args[2]['Blue']; del args[5]['Blue']
    result = build(args)
    assert np.isnan(result['density'][:, :, 2]).all()
    assert not result['coverage'][:, :, 2].any()
    assert 'missing:10' in result['populations']
    np.testing.assert_array_equal(result['rgb'][0, 0], [255, 0, 255])
    p = result['populations']['missing:10']
    assert p['states'].shape == (0, 40) and np.isnan(p['occupancy']).all()


def test_shared_source_uncertain_not_off_and_roi_truncation():
    args = list(experiment(10))
    args[1]['Red']['picks'].rectangle_corners_nm[1] = args[1]['Red']['picks'].rectangle_corners_nm[0]
    p = build(args)['populations']['Red']
    assert (p['states'][:2, 0] == -1).all()
    assert (p['reasons'][:2, 0] == 'shared_localizations').all()
    p = build(experiment(10), classifier=OccupancySettings(radius_nm=3))['populations']['Red']
    assert (p['states'][:, :5] == -1).all()  # radius exceeds top edge margin
    assert (p['reasons'][:, :5] == 'roi_truncated').all()


@pytest.mark.parametrize('mirror_x,mirror_y', [(False, False), (True, False), (False, True), (True, True)])
def test_alignment_mirrors_preserve_site_identity(mirror_x, mirror_y):
    result = build(experiment(10, mirror_x=mirror_x, mirror_y=mirror_y))
    assert result['populations']['Red']['occupancy'][0] == .7
    np.testing.assert_array_equal(result['site_rgb'][0, 0], [179]*3)
    np.testing.assert_array_equal(result['site_rgb'][-2, -2], [0]*3)


def test_rigid_alignment_and_section_rotation_mapping():
    args = list(experiment(1))
    # Make one molecule with only the design's top-left site ON.
    layout = manifest_layout(args[4], args[1], args[2], args[5], args[3])
    local = layout['Red']['sites'][0]
    theta = np.deg2rad(73)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    args[1] = {'Red': payload([footprint((50, 60), 73)])}
    args[1]['Red']['params'] = {}
    args[2] = {'Red': dict(args[2]['Red'], rotation_deg=90, mirror_x=True)}
    args[5] = {'Red': args[5]['Red']}
    args[0] = np.repeat((local @ rotation.T + [50, 60])[None, :], 3, axis=0)
    result = build(args)
    assert result['populations']['Red']['occupancy'][0] == 1
    assert result['density'][0, 0, 0] == 1  # 90 CW then horizontal reflection => transpose
    assert result['density'][20, 20, 0] == 0


def test_overlay_unchanged_and_common_frame_comparison_reorientation():
    args = experiment(10)
    existing = reconstruct_composite(*args[:4])
    result = build(args, compare=True)
    for key in ('density', 'coverage', 'rgb', 'report'):
        if key == 'report':
            assert existing[key] == result['overlay'][key]
        else:
            np.testing.assert_array_equal(existing[key], result['overlay'][key])
    assert not result['report']['comparison']['quantitative']
    assert result['difference'].shape == result['density'].shape
    np.testing.assert_allclose(result['difference'], result['density']-result['overlay_sampled'])
    rotated = reorient_composite(result, 90, True)
    restored = reorient_composite(rotated, 0)
    for key in ('density', 'coverage', 'rgb', 'target', 'difference', 'overlay_sampled',
                'site_rgb', 'gaussian_density', 'gaussian_signal', 'gaussian_target_signal',
                'gaussian_overlay_sampled', 'gaussian_difference'):
        np.testing.assert_array_equal(restored[key], result[key])
    calibrated = build(args, compare=True, classifier=OccupancySettings(overlay_density_per_on=5))
    assert calibrated['report']['comparison']['quantitative']
    assert calibrated['report']['comparison']['density_scale'] == 5


def test_threshold_band_and_invalid_settings():
    args = experiment(10)
    p = build(args, classifier=OccupancySettings(uncertainty_margin=1))['populations']['Red']
    assert p['states'][0, 0] == -1  # exactly 3 localizations in [2,4)
    assert p['states'][7, 0] == 0  # zero is still a valid OFF
    for settings in (OccupancySettings(radius_nm=0), OccupancySettings(min_localizations=0),
                     OccupancySettings(min_localizations=2.1), OccupancySettings(uncertainty_margin=-1)):
        with pytest.raises(ValueError):
            settings.validate()
    with pytest.raises(ValueError, match='saved RGB'):
        reconstruct_occupancy(*args[:4], None, args[5])


def test_exports_plots_subsampling_and_endpoint_controls(tmp_path):
    result = build(experiment(20), compare=True)
    path = tmp_path / 'occupancy.png'
    export_occupancy(result, path)
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as archive:
        assert archive['population_0_states'].shape == (20, 40)
        np.testing.assert_array_equal(archive['gaussian_density'], result['gaussian_density'])
        assert archive['gaussian_missing'].dtype == bool
        assert archive['population_0_site_ids'][0] == 'H01-06'
    import csv
    rows = list(csv.DictReader(path.with_name('occupancy_individual.csv').open()))
    assert len(rows) == 20*40*3
    assert rows[0]['state'] == '1' and rows[0]['localizations'] == '3'
    assert json.loads(path.with_suffix('.json').read_text())['occupancy_settings']['min_localizations'] == 3
    for view in ('Individual sites', 'Count distributions', 'Barcode maps', 'Target errors', 'Composite', 'Red', 'Coverage', 'Site occupancy'):
        figure = occupancy_figure(result, view)
        figure.savefig(tmp_path / (view+'.png'))
        figure.clear()
    samples = subsample_occupancy(result, (5, 20, 100), repeats=2, seed=42)
    assert samples == subsample_occupancy(result, (5, 20, 100), repeats=2, seed=42)
    full = next(row for row in samples if row['sample_size'] == 20)
    assert full['occupancy'][0] == .7
    assert all(row['status'] == 'insufficient' for row in samples if row['sample_size'] == 100)
    export_subsampling(result, tmp_path/'samples.json', (5, 100), repeats=2, seed=42)
    assert len(list((tmp_path/'samples_images').glob('*.npz'))) == 4
    controls = control_rates(result['populations'])
    assert controls['false_positive_fraction'] == 0
    assert controls['false_negative_fraction'] is None


def test_reconstruction_archive_roundtrip_keeps_matrices_and_config(tmp_path):
    args = experiment(10)
    classifier = OccupancySettings(per_position_thresholds={'H01-06': 20}, manual_uncertain={'Red': {'0': ['H01-06']}})
    original = build(args, classifier=classifier)
    configuration = dict(format='paint-composite-mappings-v1', settings=asdict(args[3]), mappings=args[2],
        rgb_batch_manifest=args[4], occupancy_settings=asdict(classifier), reconstruction_mode='Binary Occupancy (New)')
    path = tmp_path/'binary.paintrecon'
    save_reconstruction_session(path, args[0], args[1], configuration, {}, args[5])
    loaded = load_reconstruction_session(path)
    results = loaded.origami_multi_template_results
    restored = reconstruct_occupancy(loaded.origami_source_points_nm, results, args[2], args[3], args[4],
        {n: p['params']['_reconstruction_grid_nm'] for n, p in results.items()}, classifier)
    np.testing.assert_array_equal(restored['density'], original['density'])
    for name in results:
        np.testing.assert_array_equal(restored['populations'][name]['states'], original['populations'][name]['states'])
    assert loaded.origami_composite_configuration == configuration


def test_all_40_positions_keep_manifest_order_and_rgb_assignment():
    args = list(experiment(10))
    # Permute manifest ordering, then synthesize an asymmetric per-site pattern.
    rng = np.random.default_rng(27)
    for design in args[4]['designs']:
        order = rng.permutation(40)
        design['image_sites'] = [design['image_sites'][j] for j in order]
    layout = manifest_layout(args[4], args[1], args[2], args[5], args[3])
    source = []
    expected = {}
    for k, (name, row) in enumerate(layout.items()):
        expected[name] = (np.arange(40) + k) % 11 / 10
        for j, site in enumerate(row['sites']):
            for n in range(int(round(expected[name][j]*10))):
                source.extend([site + [n*200, k*200]]*3)
    args[0] = np.asarray(source)
    result = build(args)
    for k, (name, p) in enumerate(result['populations'].items()):
        np.testing.assert_allclose(p['occupancy'], expected[name])
        for j, cell in enumerate(p['cells']):
            x, y = np.floor(cell.mean(axis=0)).astype(int)
            assert result['density'][y, x, k] == expected[name][j]


def test_brightness_smoothing_fill_do_not_change_occupancy():
    args = list(experiment(10))
    baseline = build(args)
    args[3] = replace(args[3], sigma_nm=3, global_sigma_nm=2, average_species=False)
    for mapping in args[2].values():
        mapping['brightness'] = 25
    actual = build(args)
    np.testing.assert_array_equal(actual['density'], baseline['density'])
    args[2]['Blue'].update(include=False, fill_percent=100)
    actual = build(args)
    assert np.isnan(actual['density'][:, :, 2]).all()


def test_empty_populations_remain_missing_and_ambiguous_pose_is_uncertain():
    args = experiment(10)
    for p in args[1].values():
        p['picks'].accepted_mask[:] = False
    result = build(args)
    assert np.isnan(result['density']).all()
    assert not result['coverage'].any()
    args = experiment(10)
    args[1]['Red']['picks'].orientation_ambiguous = np.ones(10, bool)
    p = build(args)['populations']['Red']
    assert (p['states'] == -1).all()
    assert (p['reasons'] == 'alignment_ambiguous').all()


def test_interleaved_designs_use_existing_pixel_lookup_not_tile_overlays():
    args = list(experiment(10))
    # Two X phases each contain 40 analog sites but map to alternate fine pixels.
    sx = args[4]['geometry']['site_pitch_x_nm']
    args[4]['geometry']['pixel_edges_x_nm'] = (np.arange(11)*sx/2).tolist()
    for k, design in enumerate(args[4]['designs']):
        phase = k % 2
        design['x_shift_nm'] = phase*sx/2
        args[2][design['name']]['x_shift_nm'] = phase*sx/2
        for site in design['image_sites']:
            site['pixel_x'] = 2*site['pixel_x'] + phase
    result = build(args)
    # Red/blue phase 0 and green phase 1 are sampled independently.
    assert result['density'][0, 0, 0] == .7
    assert np.isnan(result['density'][0, 0, 1])
    assert result['density'][0, 5, 1] == .7
    assert np.isnan(result['density'][0, 5, 0])


def test_gui_occupancy_settings_validate_and_restore():
    from test_composite_export_progress import window_method
    from test_composite_session import Variable
    fields = dict(radius_nm=Variable('2'), min_localizations=Variable('3'),
                  uncertainty_margin=Variable('0'), overlay_density_per_on=Variable(''), gaussian_fwhm_nm=Variable(''))
    window = SimpleNamespace(occupancy_variables=fields, occupancy_advanced={})
    namespace = dict(OccupancySettings=OccupancySettings, asdict=asdict)
    get = window_method('occupancy_settings', namespace)
    apply = window_method('apply_occupancy_settings', namespace)
    assert get(window) == OccupancySettings()
    settings = dict(radius_nm=1.5, min_localizations=5, uncertainty_margin=1, gaussian_fwhm_nm=8.0,
                    per_position_thresholds={'H01-06': 7}, manual_uncertain={'Red': {'0': ['H01-06']}})
    apply(window, settings)
    assert get(window) == OccupancySettings(**settings)
    before = get(window)
    with pytest.raises(ValueError):
        apply(window, dict(radius_nm=-1))
    assert get(window) == before
