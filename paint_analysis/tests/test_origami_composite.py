from dataclasses import replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from origami_composite import (CompositeSettings, encoding_codebook, extract_image_points,
                               reconstruct_composite, export_composite)
from scipy.spatial import cKDTree


def footprint(center=(0, 0), angle=0):
    # Deliberately smaller than the image side: fitting only sees the barcode.
    corners = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return corners @ rotation.T + center


def payload(corners, accepted=None):
    return dict(picks=SimpleNamespace(rectangle_corners_nm=np.array(corners),
                accepted_mask=np.array(accepted if accepted is not None else [True] * len(corners))), params={})


def mapping(color='Red', x=0, y=0, code=''):
    return dict(color=color, x_shift_nm=x, y_shift_nm=y, code=code, include=True)


def settings():
    return CompositeSettings(left_nm=4, top_nm=4, width_nm=4, height_nm=4, pixel_nm=1, sigma_nm=0)


def test_recover_image_outside_barcode_crop_and_rotated_pose():
    local = np.array([[4.5, 3.5], [6.5, 1.5], [0, 0], [9, 9]])
    angle = np.deg2rad(73)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    world = local @ rotation.T + [200, 100]
    ids, points = extract_image_points(world, cKDTree(world), footprint((200, 100), 73), settings())
    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_allclose(points, [[.5, .5], [2.5, 2.5]])


def test_rgb_shifts_mean_and_missing_species():
    source = np.array([[4.5, 3.5], [24.5, 3.5], [44.5, 3.5]])
    results = dict(red=payload([footprint(), footprint((20, 0))]),
                   green=payload([footprint((40, 0))]), blue=payload([footprint((60, 0))], [False]))
    mappings = dict(red=mapping(), green=mapping('Green', 1, 2), blue=mapping('Blue', -1, -2))
    result = reconstruct_composite(source, results, mappings, settings())
    density = result['density']
    assert density[:, :, 0].sum() == 1
    assert density[:, :, 1].sum() == 1
    assert density[:, :, 2].sum() == 0
    np.testing.assert_array_equal(np.argwhere(density[:, :, 1]), [[4, 2]])
    assert result['report']['species'][2]['missing']
    assert not result['coverage'][:, :, 2].any()
    summed = reconstruct_composite(source, results, mappings, replace(settings(), average_species=False))
    assert summed['density'][:, :, 0].sum() == 2


def test_inner_markers_preserved_but_ambiguous_overlapping_signal_excluded():
    source = np.array([[4.5, 3.5], [6.5, 1.5]])
    results = dict(a=payload([footprint()]))
    excluded = reconstruct_composite(source, results, dict(a=mapping()), replace(settings(), exclusion_radius_nm=.2),
                                     dict(a=[[4.5, 3.5]]))
    assert excluded['density'].sum() == 2
    overlap = reconstruct_composite(source, dict(a=payload([footprint(), footprint()])), dict(a=mapping()), settings())
    assert overlap['density'].sum() == 0
    assert overlap['report']['shared_localizations_excluded'] == 2


def test_rotation_and_export(tmp_path):
    source = np.array([[4.5, 3.5]])
    args = (source, dict(a=payload([footprint()])), dict(a=mapping()))
    first = reconstruct_composite(*args, settings())
    rotated = reconstruct_composite(*args, replace(settings(), rotation_deg=90.0))
    np.testing.assert_array_equal(rotated['density'], np.rot90(first['density'], -1))
    path = tmp_path / 'composite.png'
    export_composite(rotated, path)
    assert path.exists()
    np.testing.assert_array_equal(np.load(path.with_suffix('.npz'))['density'], rotated['density'])
    assert json.loads(path.with_suffix('.json').read_text())['settings']['rotation_deg'] == 90


def test_import_codebook_uses_physical_placements_not_state_indices():
    geometry = dict(codebook=[dict(code=3, bits=[0, 1, 1], color='R1 G', x_shift=1, y_shift=2,
                                   x_shift_nm=62.2, y_shift_nm=48.8)])
    _, book = encoding_codebook(dict(geometry=geometry))
    assert book['011']['x_shift_nm'] == 62.2
    assert book['011']['y_shift_nm'] == 48.8
    assert book['011']['color'] == 'Green'
    with pytest.raises(ValueError, match='duplicate'):
        encoding_codebook(dict(codebook=geometry['codebook'] * 2))


@pytest.mark.parametrize('change', [dict(pixel_nm=0), dict(width_nm=-1), dict(left_nm=float('nan')),
                                    dict(sigma_nm=-1), dict(rotation_deg=45)])
def test_invalid_geometry(change):
    with pytest.raises(ValueError):
        replace(settings(), **change).validate()


def test_mapping_and_memory_validation():
    source = np.array([[4.5, 3.5]])
    results = dict(a=payload([footprint()]))
    with pytest.raises(ValueError, match='Assign'):
        reconstruct_composite(source, results, {}, settings())
    with pytest.raises(ValueError, match='finite'):
        reconstruct_composite(source, results, dict(a=mapping(x=float('nan'))), settings())
    with pytest.raises(ValueError, match='million'):
        reconstruct_composite(source, results, dict(a=mapping()), replace(settings(), pixel_nm=.00001))
    with pytest.raises(ValueError, match='duplicate'):
        reconstruct_composite(source, dict(a=results['a'], b=results['a']),
                              dict(a=mapping(code='01'), b=mapping(code='01')), settings())


def test_empty_image_region_still_counts_dark_origami_in_average():
    result = reconstruct_composite(np.array([[4.5, 3.5]]), dict(a=payload([footprint(), footprint((20, 0))])),
                                   dict(a=mapping()), settings())
    assert result['density'].sum() == .5
    assert result['report']['species'][0]['accepted_origami'] == 2


def test_vertical_reference_mirror_is_undone_before_composite_placement():
    normal = dict(a=payload([footprint()]))
    reflected = dict(a=payload([footprint()]))
    reflected['a']['params']['mirror_template_y'] = True
    a = reconstruct_composite(np.array([[4.5,3.5]]), normal, dict(a=mapping()), settings())
    b = reconstruct_composite(np.array([[4.5,.5]]), reflected, dict(a=mapping()), settings())
    np.testing.assert_array_equal(a['density'], b['density'])
    assert b['report']['species'][0]['mirror_template_y']


def test_hidden_sixth_template_error_names_row_and_allows_excluding_it():
    from origami_composite import reconstruction_mapping_issues
    names = ['34', '12', '4', '3', '2', '1']
    results = {name: payload([footprint((20*i, 0))]) for i, name in enumerate(names)}
    mappings = {name: mapping(color=color) for name, color in zip(names, ['Blue','Green','Red','Red','Red',''])}
    points = np.array([[4.5+20*i,3.5] for i in range(6)])
    issues = reconstruction_mapping_issues(results, mappings)
    assert [name for name, _ in issues] == ['1']
    with pytest.raises(ValueError, match='Template "1"'):
        reconstruct_composite(points, results, mappings, settings())
    mappings['1']['include'] = False
    assert reconstruction_mapping_issues(results, mappings) == []
    result = reconstruct_composite(points, results, mappings, settings())
    assert result['report']['excluded_templates'] == ['1']
    mappings['1'].update(include=True, color='Blue')
    assert reconstruction_mapping_issues(results, mappings) == []
    result = reconstruct_composite(points, results, mappings, settings())
    assert len(result['report']['species']) == 6  # Blank code fields are valid manual mappings.


def test_mapping_errors_list_all_bad_colors_and_shifts():
    from origami_composite import reconstruction_mapping_issues
    rows = dict(a=mapping(color=''), b=mapping(x='bad'), c=mapping())
    issues = reconstruction_mapping_issues(rows, rows)
    assert [name for name, _ in issues] == ['a','b']
    assert 'finite' in issues[1][1]


@pytest.mark.parametrize('rotation', [0,90,180,270])
@pytest.mark.parametrize('mirror_x,mirror_y', [(False,False),(True,False),(False,True),(True,True)])
def test_live_orientation_matches_build_and_preserves_density(rotation, mirror_x, mirror_y):
    from origami_composite import reorient_composite
    base_settings=replace(settings(),width_nm=6)
    args=(np.array([[4.5,3.5],[6.5,3.5],[4.5,1.5]]),dict(a=payload([footprint()])),dict(a=mapping()))
    original=reconstruct_composite(*args,base_settings)
    intermediate=reorient_composite(original,90,True,False)
    updated=reorient_composite(intermediate,rotation,mirror_x,mirror_y)
    direct=reconstruct_composite(*args,replace(base_settings,rotation_deg=rotation,mirror_x=mirror_x,mirror_y=mirror_y))
    for key in ('density','coverage','rgb'):
        np.testing.assert_array_equal(updated[key],direct[key])
        np.testing.assert_array_equal(reorient_composite(updated,0)[key],original[key])
    assert original['report']['settings']['rotation_deg']==0
    assert original['report']['settings']['mirror_x'] is False
    assert updated['report']['settings']['mirror_x']==mirror_x
    assert updated['density'].sum()==3


def test_orientation_export_and_mapping_metadata_agree(tmp_path):
    from origami_composite import reorient_composite
    from PIL import Image
    original=reconstruct_composite(np.array([[4.5,3.5]]),dict(a=payload([footprint()])),dict(a=mapping()),settings())
    original['report']['mapping_configuration']=dict(settings=original['report']['settings'])
    changed=reorient_composite(original,90,True,True)
    path=tmp_path/'mirrored.png'
    export_composite(changed,path)
    np.testing.assert_array_equal(np.array(Image.open(path)),changed['rgb'])
    np.testing.assert_array_equal(np.load(path.with_suffix('.npz'))['density'],changed['density'])
    report=json.loads(path.with_suffix('.json').read_text())
    assert report['settings']==report['mapping_configuration']['settings']
    assert report['settings']['rotation_deg']==90
    assert report['settings']['mirror_x'] and report['settings']['mirror_y']


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
@pytest.mark.parametrize('mirror_x,mirror_y', [(False, False), (True, False), (False, True), (True, True)])
def test_independent_section_orientation_and_rectangular_coverage(rotation, mirror_x, mirror_y):
    cfg = replace(settings(), width_nm=6)
    source = np.array([[4.5, 3.5], [8.5, 2.5], [5.5, .5], [24.5, 3.5]])
    results = dict(a=payload([footprint()]), b=payload([footprint((20, 0))]))
    a = dict(mapping(x=-3, y=-2), rotation_deg=rotation, mirror_x=mirror_x, mirror_y=mirror_y)
    mappings = dict(a=a, b=mapping('Green', 10, 8))
    result = reconstruct_composite(source, results, mappings, cfg)
    expected = np.zeros((4, 6))
    expected[0, 0] = expected[1, 4] = expected[3, 1] = 1
    expected = np.rot90(expected, -rotation // 90)
    if mirror_x:
        expected = np.fliplr(expected)
    if mirror_y:
        expected = np.flipud(expected)
    h, w = expected.shape
    np.testing.assert_array_equal(result['density'][:h, :w, 0], expected)
    assert result['density'][:, :, 0].sum() == 3
    assert result['coverage'][:h, :w, 0].all()
    assert result['coverage'][:, :, 0].sum() == h * w
    np.testing.assert_array_equal(np.argwhere(result['density'][:, :, 1]), [[10, 13]])
    assert result['coverage'][:, :, 1].sum() == 24
    assert result['report']['species'][0]['rotation_deg'] == rotation
    single = reconstruct_composite(source[:3], {'a': results['a']}, {'a': a}, cfg)
    assert single['density'].shape == (h, w, 3)
    np.testing.assert_array_equal(single['density'][:, :, 0], expected)
    assert single['coverage'][:, :, 0].all()


def test_invalid_section_rotation_names_template():
    from origami_composite import reconstruction_mapping_issues
    issues = reconstruction_mapping_issues({'barcode': {}}, {'barcode': dict(mapping(), rotation_deg='invalid')})
    assert issues[0][0] == 'barcode'
    assert 'Section rotation' in issues[0][1]


@pytest.mark.parametrize('average', [True, False])
def test_section_brightness_weights_signal_before_combining(average, tmp_path):
    from PIL import Image
    cfg = replace(settings(), average_species=average, sigma_nm=.5)
    source = np.array([[4.5, 3.5], [24.5, 3.5], [44.5, 3.5], [64.5, 3.5]])
    results = dict(a=payload([footprint(), footprint((20, 0))]),
                   b=payload([footprint((40, 0))]), c=payload([footprint((60, 0))]))
    base_mappings = dict(a=mapping(), b=mapping(), c=mapping('Green'))
    base = reconstruct_composite(source, results, base_mappings, cfg)
    weighted = dict(a=dict(base_mappings['a'], brightness=.5),
                    b=dict(base_mappings['b'], brightness=2),
                    c=dict(base_mappings['c'], brightness=0))
    result = reconstruct_composite(source, results, weighted, cfg)
    # Red sections overlap in composite space and must be weighted separately.
    expected_ratio = 2.5 / 2 if average else 3 / 3
    np.testing.assert_allclose(result['density'][:, :, 0], base['density'][:, :, 0] * expected_ratio)
    assert not result['density'][:, :, 1].any()
    np.testing.assert_array_equal(result['coverage'], base['coverage'])
    assert result['report']['candidates'] == base['report']['candidates']
    path = tmp_path / 'weighted.png'
    export_composite(result, path)
    np.testing.assert_array_equal(np.array(Image.open(path)), result['rgb'])
    np.testing.assert_array_equal(np.load(path.with_suffix('.npz'))['density'], result['density'])
    report = json.loads(path.with_suffix('.json').read_text())
    assert [s['brightness'] for s in report['species']] == [.5, 2, 0]


def test_relative_rgb_brightness_and_all_zero():
    source = np.array([[4.5, 3.5], [24.5, 3.5]])
    results = dict(a=payload([footprint()]), b=payload([footprint((20, 0))]))
    mappings = dict(a=dict(mapping(), brightness=.5), b=mapping('Green'))
    result = reconstruct_composite(source, results, mappings, settings())
    np.testing.assert_array_equal(result['rgb'][0, 0], [128, 255, 0])
    assert result['report']['species'][1]['brightness'] == 1
    for row in mappings.values():
        row['brightness'] = 0
    black = reconstruct_composite(source, results, mappings, settings())
    assert not black['rgb'].any()
    assert not black['density'].any()
    assert np.isfinite(black['density']).all()


@pytest.mark.parametrize('brightness', [-1, 'bad', '', None, np.nan, np.inf])
def test_invalid_brightness_names_template(brightness):
    from origami_composite import reconstruction_mapping_issues
    issues = reconstruction_mapping_issues({'barcode': {}}, {'barcode': dict(mapping(), brightness=brightness)})
    assert issues[0][0] == 'barcode'
    assert 'Brightness' in issues[0][1]


def batch_fixture():
    grid=np.array([[x*9.9+(10 if x>=6 else 0),y*5.1] for x in range(12) for y in range(8)])
    document=dict(runs=[{'run_name':'red'}],geometry=dict(image_columns_left_to_right=['06','05','04','03','02'],
        site_pitch_x_nm=9.9,site_pitch_y_nm=5.1,image_origin_from_C12_H01_nm=[69.4,0],origami_rotation_degrees_clockwise=90),
        designs=[dict(name='red',bits=[1,1,0,0,0,0],color='R1 R',x_shift_nm=49.5,y_shift_nm=40.8)])
    return document,grid


def test_saved_rgb_batch_maps_names_and_geometry():
    from origami_composite import saved_rgb_batch_mapping
    document,grid=batch_fixture()
    _,codes,mappings,settings=saved_rgb_batch_mapping(document,['red','control'],grid,{})
    assert set(mappings)=={'red'}
    assert mappings['red']['color']=='Red'
    assert mappings['red']['x_shift_nm']==49.5
    assert mappings['red']['y_shift_nm']==40.8
    assert settings['width_nm']==49.5
    assert settings['height_nm']==pytest.approx(40.8)
    assert settings['rotation_deg']==90
    assert set(codes)=={'110000'}
    _,_,mapped,_=saved_rgb_batch_mapping(document,['design_0001_code_048'],grid,{})
    assert mapped['design_0001_code_048']==mappings['red']


def test_saved_rgb_batch_rejects_duplicates_and_incompatible_geometry():
    from origami_composite import saved_rgb_batch_mapping
    document,grid=batch_fixture()
    with pytest.raises(ValueError):saved_rgb_batch_mapping({},[],grid,{})
    with pytest.raises(ValueError):saved_rgb_batch_mapping(document,[],grid,{'columns':6})
    document['designs']*=2
    with pytest.raises(ValueError,match='duplicate'):saved_rgb_batch_mapping(document,[],grid,{})


@pytest.mark.parametrize('mirror_x,mirror_y', [(True,False),(False,True),(True,True)])
def test_imported_region_and_reconstruction_follow_identification_mirrors(mirror_x, mirror_y):
    from origami_composite import image_region_defaults
    document, grid = batch_fixture()
    grid = grid - grid.mean(axis=0)
    # Asymmetric vertical crop origin exercises reflection beyond a centered rectangle.
    document['geometry']['image_origin_from_C12_H01_nm'][1] = 3
    normal_defaults = image_region_defaults(document['geometry'], grid, {})
    signs = np.array([-1 if mirror_x else 1, -1 if mirror_y else 1])
    params = dict(mirror_all_templates=mirror_x, mirror_template_y=mirror_y)
    reflected_defaults = image_region_defaults(document['geometry'], grid * signs, params)
    normal_settings = CompositeSettings(**normal_defaults, sigma_nm=0)
    reflected_settings = CompositeSettings(**reflected_defaults, sigma_nm=0)
    source = np.array([[normal_settings.left_nm+1.5, normal_settings.top_nm-2.5],
                       [normal_settings.left_nm+12.5, normal_settings.top_nm-18.5]])
    normal = dict(a=payload([footprint()]))
    reflected = dict(a=payload([footprint()]))
    reflected['a']['params'] = params
    a = reconstruct_composite(source, normal, dict(a=mapping()), normal_settings)
    b = reconstruct_composite(source * signs, reflected, dict(a=mapping()), reflected_settings)
    np.testing.assert_array_equal(a['density'], b['density'])
    assert b['density'].sum() == 2


def test_export_figure_labels_physical_size_after_rotation(tmp_path):
    from origami_composite import composite_figure, reorient_composite
    source = np.array([[4.5, 3.5]])
    result = reconstruct_composite(source, dict(red=payload([footprint()])),
                                   dict(red=mapping()), replace(settings(), width_nm=6, pixel_nm=.5))
    result = reorient_composite(result, 90, True, False)
    figure = composite_figure(result)
    axis = figure.axes[0]
    height, width = result['rgb'].shape[:2]
    assert axis.get_xlabel() == 'Display X (nm)'
    assert axis.get_ylabel() == 'Display Y (nm)'
    assert tuple(axis.images[0].get_extent()) == (0, width*.5, height*.5, 0)
    np.testing.assert_array_equal(axis.images[0].get_array(), result['rgb'])
    path = tmp_path/'measured.png'
    export_composite(result, path)
    assert path.with_name('measured_axes.png').is_file()
    from PIL import Image
    np.testing.assert_array_equal(np.asarray(Image.open(path)), result['rgb'])


def test_overlap_tracking_saturates_instead_of_wrapping():
    # A byte counter must stay 'shared' even after more than 255 overlaps.
    result = reconstruct_composite(np.array([[4.5, 3.5]]),
        {'a': payload([footprint() for _ in range(257)])}, {'a': mapping()}, settings())
    assert result['density'].sum() == 0
    assert result['report']['shared_localizations_excluded'] == 1


def test_whole_image_smoothing_matches_filter_after_weighted_assembly(tmp_path):
    from scipy.ndimage import gaussian_filter
    source = np.array([[4.5, 3.5], [24.5, 3.5]])
    results = dict(a=payload([footprint()]), b=payload([footprint((20, 0))]))
    mappings = dict(a=dict(mapping('Red'), brightness=2), b=mapping('Blue', 4, 0))
    base = reconstruct_composite(source, results, mappings, settings())
    smoothed = reconstruct_composite(source, results, mappings, replace(settings(), global_sigma_nm=1))
    # New padding leaves room for the second Gaussian's tails.
    padded = np.pad(base['density'], ((4, 4), (4, 4), (0, 0)))
    expected = gaussian_filter(padded, sigma=(1, 1, 0), mode='constant')
    np.testing.assert_allclose(smoothed['density'], expected)
    assert not smoothed['density'][:, :, 1].any()  # no color-channel mixing
    np.testing.assert_array_equal(smoothed['coverage'], np.pad(base['coverage'], ((4,4),(4,4),(0,0))))
    assert smoothed['density'][4, 8, 0] > 0  # red crosses into the adjacent section
    path = tmp_path / 'smoothed.png'
    export_composite(smoothed, path)
    np.testing.assert_array_equal(np.load(path.with_suffix('.npz'))['density'], expected)
    assert json.loads(path.with_suffix('.json').read_text())['settings']['global_sigma_nm'] == 1


def test_whole_image_smoothing_defaults_off_for_old_settings():
    from dataclasses import asdict
    old = asdict(settings())
    old.pop('global_sigma_nm')
    assert CompositeSettings(**old).global_sigma_nm == 0


@pytest.mark.parametrize('sigma', [-1, np.nan, np.inf])
def test_invalid_whole_image_smoothing_is_rejected(sigma):
    with pytest.raises(ValueError):
        replace(settings(), global_sigma_nm=sigma).validate()


def test_chunked_source_matches_full_tree_with_overlaps_and_rotations(monkeypatch):
    import origami_composite as composite
    rng = np.random.default_rng(418)
    source = rng.uniform(-30, 30, (12000, 2)).astype(np.float32)
    # Include duplicates and exact extraction boundaries.
    source = np.concatenate((source, [[4, 4], [4, 4], [8, 0], [6, 2]]))
    results = dict(a=payload([footprint(), footprint((1, 1)), footprint((10, 10), 73)]),
                   b=payload([footprint((-12, -5), 123)]))
    mappings = dict(a=mapping(), b=mapping('Blue', 4, 7))
    exclusions = dict(a=[[5, 3]])
    original = composite._reconstruction_source
    monkeypatch.setattr(composite, '_reconstruction_source',
                        lambda *args: original(*args, chunk_size=137))
    messages = []
    actual = reconstruct_composite(source, results, mappings, settings(), exclusions, messages.append)
    monkeypatch.setattr(composite, '_reconstruction_source', lambda source, *args: source)
    expected = reconstruct_composite(source, results, mappings, settings(), exclusions)
    for key in ('density', 'coverage', 'rgb'):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual['report'] == expected['report']
    assert any('100%' in message for message in messages)


def test_chunked_source_memmap_filters_distant_points_and_preserves_duplicates(tmp_path):
    from origami_composite import _reconstruction_source
    source = np.memmap(tmp_path / 'source.dat', mode='w+', dtype=np.float32, shape=(1003, 2))
    source[:] = 10000
    source[500:502] = [6, 2]
    source[1002] = [4, 4]
    result = _reconstruction_source(source, dict(a=payload([footprint()])), ['a'], settings(), chunk_size=100)
    np.testing.assert_array_equal(result, [[6, 2], [6, 2], [4, 4]])
    assert result.dtype == np.float64
    assert source[0, 0] == 10000
    source[-1] = np.nan
    with pytest.raises(ValueError, match='finite'):
        _reconstruction_source(source, dict(a=payload([footprint()])), ['a'], settings(), chunk_size=100)


def test_no_nearby_source_produces_empty_composite():
    result = reconstruct_composite(np.array([[10000, 10000]]), dict(a=payload([footprint()])),
                                   dict(a=mapping()), settings())
    assert result['density'].sum() == 0
    assert result['report']['species'][0]['accepted_origami'] == 1
    assert result['report']['shared_localizations_excluded'] == 0


def test_build_progress_covers_preparation_rendering_and_finalization():
    messages = []
    reconstruct_composite(np.array([[4.5, 3.5]]), dict(a=payload([footprint()])),
                          dict(a=mapping()), replace(settings(), sigma_nm=1, global_sigma_nm=1),
                          progress=messages.append)
    stages = ['Validating', 'Preparing extraction regions', 'Indexing 1 extraction-region',
              'Scanning source coordinates', 'Assembling', 'nearby source coordinates for reconstruction',
              'Allocating overlap', 'Checking shared signal', 'composite and coverage rasters',
              'Rendering', 'Averaging and smoothing', 'Combining weighted signal',
              'Smoothing the assembled', 'Applying final', 'Scaling RGB', 'Compiling',
              'Composite reconstruction complete.']
    positions = [next(i for i, message in enumerate(messages) if stage in message) for stage in stages]
    assert positions == sorted(positions)
    assert messages[-1] == 'Composite reconstruction complete.'


@pytest.mark.parametrize('fail', [False, True])
def test_progress_heartbeat_updates_opaque_work_and_stops_on_exit(fail):
    import threading
    from origami_composite import _progress_heartbeat
    tick = threading.Event()
    threads = []
    messages = []

    def callback(message):
        messages.append(message)
        if 'working,' in message:
            threads.append(threading.current_thread())
            tick.set()

    try:
        with _progress_heartbeat(callback, interval=.01) as progress:
            progress('Indexing test coordinates…')
            assert tick.wait(2), 'No heartbeat during a blocked operation'
            if fail:
                raise ValueError('test failure')
    except ValueError:
        assert fail
    assert any('Indexing test coordinates… · working,' in message for message in messages)
    assert all(not thread.is_alive() for thread in threads)


def test_export_progress_covers_every_output(tmp_path):
    result = reconstruct_composite(np.array([[4.5, 3.5]]), dict(a=payload([footprint()])),
                                   dict(a=mapping()), settings())
    messages = []
    export_composite(result, tmp_path / 'progress.png', progress=messages.append)
    for step in range(1, 5):
        assert any(f'({step}/4)' in message for message in messages)
    assert messages[-1] == 'Composite export complete.'


def test_excluded_section_can_be_replaced_by_uniform_half_green():
    source = np.array([[4.5, 3.5], [24.5, 3.5]])
    results = dict(red=payload([footprint()]), design_46_code_45=payload([footprint((20, 0))]))
    mappings = dict(red=mapping(), design_46_code_45=dict(mapping('Green', 0, 40.8), include=False, fill_percent='50'))
    cfg = replace(settings(), sigma_nm=1, global_sigma_nm=1, width_nm=6)
    result = reconstruct_composite(source, results, mappings, cfg)
    lower = np.array(result['report']['pre_rotation_origin_nm'])
    xs = lower[0] + (np.arange(result['rgb'].shape[1]) + .5) * cfg.pixel_nm
    ys = lower[1] + (np.arange(result['rgb'].shape[0]) + .5) * cfg.pixel_nm
    section = (xs[None, :] >= 0) & (xs[None, :] < 6) & (ys[:, None] >= 40.8) & (ys[:, None] < 44.8)
    assert section.sum() > 0
    assert np.all(result['rgb'][:, :, 1][section] == 128)
    assert not result['rgb'][:, :, 1][~section].any()
    assert not result['coverage'][:, :, 1].any()
    assert all(row['template'] != 'design_46_code_45' for row in result['report']['candidates'])
    fill = result['report']['synthetic_replacements'][0]
    assert fill['template'] == 'design_46_code_45'
    assert fill['fill_percent'] == 50 and fill['synthetic']
    assert fill['x_shift_nm'] == 0 and fill['y_shift_nm'] == 40.8


def test_uniform_fill_preserves_other_channels_and_follows_display_orientation(tmp_path):
    source = np.array([[4.5, 3.5], [24.5, 3.5]])
    results = dict(red=payload([footprint()]), green=payload([footprint((20, 0))]))
    mappings = dict(red=mapping(), green=dict(mapping('Green'), include=False))
    original = reconstruct_composite(source, results, mappings, settings())
    mappings['green']['fill_percent'] = 50
    result = reconstruct_composite(source, results, mappings, settings())
    np.testing.assert_array_equal(result['rgb'][:, :, 0], original['rgb'][:, :, 0])
    np.testing.assert_array_equal(result['rgb'][:, :, 2], original['rgb'][:, :, 2])
    np.testing.assert_array_equal(result['coverage'], original['coverage'])
    assert np.all(result['rgb'][:, :, 1] == 128)
    cfg = replace(settings(), rotation_deg=90, mirror_x=True)
    rotated = reconstruct_composite(source, results, mappings, cfg)
    np.testing.assert_array_equal(rotated['rgb'], np.fliplr(np.rot90(result['rgb'], -1)))
    path = tmp_path / 'filled.png'
    export_composite(result, path)
    with np.load(path.with_suffix('.npz')) as archive:
        np.testing.assert_array_equal(archive['density'], result['density'])
    assert json.loads(path.with_suffix('.json').read_text())['synthetic_replacements'][0]['fill_percent'] == 50


@pytest.mark.parametrize('percent', ['bad', -1, 101, float('nan'), float('inf')])
def test_invalid_excluded_fill_names_template(percent):
    from origami_composite import reconstruction_mapping_issues
    mappings = dict(green=dict(mapping('Green'), include=False, fill_percent=percent))
    assert reconstruction_mapping_issues({'green': {}}, mappings)[0][0] == 'green'


def test_blank_fill_stays_excluded_and_fill_is_ignored_when_included():
    source = np.array([[4.5, 3.5]])
    results = dict(red=payload([footprint()]), green=payload([footprint()]))
    mappings = dict(red=mapping(), green=dict(mapping('Green'), include=False, fill_percent=''))
    result = reconstruct_composite(source, results, mappings, settings())
    assert not result['rgb'][:, :, 1].any()
    assert result['report']['synthetic_replacements'] == []
    mappings['green'].update(include=True, fill_percent=50)
    result = reconstruct_composite(source, results, mappings, settings())
    assert result['report']['synthetic_replacements'] == []
    assert result['report']['shared_localizations_excluded'] == 1


def test_zero_fill_replaces_only_selected_channel():
    source = np.array([[4.5, 3.5]])
    results = dict(red=payload([footprint()]), green=payload([footprint()]))
    mappings = dict(red=mapping(), green=dict(mapping('Green'), include=False, fill_percent=0))
    result = reconstruct_composite(source, results, mappings, settings())
    assert result['rgb'][:, :, 0].max() == 255
    assert not result['rgb'][:, :, 1].any()
    assert result['report']['synthetic_replacements'][0]['fill_percent'] == 0


def test_gaussian_marker_transmission_and_disabled_exclusion():
    from origami_composite import marker_exclusion_weights
    points = np.column_stack((np.arange(5.), np.zeros(5)))
    expected = 1 - np.exp(-.5 * np.arange(5.) ** 2)
    np.testing.assert_allclose(marker_exclusion_weights(points, [[0, 0]], 1), expected)
    np.testing.assert_array_equal(marker_exclusion_weights(points, [[0, 0]], 0), np.ones(5))
    np.testing.assert_array_equal(marker_exclusion_weights(points, [], 1), np.ones(5))
    # Multiple nearby markers use the nearest site's suppression.
    assert marker_exclusion_weights([[1, 0]], [[0, 0], [2, 0]], 1)[0] == pytest.approx(expected[1])


def test_inner_rectangle_ignores_markers_in_aligned_frame_and_report():
    local = np.array([[4.5, 3.5], [5.5, 3.5], [6.5, 3.5], [7.5, 3.5]])
    theta = np.deg2rad(73)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    source = local @ rotation.T + [200, 100]
    cfg = replace(settings(), exclusion_radius_nm=1)
    result = reconstruct_composite(source, dict(a=payload([footprint((200, 100), 73)])),
                                   dict(a=mapping()), cfg, dict(a=[[4.5, 3.5]]))
    expected = np.ones(4)
    np.testing.assert_allclose(result['density'][0, :, 0], expected, atol=1e-13)
    assert result['report']['species'][0]['weighted_image_localizations'] == pytest.approx(expected.sum())
    assert result['report']['candidates'][0]['weighted_image_localizations'] == pytest.approx(expected.sum())
    assert 'Gaussian' in result['report']['marker_exclusion_meaning']


def test_blend_rectangle_centering_and_gaussian_tails():
    from origami_composite import section_blend_weights
    cfg = replace(settings(), height_nm=45, blend_height_nm=55)
    cfg.validate()
    np.testing.assert_array_equal(cfg.blend_margin, [0, 5])
    weights = section_blend_weights([[2, 0], [2, 45], [2, -5/3], [2, 50], [2, 51], [-1, 20]],
                                    [4, 45], cfg.blend_margin)
    np.testing.assert_allclose(weights, [1, 1, np.exp(-.5), np.exp(-4.5), 0, 0])
    with pytest.raises(ValueError, match='Overlap blend'):
        replace(cfg, blend_height_nm=44).validate()


def test_adjacent_blends_use_extra_signal_without_bright_seam():
    cfg = replace(settings(), blend_height_nm=10, normalize_overlap=True)
    local = np.array([[x, y] for y in np.arange(-2.5, 7) for x in np.arange(4.5, 8)])
    source = np.concatenate((local, local + [50, 0]))
    result = reconstruct_composite(source, dict(a=payload([footprint()]), b=payload([footprint((50, 0))])),
                                   dict(a=mapping(), b=mapping(y=4)), cfg)
    red = result['density'][:, :, 0]
    assert red.shape == (14, 4)
    np.testing.assert_allclose(red[3:11], 1)
    assert 0 < red[0, 0] < red[1, 0] < red[2, 0] < 1
    assert result['report']['species'][0]['image_localizations'] == 40
    assert result['report']['shared_localizations_excluded'] == 0
    # Adjacent sections in a different color do not dilute each other.
    separate = reconstruct_composite(source, dict(a=payload([footprint()]), b=payload([footprint((50, 0))])),
                                     dict(a=mapping(), b=mapping('Green', y=4)), cfg)
    np.testing.assert_allclose(separate['density'][3:7, :, 0], 1)
    assert separate['density'][7, 0, 0] > 0


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
@pytest.mark.parametrize('mirror', [False, True])
def test_blend_tails_follow_section_rotation_and_mirrors(rotation, mirror):
    cfg = replace(settings(), blend_width_nm=8, blend_height_nm=10)
    source = np.array([[3.5, 5.5], [4.5, 3.5], [8.5, -1.5]])
    base = reconstruct_composite(source, dict(a=payload([footprint()])), dict(a=mapping()), cfg)
    transformed = reconstruct_composite(source, dict(a=payload([footprint()])),
                                        dict(a=dict(mapping(), rotation_deg=rotation, mirror_x=mirror)), cfg)
    expected = np.rot90(base['density'], -rotation // 90)
    if mirror:
        expected = np.fliplr(expected)
    np.testing.assert_allclose(transformed['density'], expected)
    assert transformed['report']['species'][0]['image_localizations'] == 3


def test_blend_validation_tolerates_roundoff_but_identifies_smaller_axis():
    cfg = replace(settings(), height_nm=40.800000000000004, blend_height_nm=40.8)
    cfg.validate()
    np.testing.assert_array_equal(cfg.blend_margin, [0, 0])
    with pytest.raises(ValueError, match='Overlap blend height is 5 nm; normal height is 40.8 nm'):
        replace(cfg, blend_height_nm=5).validate()
    with pytest.raises(ValueError, match='Overlap blend width is 3 nm; normal width is 4 nm'):
        replace(cfg, blend_width_nm=3).validate()
    with pytest.raises(ValueError, match='Overlap blend height'):
        replace(cfg, blend_height_nm=-1e-12).validate()


def test_overlap_default_adds_weighted_signals_without_normalizing():
    cfg = replace(settings(), blend_height_nm=10)
    assert not cfg.normalize_overlap
    local = np.array([[x, y] for y in np.arange(-2.5, 7) for x in np.arange(4.5, 8)])
    source = np.concatenate((local, local + [50, 0]))
    results = dict(a=payload([footprint()]), b=payload([footprint((50, 0))]))
    mappings = dict(a=mapping(), b=mapping(y=4))
    combined = reconstruct_composite(source, results, mappings, cfg)
    a_only = reconstruct_composite(source, results, dict(a=mappings['a'], b=dict(mappings['b'], brightness=0)), cfg)
    b_only = reconstruct_composite(source, results, dict(a=dict(mappings['a'], brightness=0), b=mappings['b']), cfg)
    np.testing.assert_allclose(combined['density'], a_only['density'] + b_only['density'])
    assert combined['density'][6, 0, 0] == pytest.approx(1 + np.exp(-.5 * .5 ** 2))
    assert combined['report']['settings']['normalize_overlap'] is False
    from dataclasses import asdict
    legacy = asdict(cfg)
    legacy.pop('normalize_overlap')
    assert not CompositeSettings(**legacy).normalize_overlap


@pytest.mark.parametrize('angle', [0, 73])
def test_marker_exclusion_only_affects_overlap_localizations(angle):
    from origami_composite import image_region_weights
    cfg = replace(settings(), blend_height_nm=10, exclusion_radius_nm=1)
    local = np.array([[4.5, 3.5], [5.5, 5.5], [6.5, 5.5], [7.5, 5.5]])
    markers = np.array([[4.5, 3.5], [5.5, 5.5]])
    expected = np.array([1, 0, np.exp(-1.125) * (1 - np.exp(-.5)),
                         np.exp(-1.125) * (1 - np.exp(-2))])
    np.testing.assert_allclose(image_region_weights(local, markers, cfg), expected)
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    source = local @ rotation.T + [200, 100]
    result = reconstruct_composite(source, dict(a=payload([footprint((200, 100), angle)])),
                                   dict(a=mapping()), cfg, dict(a=markers))
    assert result['density'].sum() == pytest.approx(expected.sum())
    assert result['density'][3, 0, 0] == pytest.approx(1)
    assert result['report']['species'][0]['weighted_image_localizations'] == pytest.approx(expected.sum())
