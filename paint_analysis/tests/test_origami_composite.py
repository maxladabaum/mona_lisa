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


def test_markers_and_ambiguous_overlapping_signal_excluded():
    source = np.array([[4.5, 3.5], [6.5, 1.5]])
    results = dict(a=payload([footprint()]))
    excluded = reconstruct_composite(source, results, dict(a=mapping()), replace(settings(), exclusion_radius_nm=.2),
                                     dict(a=[[4.5, 3.5]]))
    assert excluded['density'].sum() == 1
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
