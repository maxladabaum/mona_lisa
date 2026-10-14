import copy
from types import SimpleNamespace

import numpy as np

from paint_analysis_gui import (mirror_origami_template_y, mirrored_origami_run_params,
                                PaintAnalysisApp, ORIGAMI_ALIGNMENT_CACHE_KEYS, origami_stage_cache_matches,
                                alignment_template_overlay_points, origami_grid_points)
from origami_analysis import direct_digital_group_localization_evidence


def template():
    return dict(image=np.array([[0.,1.,0.],[1.,0.,0.]]),
                overlay_points_nm=np.array([[-10.,-2.5],[0.,2.5]]), rows=2,columns=3,
                spacing_x_nm=10.,spacing_y_nm=5.,
                logical_model=dict(physical_shape=(2,3),bit_ids=('a','b'),active_bits=(True,False),
                                   bit_cells=((0,1),(5,)),bit_physical_cells=((0,1),(5,)),alignment_cells=(2,),
                                   column_offsets_nm=(0.,0.,0.)))


def test_vertical_mirror_preserves_columns_ids_and_source_arrays():
    original=template(); saved=copy.deepcopy(original)
    flipped=mirror_origami_template_y(original)
    np.testing.assert_array_equal(flipped['image'], original['image'][::-1])
    np.testing.assert_array_equal(flipped['overlay_points_nm'], original['overlay_points_nm']*[1,-1])
    assert flipped['logical_model']['bit_cells']==((3,4),(2,))
    assert flipped['logical_model']['alignment_cells']==(5,)
    assert flipped['logical_model']['bit_ids']==('a','b')
    assert flipped['logical_model']['active_bits']==(True,False)
    assert original['logical_model']==saved['logical_model']
    np.testing.assert_array_equal(original['image'],saved['image'])
    restored=mirror_origami_template_y(flipped)
    np.testing.assert_array_equal(restored['image'],original['image'])
    assert restored['logical_model']==original['logical_model']


def test_detection_and_alignment_share_mirror_and_no_double_flip():
    t=template()
    app=SimpleNamespace(origami_shared_alignment_template=t,
                        origami_mirror_template_y=SimpleNamespace(get=lambda:True))
    detected=PaintAnalysisApp._origami_candidate_template_points(app)
    raw=dict(mirror_template_y=True,shared_alignment_template=t,custom_templates=[t],
             alignment_template_image=t['image'],digital_pixel_model=t['logical_model'],
             candidate_template_points_nm=detected)
    snapshot=mirrored_origami_run_params(raw)
    np.testing.assert_array_equal(alignment_template_overlay_points(np.empty((0,2)),snapshot),detected)
    np.testing.assert_array_equal(snapshot['candidate_template_points_nm'],detected)
    np.testing.assert_array_equal(snapshot['alignment_template_image'],t['image'][::-1])
    assert snapshot['digital_pixel_model']==snapshot['custom_templates'][0]['logical_model']
    assert 'mirror_template_y' in ORIGAMI_ALIGNMENT_CACHE_KEYS
    assert not origami_stage_cache_matches(dict(mirror_template_y=False),dict(mirror_template_y=True),ORIGAMI_ALIGNMENT_CACHE_KEYS)


def test_barcode_evidence_follows_reflected_rows():
    t=template(); grid=origami_grid_points(2,3,10,5,{})
    points=np.repeat(grid[[0,1]],8,axis=0)
    flipped=mirror_origami_template_y(t)
    before=direct_digital_group_localization_evidence([points],grid,t['logical_model']['bit_cells'],assignment_radius_nm=1)
    after=direct_digital_group_localization_evidence([points*[1,-1]],grid,flipped['logical_model']['bit_cells'],assignment_radius_nm=1)
    np.testing.assert_array_equal(before,after)


def test_disabled_leaves_run_snapshot_unchanged():
    params=dict(mirror_template_y=False,shared_alignment_template=template())
    assert mirrored_origami_run_params(params) is params


def test_vertical_mirror_preserves_dark_mask_identity_and_brightness():
    original = template()
    original['logical_model']['dark_groups'] = ({'cells': (0, 2), 'mask_radius_nm': 3.0},)
    original['logical_model']['brightness_factors'] = (0.5, 2.0)
    reflected = mirror_origami_template_y(original)
    assert reflected['logical_model']['dark_groups'][0]['cells'] == (3, 5)
    assert reflected['logical_model']['brightness_factors'] == (0.5, 2.0)
    assert original['logical_model']['dark_groups'][0]['cells'] == (0, 2)
