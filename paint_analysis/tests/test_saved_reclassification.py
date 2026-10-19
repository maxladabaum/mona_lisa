from copy import deepcopy
from unittest.mock import Mock, patch

import numpy as np
import pytest

from test_analysis_session import fixture, RECORDS
from analysis_session import save_analysis_session, load_analysis_session
from paint_analysis_gui import PaintAnalysisApp, saved_measurement_classification_params


def saved_fixture():
    state = fixture()['state']
    picks = state['origami_pick_result']
    n = len(picks.regions)
    assert n > 0
    model = dict(bit_ids=('a',), bit_cells=((0,),), bit_physical_cells=((0,),),
                 bit_brightness_factors=(1.,), alignment_cells=(), physical_shape=(2, 2), active_bits=(False,))
    template = dict(name='off', image=np.zeros((2, 2)), rows=2, columns=2,
                    spacing_x_nm=10., spacing_y_nm=10., rectangle_margin_nm=1.,
                    template_pixel_size_x_nm=1., template_pixel_size_y_nm=1., logical_model=model)
    params = dict(source_path=state['loaded'].path, shared_alignment_template=template,
                  digital_pixel_model=model, logical_model=model,
                  digital_pixel_ids=('a',), digital_pixel_probabilities=np.full((n, 1), .9),
                  digital_group_localization_evidence=np.full((n, 1), 10.),
                  digital_group_prominences=np.full((n, 1), .8),
                  classification_lookup_eligible=(True,)*n, min_site_localizations=1,
                  min_site_evidence=.1, min_candidate_points=1, connect_distance_nm=5., pick_bin_size_nm=2.)
    return picks, params, template


def test_round_trip_reclassifies_new_pattern_without_detection_or_measurement(tmp_path):
    picks, params, template = saved_fixture()
    path = tmp_path/'saved.paintanalysis'
    save_analysis_session(path, dict(picks=picks, params=params))
    saved = load_analysis_session(path, RECORDS)
    on = deepcopy(template)
    on.update(name='new_on', logical_model=dict(template['logical_model'], active_bits=(True,)))
    prepared = saved_measurement_classification_params(saved['picks'], saved['params'], [template, on])
    app = PaintAnalysisApp.__new__(PaintAnalysisApp)
    for name in ('_identify_origami_with_params', '_remeasure_origami_sites', '_measure_direct_digital_groups'):
        setattr(app, name, Mock(side_effect=AssertionError('must reuse saved measurements')))
    with patch('paint_analysis_gui.pick_origami_candidates', side_effect=AssertionError('must not detect')):
        kind, result = app._identify_origami_worker(np.empty((0, 2)), prepared, lambda *args: None)
    assert kind == 'origami_multi_picks'
    assert result['counts'].tolist() == [0, len(picks.regions)]
    assert result['templates'][1]['picks'].aligned_regions is saved['picks'].aligned_regions
    np.testing.assert_array_equal(prepared['digital_pixel_probabilities'], params['digital_pixel_probabilities'])
    # A second reclassification must also work after a save, including previously unclassified particles.
    again = saved_measurement_classification_params(result['templates'][1]['picks'], result['templates'][1]['params'], [template])
    with patch('paint_analysis_gui.pick_origami_candidates', side_effect=AssertionError('must not detect')):
        _, result2 = app._identify_origami_worker(np.empty((0, 2)), again, lambda *args: None)
    assert result2['counts'].tolist() == [0]
    assert result2['unclassified_count'] == len(picks.regions)
    third = saved_measurement_classification_params(result2['templates'][0]['picks'], result2['templates'][0]['params'], [on])
    with patch('paint_analysis_gui.pick_origami_candidates', side_effect=AssertionError('must not detect')):
        _, result3 = app._identify_origami_worker(np.empty((0, 2)), third, lambda *args: None)
    assert result3['counts'].tolist() == [len(picks.regions)]


def test_rejects_changed_groups_and_missing_measurements():
    picks, params, template = saved_fixture()
    changed = deepcopy(template)
    changed['logical_model']['bit_cells'] = ((1,),)
    with pytest.raises(ValueError, match='layout'):
        saved_measurement_classification_params(picks, params, [changed])
    del params['digital_pixel_probabilities']
    with pytest.raises(ValueError, match='missing or stale'):
        saved_measurement_classification_params(picks, params, [template])


def test_only_unclassified_subset_is_processed_and_existing_classes_survive(tmp_path):
    from dataclasses import replace
    from origami_analysis import concatenate_origami_pick_results
    from saved_reclassification import unclassified_subset, merge_unclassified_results
    import paint_analysis_gui as gui

    one, saved, old_template = saved_fixture()
    copies = [replace(one, regions=[r + [i*100., 0] for r in one.regions],
                      bounds_nm=one.bounds_nm + [i*100., i*100., 0, 0]) for i in range(5)]
    picks = concatenate_origami_pick_results(copies)
    picks = replace(picks, accepted_mask=np.array([True, False, False, False, False]))
    saved.update(custom_templates=[old_template],
                 classification_dispositions=('classified as off', 'unclassified: no template passed',
                     'excluded: alignment rejected at Step 2', 'suppressed duplicate of a stronger nearby fit',
                     'unclassified: no template passed'),
                 classification_lookup_eligible=(True, True, False, True, True),
                 classification_qc_eligible=(True, True, False, True, True),
                 digital_pixel_probabilities=((.9,), (.9,), (.9,), (.9,), (.1,)),
                 digital_group_localization_evidence=((10.,),)*5,
                 digital_group_prominences=((.8,),)*5)
    previous = {'off': dict(picks=picks, params=saved)}
    path = tmp_path/'completed.paintanalysis'
    save_analysis_session(path, dict(picks=picks, params=saved, previous=previous))
    loaded = load_analysis_session(path, RECORDS)
    picks, saved, previous = loaded['picks'], loaded['params'], loaded['previous']
    indices, subset, sliced = unclassified_subset(picks, saved, previous)
    assert indices.tolist() == [1, 4]
    assert len(subset.regions) == 2
    new_template = deepcopy(old_template)
    new_template.update(name='new_on', logical_model=dict(old_template['logical_model'], active_bits=(True,)))
    prepared = saved_measurement_classification_params(subset, sliced, [new_template])
    app = PaintAnalysisApp.__new__(PaintAnalysisApp)
    for name in ('_identify_origami_with_params', '_remeasure_origami_sites', '_measure_direct_digital_groups'):
        setattr(app, name, Mock(side_effect=AssertionError('must reuse measurements')))
    with patch('paint_analysis_gui.pick_origami_candidates', side_effect=AssertionError('must not detect')), \
         patch('paint_analysis_gui.digital_group_template_evidence', wraps=gui.digital_group_template_evidence) as evidence:
        _, payload = app._identify_origami_worker(np.empty((0, 2)), prepared, lambda *args: None)
    assert evidence.call_args.args[0].shape == (2, 1)
    result = merge_unclassified_results(payload, indices, picks, saved, previous)
    assert result['counts'].tolist() == [1, 1]
    assert result['unclassified_count'] == 1
    old, new = result['templates']
    np.testing.assert_array_equal(old['picks'].accepted_mask, [True, False, False, False, False])
    np.testing.assert_array_equal(new['picks'].accepted_mask, [False, True, False, False, False])
    assert old['picks'] is picks
    assert previous['off']['params']['classification_dispositions'][1].startswith('unclassified:')
    assert new['params']['classification_dispositions'][0] == 'classified as off'
    save_analysis_session(path, result)
    restored = load_analysis_session(path, RECORDS)
    next_previous = {r['name']: r for r in restored['templates']}
    next_indices, _, _ = unclassified_subset(restored['templates'][0]['picks'], restored['params'], next_previous)
    assert next_indices.tolist() == [4]


def test_no_unclassified_does_not_start_work():
    from saved_reclassification import unclassified_subset
    picks, params, template = saved_fixture()
    params['classification_dispositions'] = ('classified as off',)*len(picks.regions)
    with pytest.raises(ValueError, match='no previously unclassified'):
        unclassified_subset(picks, params, {'off': dict(picks=picks, params=params)})
