from copy import deepcopy
import threading
from unittest.mock import Mock

import numpy as np

from test_tile_checkpoint import run_fixture
from test_analysis_view_restore import restore_tk_default_root
from origami_composite import CompositeSettings, reconstruct_composite


def control_for(live, directory, run):
    control = dict(directory=directory, run=run, pause=threading.Event(), state='running')
    live._tiled_control = control
    live.origami_pause_tiles_button = Mock()
    live.origami_continue_tiles_button = Mock()
    live._set_origami_step_progress = Mock()
    return control


def test_pause_saves_current_tile_builds_partial_composite_and_resumes(tmp_path):
    live, run, payload = run_fixture()
    control = control_for(live, tmp_path, run)
    def classify(*args):
        live.pause_tiled_analysis()
        return 'origami_multi_picks', deepcopy(payload)
    live._identify_origami_worker = Mock(side_effect=classify)
    kind, partial = live._checkpointed_tiled_worker(tmp_path, run, initialize=True)
    assert kind == 'origami_multi_picks'
    assert (tmp_path / 'tile-00000000.paintanalysis').exists()
    assert not (tmp_path / 'tile-00000001.paintanalysis').exists()
    assert partial['params']['_tiled_paused']
    assert partial['params']['_checkpoint_preview'] == dict(completed=1, planned=2)
    assert not partial['params']['_tiled_full_field_display']
    np.testing.assert_array_equal(partial['counts'], payload['counts'])
    assert len(partial['locs']) < len(run['context']['source_locs'])
    # The partial payload provides the same inputs used by the composite editor.
    templates = {item['name']: dict(picks=item['picks'], params=item['params']) for item in partial['templates']}
    mappings = {name: dict(include=True, color='Red', x_shift_nm=0, y_shift_nm=0) for name in templates}
    composite = reconstruct_composite(partial['points_nm'], templates, mappings,
        CompositeSettings(left_nm=-10, top_nm=10, width_nm=20, height_nm=20, pixel_nm=2, sigma_nm=0))
    assert np.isfinite(composite['density']).all()
    live._complete_tiled_pause_control(partial)
    assert control['state'] == 'paused'
    assert 'Paused: 1/2' in live.status.get()
    # Continue uses the original in-memory run; no software/source reload dialog.
    live._resume_tiled_checkpoint = Mock()
    live.continue_tiled_analysis()
    live._resume_tiled_checkpoint.assert_called_once_with(tmp_path, run)
    before = (tmp_path / 'tile-00000000.paintanalysis').read_bytes()
    control_for(live, tmp_path, run)
    live._identify_origami_worker = Mock(return_value=('origami_multi_picks', deepcopy(payload)))
    _, complete = live._checkpointed_tiled_worker(tmp_path, run)
    live._identify_origami_worker.assert_called_once()
    assert (tmp_path / 'tile-00000000.paintanalysis').read_bytes() == before
    np.testing.assert_array_equal(complete['counts'], payload['counts'] * 2)
    assert '_tiled_paused' not in complete['params']
    live._complete_tiled_pause_control(complete)
    assert live._tiled_control is None


def test_pause_requested_before_first_tile_still_produces_one_complete_tile(tmp_path):
    live, run, payload = run_fixture()
    control_for(live, tmp_path, run)
    live.pause_tiled_analysis()
    live._identify_origami_worker = Mock(return_value=('origami_multi_picks', deepcopy(payload)))
    _, partial = live._checkpointed_tiled_worker(tmp_path, run, initialize=True)
    live._identify_origami_worker.assert_called_once()
    assert partial['params']['_checkpoint_preview']['completed'] == 1


def test_pause_on_last_tile_finishes_normally(tmp_path):
    live, run, payload = run_fixture()
    run['context']['tile_indices'] = np.array([0])
    control_for(live, tmp_path, run)
    def classify(*args):
        live.pause_tiled_analysis()
        return 'origami_multi_picks', deepcopy(payload)
    live._identify_origami_worker = Mock(side_effect=classify)
    _, complete = live._checkpointed_tiled_worker(tmp_path, run, initialize=True)
    assert '_tiled_paused' not in complete['params']
    live._complete_tiled_pause_control(complete)
    assert live._tiled_control is None


def test_continue_waits_for_composite_and_worker_completion(tmp_path, monkeypatch):
    live, run, _ = run_fixture()
    control = control_for(live, tmp_path, run)
    control['state'] = 'paused'
    live._resume_tiled_checkpoint = Mock()
    import paint_analysis_gui as gui
    message = Mock()
    monkeypatch.setattr(gui.messagebox, 'showinfo', message)
    live.active_worker_count = 1
    live._refresh_tiled_pause_controls()
    live.origami_continue_tiles_button.state.assert_called_with(['disabled'])
    live.continue_tiled_analysis()
    live.active_worker_count = 0
    live.origami_composite_window = Mock(busy=True)
    live.continue_tiled_analysis()
    assert message.call_count == 2
    live._resume_tiled_checkpoint.assert_not_called()


def test_worker_keeps_probability_matrices_disk_backed(tmp_path):
    live, run, payload = run_fixture()
    key = 'classification_template_probability_vectors'
    for item in payload['templates']:
        item['params'][key] = tuple((.25, .75) for _ in item['picks'].regions)
    live._identify_origami_worker = Mock(return_value=('origami_multi_picks', payload))
    _, combined = live._checkpointed_tiled_worker(tmp_path, run, initialize=True)
    a, b = [item['params'][key] for item in combined['templates']]
    assert isinstance(a, np.memmap)
    assert a is b
    np.testing.assert_array_equal(a, np.tile([.25, .75], (len(a), 1)))
