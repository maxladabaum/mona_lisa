from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd
import pytest

from analysis_session import save_analysis_session
from reconstruction_analysis import ReconstructionArchive, load_reconstruction_from_analysis
from origami_composite import reconstruct_composite
from test_origami_composite import footprint, settings, mapping
from test_composite_export_progress import window_method


@dataclass
class SavedPicks:
    accepted_mask: np.ndarray
    rectangle_corners_nm: np.ndarray
    regions: np.ndarray


def picks(center=(0, 0)):
    return SavedPicks(np.array([False, True]), np.array([footprint((-20, -20)), footprint(center)]),
                      np.zeros((300, 300)))


def params():
    return dict(rows=8, columns=12, spacing_x_nm=9.9, spacing_y_nm=5.1,
                column_offsets_nm=tuple([0] * 6 + [10] * 6),
                alignment_template_overlay_points_nm=((0, 0),),
                logical_model=dict(bit_physical_cells=((1, 2),), classification_matrix=np.zeros((300, 300))),
                digital_pixel_probabilities=np.ones((300, 300)), mirror_template_y=False)


def config():
    return dict(format='paint-composite-mappings-v1', settings=asdict(settings()), mappings={'red': mapping()},
                codebook={}, geometry={}, rgb_batch_manifest={'designs': [{'name': 'red'}]})


def save_full(path, with_config=True):
    # Exercise disk caching for genuinely large coordinates; tiny pose arrays
    # must stay in memory instead of creating hundreds of tiny cache files.
    source = np.concatenate(([[4.5, 3.5], [4.5, 3.5]], np.full((100000, 2), 10000.)))
    state = dict(origami_source_points_nm=source,
                 origami_multi_template_results={'red': dict(picks=picks(), params=params())},
                 origami_multi_template_unclassified_count=9, origami_loaded_source_path=Path('/absent/source.csv'),
                 loaded=pd.DataFrame({'x': np.ones(5000), 'unneeded': np.ones(5000)}),
                 origami_multi_template_overlay_results=np.ones((300, 300)))
    if with_config:
        state['origami_composite_configuration'] = config()
    save_analysis_session(path, dict(state=state, settings={}, diagnostics={'large': np.ones((300, 300))}))
    return source, state


def remove_unused_arrays(path):
    """Deleting ignored payloads proves the loader never even opens them."""
    with ReconstructionArchive(path) as reader:
        state = reader.fields(reader.fields(reader.root)['state'])
        result = reader.fields(reader.fields(state['origami_multi_template_results'])['red'])
        pick_fields = reader.fields(result['picks'])
        param_fields = reader.fields(result['params'])
        logical_fields = reader.fields(param_fields['logical_model'])
        unused = {reader.node(ref)['file'] for ref in (
            state['origami_multi_template_overlay_results'], pick_fields['regions'],
            param_fields['digital_pixel_probabilities'], logical_fields['classification_matrix'])}
        unused.update(reader.node(ref)['file'] for ref in reader.node(state['loaded'])['values'])
    replacement = path.with_suffix('.new')
    with ZipFile(path) as original, ZipFile(replacement, 'w') as out:
        for info in original.infolist():
            if info.filename not in unused:
                out.writestr(info.filename, original.read(info.filename))
    replacement.replace(path)


@pytest.mark.parametrize('with_config', [False, True])
def test_loads_only_reconstruction_fields_and_preserves_results(tmp_path, with_config):
    path = tmp_path / 'saved.paintanalysis'
    source, state = save_full(path, with_config)
    remove_unused_arrays(path)
    messages = []
    owner = load_reconstruction_from_analysis(path, messages.append)
    assert isinstance(owner.origami_source_points_nm, np.memmap)
    assert owner.origami_composite_configuration == (config() if with_config else None)
    assert owner.reconstruction_region is None
    assert owner.origami_multi_template_unclassified_count == 9
    assert owner.origami_loaded_source_path == '/absent/source.csv'
    restored = owner.origami_multi_template_results
    np.testing.assert_array_equal(restored['red']['picks'].candidate_indices, [1])
    assert 'digital_pixel_probabilities' not in restored['red']['params']
    assert 'classification_matrix' not in restored['red']['params']['logical_model']
    expected = reconstruct_composite(source, state['origami_multi_template_results'], {'red': mapping()}, settings())
    actual = reconstruct_composite(owner.origami_source_points_nm, restored, {'red': mapping()}, settings())
    np.testing.assert_array_equal(actual['density'], expected['density'])
    assert actual['report'] == expected['report']
    assert any('skipping localization tables' in message for message in messages)
    # Subsequent opens reuse source NPY data without rewriting it.
    cached = Path(owner.origami_source_points_nm.filename)
    before = cached.stat().st_mtime_ns
    load_reconstruction_from_analysis(path)
    assert cached.stat().st_mtime_ns == before


def save_run(directory, completed=(0, 1), filtered=False):
    directory.mkdir(exist_ok=True)
    lattice = [(0., 20., 0., 20.), (20., 40., 0., 20.)]
    source = pd.DataFrame(dict(x=[2.25, 12.25, 100.], y=[1.75, 1.75, 100.],
                               photons=[5., 20., 99.], junk=[999., 999., 999.]))
    run = dict(checkpoint_version=1, run_id='run-1',
               analysis=dict(state={'origami_source_points_nm': np.array([[-999., -999.]]),
                                    'origami_loaded_source_path': Path('/source.csv')}, settings={}),
               context=dict(tile_lattice=lattice, tile_indices=np.array([0, 1]), pixelsize=2., source_locs=source,
                            source_params=dict(source='Filtered' if filtered else 'Raw',
                                               active_filters=[('photons', (10, 30))] if filtered else [],
                                               exposure_ms=100, link_radius_nm=10, max_gap_frames=1)))
    path = directory / 'run.paintanalysis'
    save_analysis_session(path, run)
    for index in completed:
        payload = dict(templates=[dict(name='red', picks=picks((20 * index, 0)), params=params())],
                       unclassified_count=2, unclassified_details=np.ones((300, 300)))
        save_analysis_session(directory / f'tile-{index:08d}.paintanalysis',
                              dict(run_id='run-1', index=index, mode='multi', bounds=lattice[index], payload=payload))
    return path


def test_tiled_run_uses_full_field_xy_and_accepted_tile_poses(tmp_path):
    path = save_run(tmp_path)
    owner = load_reconstruction_from_analysis(path)
    np.testing.assert_array_equal(owner.origami_source_points_nm, [[4.5, 3.5], [24.5, 3.5], [200, 200]])
    result = owner.origami_multi_template_results['red']
    np.testing.assert_array_equal(result['picks'].candidate_indices, [1, 3])
    assert owner.origami_multi_template_unclassified_count == 4
    assert result['params']['_checkpoint_preview'] == dict(completed=2, planned=2)
    image = reconstruct_composite(owner.origami_source_points_nm, owner.origami_multi_template_results,
                                  {'red': mapping()}, settings())
    assert image['density'].sum() == 1  # two counts averaged across two origami


def test_partial_run_excludes_uncompleted_tiles_and_marks_scope(tmp_path):
    owner = load_reconstruction_from_analysis(save_run(tmp_path, completed=(1,)))
    np.testing.assert_array_equal(owner.origami_source_points_nm, [[24.5, 3.5]])
    result = owner.origami_multi_template_results['red']
    assert result['params']['_checkpoint_preview'] == dict(completed=1, planned=2)
    np.testing.assert_array_equal(result['picks'].candidate_indices, [1])


def test_saved_source_filters_are_honored_without_reading_unused_columns(tmp_path):
    path = save_run(tmp_path, filtered=True)
    calls = []
    def evaluate(frame, mode, *args):
        calls.append(mode)
        assert set(frame.columns) == {'x', 'y', 'photons'}
        return frame[mode], mode
    with pytest.raises(ValueError, match='filter evaluator'):
        load_reconstruction_from_analysis(path)
    owner = load_reconstruction_from_analysis(path, filter_series=evaluate)
    assert calls == ['photons']
    np.testing.assert_array_equal(owner.origami_source_points_nm, [[24.5, 3.5]])
    # Cached coordinates already have the source filters applied.
    np.testing.assert_array_equal(load_reconstruction_from_analysis(path).origami_source_points_nm, [[24.5, 3.5]])


def test_rejects_foreign_tiles_empty_runs_and_individual_tiles(tmp_path):
    path = save_run(tmp_path, completed=())
    with pytest.raises(ValueError, match='No completed'):
        load_reconstruction_from_analysis(path)
    tile = tmp_path / 'tile-00000000.paintanalysis'
    save_analysis_session(tile, dict(run_id='different-run', index=0, bounds=(0, 20, 0, 20), mode='multi', payload={}))
    with pytest.raises(ValueError, match='does not match'):
        load_reconstruction_from_analysis(path)
    with pytest.raises(ValueError, match='not an individual tile'):
        load_reconstruction_from_analysis(tile)


def test_gui_analysis_inputs_leave_region_editable_and_do_not_inherit_old_manifest(tmp_path):
    from test_composite_session import Variable
    from origami_composite import CompositeSettings
    path = tmp_path / 'old.paintanalysis'
    save_full(path, with_config=False)
    owner = load_reconstruction_from_analysis(path)
    window = SimpleNamespace(result='old', result_inputs='old', reconstruction_owner=None,
                             rows={'unrelated': 'old'}, codebook={'old': 'old'}, batch_manifest={'old': 'old'},
                             variables={key: Variable(0) for key in asdict(settings())
                                        if key not in ('average_species', 'mirror_x', 'mirror_y', 'normalize_overlap')},
                             average=Variable(False), normalize_overlap=Variable(False), mirror_x=Variable(False), mirror_y=Variable(False),
                             region_widgets={'width': SimpleNamespace(configure=lambda **kwargs: None)},
                             figure=SimpleNamespace(clear=lambda: None), canvas=SimpleNamespace(draw_idle=lambda: None),
                             refresh_templates=lambda: None, first_params=lambda: params(),
                             full_grid=lambda params: np.array([[0, 0], [100, 40]]), status=Variable(''))
    install = window_method('install_analysis_reconstruction', dict(CompositeSettings=CompositeSettings))
    install(window, owner)
    assert window.batch_manifest is None and window.codebook == {} and window.rows == {}
    assert window.reconstruction_owner is owner
    assert window.variables['width_nm'].get() > 0
    assert 'Load the RGB picklist batch' in window.status.get()


def test_checkpoint_coordinate_cache_reused_and_extended_for_new_tiles(tmp_path):
    path = save_run(tmp_path, completed=(0,))
    owner = load_reconstruction_from_analysis(path)
    cached = Path(owner.origami_source_points_nm.filename)
    timestamp = cached.stat().st_mtime_ns
    messages = []
    reopened = load_reconstruction_from_analysis(path, messages.append)
    assert Path(reopened.origami_source_points_nm.filename) == cached
    assert cached.stat().st_mtime_ns == timestamp
    assert any('Reusing cached source coordinates' in message for message in messages)
    save_analysis_session(tmp_path / 'tile-00000001.paintanalysis',
                          dict(run_id='run-1', index=1, mode='multi', bounds=(20., 40., 0., 20.),
                               payload=dict(templates=[dict(name='red', picks=picks((20, 0)), params=params())],
                                            unclassified_count=2)))
    expanded = load_reconstruction_from_analysis(path)
    assert Path(expanded.origami_source_points_nm.filename) != cached
    assert len(expanded.origami_source_points_nm) == 3


def test_filtered_run_with_no_remaining_signal_is_valid(tmp_path):
    path = save_run(tmp_path, filtered=True)
    owner = load_reconstruction_from_analysis(path, filter_series=lambda frame, *args: (frame.photons * 0, 'zero'))
    assert owner.origami_source_points_nm.shape == (0, 2)
    result = reconstruct_composite(owner.origami_source_points_nm, owner.origami_multi_template_results,
                                   {'red': mapping()}, settings())
    assert result['density'].sum() == 0


def test_main_button_opens_reconstruction_without_full_analysis_loader():
    import ast
    module_path = Path(__file__).parents[1] / 'paint_analysis_gui.py'
    tree = ast.parse(module_path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PaintAnalysisApp')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'load_reconstruction_from_analysis')
    calls = []
    window = SimpleNamespace(busy=False, winfo_exists=lambda: True,
                             load_analysis_reconstruction=lambda *args: calls.append(args))
    app = SimpleNamespace(origami_composite_window=window,
                          _file_dialog_initial_dir=lambda: '/tmp', _remember_file_dialog_dir=lambda path: None,
                          _open_origami_composite=lambda: None)
    namespace = dict(Path=Path, localization_series_for_mode='filter evaluator',
                     filedialog=SimpleNamespace(askopenfilename=lambda **kwargs: '/tmp/run.paintanalysis'))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(module_path), 'exec'), namespace)
    namespace[method.name](app)
    assert calls == [('/tmp/run.paintanalysis', 'filter evaluator')]
    assert 'text="Load reconstruction from analysis"' in module_path.read_text()


def test_small_tile_arrays_stay_in_memory_and_preloaded_archive_closes(tmp_path):
    save_run(tmp_path)
    tile = tmp_path / 'tile-00000000.paintanalysis'
    messages = []
    with ReconstructionArchive(tile, messages.append, preload=True) as reader:
        buffer = reader._buffer
        assert buffer is not None
        payload = reader.fields(reader.fields(reader.root)['payload'])
        template = reader.fields(reader.items(payload['templates'])[0])
        fields = reader.fields(template['picks'])
        mask = reader.decode(fields['accepted_mask'])
        assert not isinstance(mask, np.memmap)
        assert reader.decode(fields['accepted_mask']) is mask
        restored, count = reader.picks(template['picks'])
        assert count == 2
        assert restored.candidate_indices.tolist() == [1]
        reader.params(template['params'])
        assert not reader.directory.exists()  # zero small-array cache writes
    assert buffer.closed
    assert any('Reading tile sequentially' in message for message in messages)
    assert any('Decoding reconstruction metadata' in message for message in messages)


def test_tiled_progress_distinguishes_metadata_from_pose_reads(tmp_path):
    path = save_run(tmp_path)
    messages = []
    load_reconstruction_from_analysis(path, messages.append)
    assert any('Reading accepted poses: tile-00000000' in message for message in messages)
    assert any('Reading accepted poses: tile-00000001' in message for message in messages)
