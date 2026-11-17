from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile, ZIP_STORED
import json

import numpy as np
import pytest

from composite_session import REGION_KEYS, save_reconstruction_session, load_reconstruction_session
from origami_composite import reconstruct_composite, saved_rgb_batch_mapping, CompositeSettings
from test_origami_composite import settings, payload, footprint, batch_fixture
from test_composite_export_progress import window_method


def inputs():
    manifest, grid = batch_fixture()
    geometry, book, mappings, _ = saved_rgb_batch_mapping(manifest, ['red'], grid, {})
    config = dict(format='paint-composite-mappings-v1', settings=asdict(settings()),
                  mappings=mappings, codebook=book, geometry=geometry, rgb_batch_manifest=manifest)
    results = dict(red=payload([footprint((-50, 0)), footprint(), footprint((1, 0))], [False, True, True]))
    # This payload must not survive into a reconstruction-only archive.
    results['red']['classification_intermediates'] = object()
    results['red']['params']['candidate_crops'] = object()
    source = np.concatenate(([[4.5, 3.5], [4.5, 3.5], [6, 2], [0, 0]], np.full((10000, 2), 10000))).astype(np.float32)
    exclusions = dict(red=np.array([[6, 2]]))
    return source, results, config, exclusions, dict(red=grid)


def build_loaded(owner, cfg=None):
    config = cfg or owner.origami_composite_configuration
    results = owner.origami_multi_template_results
    exclusions = {name: result['params']['_reconstruction_exclusions_nm'] for name, result in results.items()}
    return reconstruct_composite(owner.origami_source_points_nm, results, config['mappings'],
                                 CompositeSettings(**config['settings']), exclusions)


def test_fixed_region_roundtrip_preserves_pixels_manifest_and_candidate_ids(tmp_path):
    args = inputs()
    source, results, config, exclusions, grids = args
    original = reconstruct_composite(source, results, config['mappings'], settings(), exclusions)
    messages = []
    path = tmp_path / 'reconstruction.paintrecon'
    save_reconstruction_session(path, *args, source_path='original.csv', unclassified_count=17, progress=messages.append)
    owner = load_reconstruction_session(path, messages.append)
    assert isinstance(owner.origami_source_points_nm, np.memmap)
    assert not owner.origami_source_points_nm.flags.writeable
    assert len(owner.origami_source_points_nm) == 3  # duplicates retained, distant acquisition discarded
    assert path.stat().st_size < source.nbytes / 4
    assert owner.origami_composite_configuration == config
    assert owner.origami_loaded_source_path == 'original.csv'
    assert owner.origami_multi_template_unclassified_count == 17
    assert owner.reconstruction_region == {key: config['settings'][key] for key in REGION_KEYS}
    restored = build_loaded(owner)
    for key in ('rgb', 'density', 'coverage'):
        np.testing.assert_array_equal(restored[key], original[key])
    assert restored['report'] == original['report']
    assert [row['candidate_index'] for row in restored['report']['candidates']] == [1, 2]
    assert 'candidate_crops' not in owner.origami_multi_template_results['red']['params']
    assert messages[-1].startswith('Reconstruction session loaded')
    assert any('100%' in message for message in messages)
    # Re-saving compact poses preserves original candidate identities, too.
    second = tmp_path / 'second.paintrecon'
    save_reconstruction_session(second, owner.origami_source_points_nm, owner.origami_multi_template_results,
                                config, exclusions, grids)
    assert build_loaded(load_reconstruction_session(second))['report'] == original['report']


def test_session_keeps_excluded_species_and_supports_smoothing_brightness_changes(tmp_path):
    source, results, config, exclusions, grids = inputs()
    results['control'] = payload([footprint((20, 0))])
    config['mappings']['control'] = dict(config['mappings']['red'], include=False, code='')
    grids['control'] = grids['red']
    source = np.concatenate((source, [[24.5, 3.5]]))
    path = tmp_path / 'all_species.paintrecon'
    save_reconstruction_session(path, source, results, config, exclusions, grids)
    owner = load_reconstruction_session(path)
    config['mappings']['control']['include'] = True
    config['mappings']['red']['brightness'] = .4
    config['settings'].update(sigma_nm=1, global_sigma_nm=2, rotation_deg=90)
    original = reconstruct_composite(source, results, config['mappings'], CompositeSettings(**config['settings']), exclusions)
    restored = build_loaded(owner, config)
    np.testing.assert_array_equal(restored['density'], original['density'])
    assert restored['report'] == original['report']


def test_save_is_atomic_and_requires_embedded_manifest(tmp_path):
    source, results, config, exclusions, grids = inputs()
    path = tmp_path / 'existing.paintrecon'
    path.write_bytes(b'keep existing file')
    source[-1] = np.nan
    with pytest.raises(ValueError, match='finite'):
        save_reconstruction_session(path, source, results, config, exclusions, grids)
    assert path.read_bytes() == b'keep existing file'
    assert len(list(tmp_path.iterdir())) == 1
    config['rgb_batch_manifest'] = None
    with pytest.raises(ValueError, match='RGB picklist batch'):
        save_reconstruction_session(path, source, results, config, exclusions, grids)


def test_empty_region_retains_dark_origami(tmp_path):
    _, results, config, exclusions, grids = inputs()
    path = tmp_path / 'dark.paintrecon'
    save_reconstruction_session(path, np.array([[10000., 10000.]]), results, config, exclusions, grids)
    owner = load_reconstruction_session(path)
    assert owner.origami_source_points_nm.shape == (0, 2)
    result = build_loaded(owner)
    assert not result['density'].any()
    assert result['report']['species'][0]['accepted_origami'] == 2


@pytest.mark.parametrize('field,value', [('version', 200), ('extraction_region', {})])
def test_rejects_incompatible_or_inconsistent_archive(tmp_path, field, value):
    path = tmp_path / 'good.paintrecon'
    save_reconstruction_session(path, *inputs())
    bad = tmp_path / 'bad.paintrecon'
    with ZipFile(path) as archive, ZipFile(bad, 'w', compression=ZIP_STORED) as output:
        for info in archive.infolist():
            data = archive.read(info.filename)
            if info.filename == 'reconstruction.json':
                doc = json.loads(data)
                doc[field] = value
                data = json.dumps(doc).encode()
            output.writestr(info.filename, data)
    with pytest.raises(ValueError):
        load_reconstruction_session(bad)


def test_open_session_cannot_be_overwritten(tmp_path):
    path = tmp_path / 'open.paintrecon'
    source, results, config, exclusions, grids = inputs()
    save_reconstruction_session(path, source, results, config, exclusions, grids)
    owner = load_reconstruction_session(path)
    with pytest.raises(ValueError, match='new filename'):
        save_reconstruction_session(path, owner.origami_source_points_nm, results, config, exclusions, grids)


class Variable:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


def test_gui_restores_saved_mappings_and_locks_region(tmp_path):
    path = tmp_path / 'gui.paintrecon'
    save_reconstruction_session(path, *inputs())
    owner = load_reconstruction_session(path)
    config = owner.origami_composite_configuration
    from binary_occupancy import MODES
    restored_occupancy = []
    window = SimpleNamespace(reconstruction_owner=owner, reconstruction_mode=Variable(MODES[2]),
                             apply_occupancy_settings=restored_occupancy.append,
                             variables={key: Variable(0) for key in asdict(settings())
                                        if key not in ('average_species', 'mirror_x', 'mirror_y', 'normalize_overlap')},
                             average=Variable(False), normalize_overlap=Variable(False), mirror_x=Variable(False), mirror_y=Variable(False),
                             rows={'red': {key: Variable(None) for key in config['mappings']['red']}},
                             refresh_templates=lambda: None, status=Variable(''))
    states = []
    window.region_widgets = {key: SimpleNamespace(configure=lambda **kwargs: states.append(kwargs['state'])) for key in REGION_KEYS}
    apply = window_method('apply_configuration', dict(CompositeSettings=CompositeSettings, REGION_KEYS=REGION_KEYS, MODES=MODES))
    apply(window, config)
    assert window.reconstruction_mode.get() == MODES[0]
    assert restored_occupancy == [{}]
    assert window.batch_manifest == config['rgb_batch_manifest']
    assert {key: var.get() for key, var in window.rows['red'].items()} == config['mappings']['red']
    assert states == ['readonly'] * len(REGION_KEYS)
    read = window_method('settings', dict(CompositeSettings=CompositeSettings, REGION_KEYS=REGION_KEYS))
    assert read(window) == settings()
    window.variables['sigma_nm'].set(3)
    assert read(window).sigma_nm == 3
    window.variables['width_nm'].set(100)
    with pytest.raises(ValueError, match='fixed image region'):
        read(window)


def test_gui_open_session_applies_mappings_without_replacing_full_analysis(tmp_path):
    import queue
    path = tmp_path / 'isolated.paintrecon'
    save_reconstruction_session(path, *inputs())
    loaded = load_reconstruction_session(path)
    full_analysis = SimpleNamespace(origami_multi_template_results={'original': object()})
    original_results = full_analysis.origami_multi_template_results
    window = SimpleNamespace(owner=full_analysis, reconstruction_owner=None, busy=True,
                             result=object(), result_inputs=object(), events=queue.Queue(),
                             status=Variable(''), after=lambda *args: 'timer', poll=lambda: None,
                             update_idletasks=lambda: None, finish_session_dialog=lambda: None)
    applied = []
    window.apply_configuration = applied.append
    window.events.put(('session_loaded', loaded))
    poll = window_method('poll', dict(queue=queue, record_diagnostic=lambda *args, **kwargs: None))
    poll(window)
    assert full_analysis.origami_multi_template_results is original_results
    assert window.reconstruction_owner is loaded
    assert applied == [loaded.origami_composite_configuration]
    assert not window.busy
    assert window.result is None and window.result_inputs is None
    assert 'RGB batch mappings applied' in window.status.get()


def test_uniform_fill_survives_reconstruction_session_roundtrip(tmp_path):
    source, results, config, exclusions, grids = inputs()
    results['green'] = payload([footprint((40, 0))], [False])
    config['mappings']['green'] = dict(color='Green', code='', x_shift_nm=0, y_shift_nm=40.8,
                                      include=False, fill_percent='50')
    grids['green'] = grids['red']
    path = tmp_path / 'fill.paintrecon'
    save_reconstruction_session(path, source, results, config, exclusions, grids)
    owner = load_reconstruction_session(path)
    assert owner.origami_composite_configuration['mappings']['green']['fill_percent'] == '50'
    actual = build_loaded(owner)
    expected = reconstruct_composite(source, results, config['mappings'], settings(), exclusions)
    np.testing.assert_array_equal(actual['rgb'], expected['rgb'])
    assert actual['report']['synthetic_replacements'] == expected['report']['synthetic_replacements']


@pytest.mark.parametrize('normalize', [False, True])
def test_blend_session_retains_extra_source_and_roundtrips(tmp_path, normalize):
    source, results, config, exclusions, grids = inputs()
    cfg = replace(settings(), blend_width_nm=10, blend_height_nm=12, normalize_overlap=normalize)
    config['settings'] = asdict(cfg)
    source = np.concatenate((source, [[3, 6], [9, -2]]))
    original = reconstruct_composite(source, results, config['mappings'], cfg, exclusions)
    path = tmp_path / 'blend.paintrecon'
    save_reconstruction_session(path, source, results, config, exclusions, grids)
    owner = load_reconstruction_session(path)
    assert owner.reconstruction_region['blend_height_nm'] == 12
    assert any(np.array_equal(p, [3, 6]) for p in owner.origami_source_points_nm)
    restored = build_loaded(owner)
    np.testing.assert_array_equal(restored['density'], original['density'])
    assert restored['report'] == original['report']


def test_legacy_session_locks_blend_to_original_region(tmp_path):
    path = tmp_path / 'legacy.paintrecon'
    save_reconstruction_session(path, *inputs())
    with ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    document = json.loads(files['reconstruction.json'])
    for key in ('blend_width_nm', 'blend_height_nm'):
        document['configuration']['settings'].pop(key)
        document['extraction_region'].pop(key)
    files['reconstruction.json'] = json.dumps(document).encode()
    with ZipFile(path, 'w', compression=ZIP_STORED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    owner = load_reconstruction_session(path)
    assert owner.reconstruction_region['blend_width_nm'] == 0
    assert owner.reconstruction_region['blend_height_nm'] == 0
    assert build_loaded(owner)['density'].size
