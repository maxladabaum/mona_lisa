"""Redundant copies share image positions but use independent physical barcodes."""
import numpy as np
import pytest

from image_encoding import barcode_bits, compile_designs, encoding_scheme, geometry, reconstruct, schemes_for_bits
from gaussian_reconstruction import render_gaussian_overlay
from alignment_reconstruction import AlignmentModel, AlignmentSettings
from test_image_encoding import prepare


def designs(key):
    info = geometry(key)
    pixels = [[[128] * 3 for _ in range(info['columns'])] for _ in range(info['rows'])]
    return compile_designs(pixels, 100, {'R1 R': 1}, key)


def test_redundant_barcodes_recipes_and_geometry():
    key = 'mono_6_tiles_2x1_r3'
    scheme = encoding_scheme(key)
    assert scheme.design_count == 6
    copies = designs(key)
    assert len(copies) == len({d['name'] for d in copies}) == len({d['barcode_code'] for d in copies}) == 6
    assert geometry(key)['redundancy'] == 3
    assert {c['code'] for c in geometry(key)['codebook']} == {d['barcode_code'] for d in copies}
    for x in range(2):
        group = [d for d in copies if d['x_shift'] == x]
        assert all(d['image_sites'] == group[0]['image_sites'] for d in group)
        for d in group:
            assert d['bits'] == barcode_bits('R1 R', x, 0, key, d['replica_index'])
            for bit, sites in zip(d['bits'], d['barcode_sites']):
                for row, col in sites:
                    assert d['site_fractions'][f'U-Apt_{row}-{col}_R1'] == bit
    runs = prepare(copies)
    assert len(runs) == 6
    assert len({w for run in runs for w in run['destination_wells']}) == sum(len(run['destination_wells']) for run in runs)
    assert reconstruct(copies) == reconstruct(designs('mono_6_tiles_2x1'))
    for d in copies[:2]:
        for site in d['image_sites']:
            site['replacement_fraction'] = 0
    assert reconstruct(copies)[0][0] == [85]*3


@pytest.mark.parametrize('normalize', [False, True])
def test_redundancy_preserves_expected_brightness(normalize):
    original = designs('mono_6_tiles_1x1')
    copies = designs('mono_6_tiles_1x1_r3')
    a = render_gaussian_overlay(original, normalize_designs=normalize)
    b = render_gaussian_overlay(copies, normalize_designs=normalize)
    np.testing.assert_allclose(a.density * a.metadata['linear_rgb_gain'],
                               b.density * b.metadata['linear_rgb_gain'], atol=1e-10)
    settings = AlignmentSettings(origami_per_design=2)
    a = AlignmentModel(original, settings, normalize_designs=normalize).expected(original)
    b = AlignmentModel(copies, settings, normalize_designs=normalize).expected(copies)
    np.testing.assert_allclose(a.density * a.metadata['linear_rgb_gain'],
                               b.density * b.metadata['linear_rgb_gain'], atol=1e-10)


def test_redundancy_capacity_validation():
    assert max(s.tile_count for s in schemes_for_bits(6, True, 3).values()) == 21
    assert max(s.design_count for s in schemes_for_bits(9, True, 3).values()) == 384
    for value in [0, -1, 1.5, 65]:
        with pytest.raises(ValueError):
            schemes_for_bits(6, True, value)
    with pytest.raises(ValueError):
        encoding_scheme('mono_6_tiles_22x1_r3')
    with pytest.raises(ValueError):
        schemes_for_bits(6, False, 3)


def test_ui_redundancy_saved_and_capacity(tmp_path, monkeypatch):
    import json
    from unittest.mock import patch
    import picklist_app
    for name in ('CONFIG_PATH', 'PLATE_STATE_PATH', 'STORAGE_STATE_PATH'):
        monkeypatch.setattr(picklist_app, name, tmp_path / (name + '.json'))
    picklist_app.CONFIG_PATH.write_text(json.dumps({'image_encoding_scheme': 'mono_6_tiles_2x1_r3'}))
    app = picklist_app.PicklistApp()
    app.withdraw()
    try:
        view = app.rgb_image_view
        assert view.redundancy.get() == '3'
        assert view.image_geometry['design_count'] == 6
        assert '21' in view.tile_count_picker['values']
        assert '22' not in view.tile_count_picker['values']
        view.redundancy.set('4')
        view._redundancy_changed()
        assert view.image_geometry['design_count'] == 8
        assert app.config['image_encoding_scheme'] == 'mono_6_tiles_2x1_r4'
        view.redundancy.set('33')
        with patch('rgb_image_view.messagebox.showerror') as error:
            view._redundancy_changed()
            error.assert_called_once()
        assert view.redundancy.get() == '4'
        assert view.image_geometry['design_count'] == 8
        view.image_mode.set('RGB')
        view._image_mode_changed()
        assert view.image_geometry['design_count'] == 6
        assert str(view.redundancy_picker['state']) == 'disabled'
    finally:
        app.destroy()


@pytest.mark.parametrize('normalize', [False, True])
def test_gaussian_optimization_keeps_copies_and_barcodes(normalize):
    from gaussian_optimization import optimize_designs
    original = designs('mono_6_tiles_1x1')
    copies = designs('mono_6_tiles_1x1_r3')
    target = np.full(render_gaussian_overlay(original).density.shape, .3)
    a, _ = optimize_designs(original, target, normalize_designs=normalize)
    b, _ = optimize_designs(copies, target, normalize_designs=normalize)
    assert len(b) == 3
    assert [d['bits'] for d in b] == [d['bits'] for d in copies]
    assert reconstruct(a) == reconstruct(b)
