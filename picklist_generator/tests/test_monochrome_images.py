"""A grayscale tile needs one physical design, with all bits available for position."""
from threading import Event
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from image_encoding import (CHANNELS, FIT_MODES, barcode_bits, bin_image, compile_designs,
                            encoding_scheme, geometry, reconstruct, schemes_for_bits)
from color_light import grayscale_pixels
from alignment_reconstruction import AlignmentModel, AlignmentSettings
from gaussian_optimization import fitted_target, optimize_designs
from gaussian_reconstruction import render_gaussian_overlay
from rgb_simulation import simulate_alignment
from test_image_encoding import prepare


@pytest.mark.parametrize('bits,capacity', [(2, 4), (6, 64), (7, 128), (8, 256), (9, 384), (48, 384)])
def test_capacity_and_unique_codes(bits, capacity):
    choices = schemes_for_bits(bits, monochrome=True)
    assert max(s.tile_count for s in choices.values()) == capacity
    assert all(s.monochrome and s.design_count == s.tile_count for s in choices.values())
    assert not any(s.monochrome for s in schemes_for_bits(bits).values())
    key = f'mono_{bits}_tiles_{capacity}x1'
    codes = [tuple(barcode_bits('R1 R', x, 0, key)) for x in range(capacity)]
    assert len(set(codes)) == capacity
    assert all(len(code) == bits for code in codes)
    with pytest.raises(ValueError):
        barcode_bits('R1 G', 0, 0, key)


def test_three_times_tiles_same_48_barcodes_and_real_picklists():
    rgb = encoding_scheme('tiles_4x4')
    key = 'mono_6_tiles_8x6'
    mono = encoding_scheme(key)
    assert mono.design_count == rgb.design_count == 48
    assert mono.tile_count == 3 * rgb.tile_count
    info = geometry(key)
    pixels = [[[255]*3 for _ in range(info['columns'])] for _ in range(info['rows'])]
    designs = compile_designs(pixels, 25, {'R1 R': 1}, key)
    assert len(designs) == 48
    assert {d['color'] for d in designs} == {'R1 R'}
    assert {d['channel'] for d in designs} == {0}
    assert len({d['barcode_code'] for d in designs}) == 48
    assert reconstruct(designs) == pixels
    runs = prepare(designs)
    assert len(runs) == 48
    assert len({well for run in runs for well in run['destination_wells']}) == 48
    from barcode_registry import from_rgb
    records = [from_rgb(run, 'monochrome') for run in runs]
    assert len({r['bits'] for r in records}) == 48
    assert all(run['image_design']['image_mode'] == 'grayscale' for run in runs)


@pytest.mark.parametrize('linear', [False, True])
@pytest.mark.parametrize('rotation', [0, 90])
def test_color_input_becomes_grayscale_and_only_one_channel(tmp_path, linear, rotation):
    path = tmp_path / 'color.png'
    Image.new('RGB', (12, 8), (255, 0, 0)).save(path)
    key = 'mono_6_tiles_2x1'
    pixels = bin_image(path, FIT_MODES[2], rotation, key, linear_light=linear)
    expected = grayscale_pixels([[[255, 0, 0]]], linear)[0][0]
    assert expected == [127 if linear else 54] * 3
    assert all(pixel == expected for row in pixels for pixel in row)
    designs = compile_designs(pixels, 100, {'R1 R': 1}, key, rotation, linear_light=linear)
    assert len(designs) == 2
    assert all(site['rgb'] == expected for d in designs for site in d['image_sites'])
    for row in reconstruct(designs):
        assert all(pixel[0] == pixel[1] == pixel[2] for pixel in row)


def test_monochrome_rendering_optimization_and_background_job(tmp_path):
    key = 'mono_6_tiles_1x1'
    designs = compile_designs([[[128]*3 for _ in range(6)] for _ in range(8)], 100, {'R1 R': 1}, key)
    model = AlignmentModel(designs, AlignmentSettings(origami_per_design=2))
    assert model.weights(designs).shape[1] == 1
    for rendered in (model.expected(designs), model.example(designs), render_gaussian_overlay(designs)):
        np.testing.assert_array_equal(rendered.density[..., 0], rendered.density[..., 1])
        np.testing.assert_array_equal(rendered.density[..., 0], rendered.density[..., 2])
        assert rendered.metadata['channel_count'] == 1
    path = tmp_path / 'target.png'
    Image.new('RGB', (6, 8), (100, 160, 80)).save(path)
    rendered = render_gaussian_overlay(designs)
    target = fitted_target(path, FIT_MODES[2], rendered.rgb.size, monochrome=True)
    optimized, _ = optimize_designs(designs, target)
    assert len(optimized) == 1
    result = simulate_alignment(designs, model.settings, 7.5, 1.,
        dict(linear_light=False, normalize_designs=False, compensate_area=False),
        str(path), FIT_MODES[2], True, .1, 42, {}, Event(), lambda *args: None)
    assert len(result['designs']) == 1
    assert result['metrics']['validation_used_for_selection'] is False
    assert result['example'].metadata['image_mode'] == 'grayscale'


def test_mode_control_capacity_and_saved_restore(tmp_path, monkeypatch):
    import json
    import picklist_app
    for name in ('CONFIG_PATH', 'PLATE_STATE_PATH', 'STORAGE_STATE_PATH'):
        monkeypatch.setattr(picklist_app, name, tmp_path / (name + '.json'))
    picklist_app.CONFIG_PATH.write_text(json.dumps({'image_encoding_scheme': 'mono_8_tiles_4x4'}))
    app = picklist_app.PicklistApp()
    app.withdraw()
    try:
        view = app.rgb_image_view
        assert view.image_mode.get() == 'Black & white'
        assert view.concentration_labels['R1 R']['text'] == 'Grayscale concentration (nM)'
        assert view.concentration_groups['R1 R'].winfo_manager() == 'pack'
        assert view.concentration_groups['R1 G'].winfo_manager() == ''
        assert view.concentration_groups['R1 B'].winfo_manager() == ''
        assert view.site_tree.heading('RGB', 'text') == 'Gray level'
        assert view.binned_frame['text'] == 'Binned grayscale'
        assert view.run_setting_labels['replacement_path']['text'] == 'Grayscale source CSV'
        assert set(view._concentration_options(view._scheme_id())['plate_defaults_nm']) == {'R1 R'}
        assert view.image_geometry['design_count'] == 16
        assert '256' in view.tile_count_picker['values']
        assert all(encoding_scheme(key).monochrome for key in view._display_scheme_ids)
        view.tile_count.set('48')
        view._tile_count_changed()
        assert view.image_geometry['design_count'] == 48
        view.image_mode.set('RGB')
        view._image_mode_changed()
        assert not encoding_scheme(view._scheme_id()).monochrome
        assert all(group.winfo_manager() == 'pack' for group in view.concentration_groups.values())
        assert view.concentration_labels['R1 R']['text'] == 'R1 R site target (nM)'
        assert view.site_tree.heading('RGB', 'text') == 'RGB'
        assert view.binned_frame['text'] == 'Binned RGB'
        assert view.image_geometry['design_count'] == 144
        view.image_mode.set('Black & white')
        view._image_mode_changed()
        view.barcode_bit_count.set('9')
        view._bit_count_changed()
        assert '384' in view.tile_count_picker['values']
        assert encoding_scheme(view._scheme_id()).monochrome
        assert all('shift' not in label.lower() for label in view.scheme_picker['values'])
        view.tile_count.set('1')
        view._tile_count_changed()
        view.optimize_enabled.set(False)
        path = tmp_path / 'image.png'
        Image.new('RGB', (6, 8), 'white').save(path)
        with patch('rgb_image_view.messagebox.showerror') as error:
            view.image_path.set(str(path))
            view._load()
            assert len(view.designs) == 1
            assert view.design_name.get().startswith('Grayscale')
            assert 'RGB' not in view.prediction_frame['text']
            fills = [view.design_canvas.itemcget(item, 'fill') for item in view.design_canvas.find_all()
                     if any(tag.startswith('site_') for tag in view.design_canvas.gettags(item))]
            assert fills and all(fill[1:3] == fill[3:5] == fill[5:7] for fill in fills)
            rows = [view.site_tree.item(item, 'values') for item in view.site_tree.get_children()]
            assert rows and all(',' not in str(row[2]) and 0 <= int(row[2]) <= 255 for row in rows)
            assert any(int(row[2]) == 255 for row in rows)
            assert view.alignment_example.metadata['image_mode'] == 'grayscale'
            view.auto_export_templates.set(False)
            view._prepare()
            assert len(view.runs) == 1
            with patch('rgb_image_view.filedialog.askdirectory', return_value=str(tmp_path / 'exports')):
                view._export()
            manifest_path = next((tmp_path / 'exports').glob('*/image_manifest.json'))
            manifest = json.loads(manifest_path.read_text())
            assert manifest['geometry']['image_mode'] == 'grayscale'
            assert len(manifest['designs']) == 1
            image_path = manifest_path.parent / 'quantized_rgb.png'
            with Image.open(image_path) as saved:
                rgb = np.asarray(saved.convert('RGB'))
                np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
            error.assert_not_called()
    finally:
        app.destroy()


def test_extreme_resolution_fails_before_dense_allocation():
    key = 'mono_9_tiles_16x24'
    info = geometry(key)
    pixels = [[[0]*3 for _ in range(info['columns'])] for _ in range(info['rows'])]
    designs = compile_designs(pixels, 25, {'R1 R': 1}, key)
    assert len(designs) == 384
    with pytest.raises(ValueError, match='two million pixels'):
        AlignmentModel(designs, AlignmentSettings(), fwhm_nm=.1)
