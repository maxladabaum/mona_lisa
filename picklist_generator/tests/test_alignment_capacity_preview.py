"""Oversized alignment layouts still provide an honest, usable Gaussian preview."""
import json
from threading import Event
from unittest.mock import patch

import numpy as np
import pytest

from alignment_reconstruction import AlignmentCapacityError, AlignmentSettings
from image_encoding import compile_designs, geometry
from rgb_simulation import simulate_alignment
from test_rgb_simulation import LIGHT, pump, view


@pytest.mark.parametrize('key', ['mono_6_tiles_8x8', 'mono_8_tiles_16x16', 'mono_9_tiles_16x24'])
def test_large_layouts_use_chunked_alignment_without_fallback(key):
    info = geometry(key)
    pixels = [[[255]*3 for _ in range(info['columns'])] for _ in range(info['rows'])]
    designs = compile_designs(pixels, 25, {'R1 R': 1}, key)
    result = simulate_alignment(designs, AlignmentSettings(), 7.5, 1., LIGHT,
        '/not-needed-without-optimization.png', 'Stretch', False, .1, 42, {}, Event(), lambda *args: None)
    assert result['designs'] is designs
    assert result['example'] is not None
    preview = result['reconstruction']
    assert preview.metadata['alignment_backend'] == 'chunked'
    assert preview.metadata['alignment_included'] is True
    assert 'alignment_fallback' not in preview.metadata
    assert preview.metadata['fwhm_nm'] == 7.5
    assert preview.metadata['alignment_mean_bytes'] < 512_000_000
    assert np.any(np.asarray(preview.rgb))


def test_fallback_ui_keeps_designs_and_exports_without_alignment(view, tmp_path):
    app, view = view
    view.optimize_enabled.set(True)
    view.alignment_enabled.set(False)
    view._volumes_changed()
    expected_designs = view.designs
    expected_density = view.reconstruction.density.copy()
    assert view.baseline_frame.winfo_manager() == 'pack'
    view.alignment_enabled.set(True)
    error = AlignmentCapacityError('Alignment raster is too large for this field.')
    with patch.object(view, '_alignment_model', side_effect=error), \
         patch('rgb_simulation.cached_alignment_model', side_effect=error):
        view._volumes_changed()
        assert view._simulation_cancel is not None
        assert view.baseline_frame.winfo_manager() == 'pack'
        pump(app, lambda: view._simulation_cancel is None)
        assert view.reconstruction is not None
        assert len(view.designs) == 3
        assert 'alignment unavailable' in view.prediction_frame['text']
        assert not view.pixel_canvases[1].find_withtag('preview_error')
        assert view.alignment_enabled.get()  # Preserve the requested setting.
        assert view.alignment_example is None
        assert view.optimization_metrics is not None
        assert view.optimization_metrics['alignment_included'] is False
        assert view.reconstruction.metadata['optimization_skipped'] is False
        assert 'Gaussian-only optimization' in view.optimization_summary.get()
        assert view.baseline_frame.winfo_manager() == 'pack'
        assert view.designs == expected_designs
        np.testing.assert_array_equal(view.reconstruction.density, expected_density)
        assert view.simulation_progress['value'] == 100
        assert 'alignment unavailable' in view.simulation_status.get()
        view._new_alignment_example()  # Does not retry an impossible model.
        view.auto_export_templates.set(False)
        with patch('rgb_image_view.messagebox.showerror') as errors:
            view._prepare()
            assert len(view.runs) == 3
            assert view.snapshot['alignment_simulation'] is None
            assert view.snapshot['alignment_example_seed'] is None
            assert view.snapshot['gaussian_reconstruction']['alignment_fallback']
            with patch('rgb_image_view.filedialog.askdirectory', return_value=str(tmp_path / 'exports')):
                view._export()
            errors.assert_not_called()
        manifest_path = next((tmp_path / 'exports').glob('*/image_manifest.json'))
        manifest = json.loads(manifest_path.read_text())
        assert manifest['alignment_simulation'] is None
        assert manifest['gaussian_reconstruction']['alignment_included'] is False
        assert manifest['gaussian_reconstruction']['optimization_skipped'] is False
        assert manifest['optimization']['alignment_included'] is False
        assert (manifest_path.parent / 'gaussian_reconstruction.png').exists()
        assert not (manifest_path.parent / 'alignment_example.png').exists()
        view.alignment_enabled.set(False)
        view._volumes_changed()
        assert view.simulation_status.get() == 'Alignment simulation off.'
        assert 'alignment unavailable' not in view.prediction_frame['text']
        assert view.baseline_frame.winfo_manager() == 'pack' 


def test_invalid_width_is_not_misreported_as_capacity_fallback():
    designs = compile_designs([[[255]*3 for _ in range(6)] for _ in range(8)],
                              25, {'R1 R': 1}, 'mono_6_tiles_1x1')
    with pytest.raises(ValueError, match='positive Gaussian'):
        simulate_alignment(designs, AlignmentSettings(), -1, 1., LIGHT,
            '/unused.png', 'Stretch', True, .1, 42, {}, Event(), lambda *args: None)


def test_252_tile_grayscale_runs_alignment_optimization(tmp_path):
    from PIL import Image
    from image_encoding import bin_image
    path = tmp_path / 'grayscale.png'
    ramp = np.tile(np.arange(0, 256, 4, dtype=np.uint8), (64, 1))
    Image.fromarray(ramp).save(path)
    key = 'mono_8_tiles_18x14'
    columns = ['05', '04', '03', '02', '01']
    pixels = bin_image(path, 'Stretch', scheme_id=key, active_image_columns=columns)
    designs = compile_designs(pixels, 100, {'R1 R': 1}, key, active_image_columns=columns)
    result = simulate_alignment(designs, AlignmentSettings(), 7.5, 1., LIGHT,
        str(path), 'Stretch', True, .05, 42, {}, Event(), lambda *args: None)
    assert len(result['designs']) == 252
    assert result['baseline'] is not None
    assert result['metrics']['optimized_rmse'] <= result['metrics']['baseline_rmse']
    assert 'alignment_fallback' not in result['reconstruction'].metadata
    assert result['reconstruction'].metadata['alignment_backend'] == 'chunked'
    assert result['reconstruction'].metadata['alignment_included'] is True
    assert result['metrics']['validation_used_for_selection'] is False
    assert result['reconstruction'].metadata['fwhm_nm'] == 7.5
    assert result['example'] is not None
