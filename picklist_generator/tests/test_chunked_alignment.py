"""Chunk boundaries preserve alignment, Gaussian overlap, and optimization."""
from copy import deepcopy
from dataclasses import replace
from concurrent.futures import CancelledError

import numpy as np
import pytest

from alignment_reconstruction import AlignmentModel, AlignmentSettings
from alignment_chunks import ChunkedMean, CHUNK_EDGE
from image_encoding import CHANNELS, compile_designs, encoding_scheme, geometry


def make_designs(key, rotation=0, columns=None):
    info = geometry(key, active_image_columns=columns)
    pixels = [[[((x*23+y*31) % 256)]*3 for x in range(info['columns'])] for y in range(info['rows'])]
    return compile_designs(pixels, 100, {c: 1 for c in CHANNELS}, key, rotation,
                           active_image_columns=columns)


@pytest.mark.parametrize('key,rotation,columns', [
    ('tiles_2x1', 0, None),
    ('tiles_2x1', 90, ['06', '04', '02']),
    ('mono_6_tiles_2x2', 180, ['05', '04', '03', '02', '01']),
    ('mono_6_tiles_2x2', 270, None),
])
def test_chunked_matches_dense_across_tile_edges(key, rotation, columns):
    ds = make_designs(key, rotation, columns)
    settings = AlignmentSettings(3, 2, 4, 5, variability_penalty=.2)
    light = dict(linear_light=True, normalize_designs=True, compensate_area=True)
    dense = AlignmentModel(ds, settings, chunked=False, **light)
    chunked = AlignmentModel(ds, settings, chunked=True, **light)
    assert isinstance(chunked.mean, ChunkedMean)
    assert all(max(block.image_shape) <= CHUNK_EDGE for block in chunked.mean.blocks)
    assert chunked.size == dense.size
    np.testing.assert_allclose(chunked.expected(ds).density, dense.expected(ds).density, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(chunked.example(ds, 42).density, dense.example(ds, 42).density, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(chunked.covariances, dense.covariances, rtol=1e-10, atol=1e-12)
    target = np.full_like(dense.expected(ds).density, .15)
    assert chunked.evaluate(ds, target) == pytest.approx(dense.evaluate(ds, target), rel=1e-10, abs=1e-12)
    a, metrics_a = dense.optimize(ds, target, regularization_strength=.05)
    b, metrics_b = chunked.optimize(ds, target, regularization_strength=.05)
    np.testing.assert_array_equal(dense.weights(a), chunked.weights(b))
    assert metrics_a['optimized_regularized_objective'] == pytest.approx(metrics_b['optimized_regularized_objective'], abs=1e-12)
    assert all(old['bits'] == new['bits'] for old, new in zip(ds, b))


def test_chunk_operator_adjoint_gram_and_coordinate_updates():
    ds = make_designs('mono_6_tiles_2x1')
    dense = AlignmentModel(ds, AlignmentSettings(), chunked=False)
    chunks = AlignmentModel(ds, dense.settings, chunked=True)
    rng = np.random.default_rng(7)
    weights = rng.normal(size=(dense.mean.shape[1], 3))
    pixels = rng.normal(size=(dense.mean.shape[0], 3))
    np.testing.assert_allclose(chunks.mean.T @ pixels, dense.mean.T @ pixels, atol=1e-12)
    assert np.sum((chunks.mean @ weights)*pixels) == pytest.approx(np.sum(weights*(chunks.mean.T @ pixels)), abs=1e-12)
    np.testing.assert_allclose(chunks.mean.T @ chunks.mean, dense.mean.T @ dense.mean, atol=1e-12)
    np.testing.assert_allclose(chunks.mean.column_squared_norms(), np.sum(dense.mean**2, axis=0), atol=1e-12)
    residual = pixels[:, 0].copy()
    for i in (0, 47, 48, 95):
        assert chunks.mean.column_dot(i, residual) == pytest.approx(dense.mean[:, i] @ residual, abs=1e-12)
        expected = residual + .3*dense.mean[:, i]
        chunks.mean.add_column(residual, i, .3)
        np.testing.assert_allclose(residual, expected, atol=1e-12)


def test_chunked_geometry_reuse_and_cancellation():
    ds = make_designs('mono_6_tiles_2x1')
    untouched = deepcopy(ds)
    count = [0]
    def cancel():
        count[0] += 1
        if count[0] == 7:
            raise CancelledError()
    with pytest.raises(CancelledError):
        AlignmentModel(ds, AlignmentSettings(), chunked=True, checkpoint=cancel)
    assert ds == untouched
    model = AlignmentModel(ds, AlignmentSettings(), chunked=True)
    reused = model.with_settings(replace(model.settings, origami_per_design=100))
    assert reused.mean is model.mean
    target = np.zeros_like(model.expected(ds).density)
    assert reused.evaluate(ds, target)['alignment_variance'] == pytest.approx(model.evaluate(ds, target)['alignment_variance']/5)
    with pytest.raises(CancelledError):
        model.example(ds, progress=lambda fraction: (_ for _ in ()).throw(CancelledError()))


def test_252_tile_ui_runs_in_background_and_reuses_chunks(tmp_path, monkeypatch):
    import json
    import picklist_app
    from PIL import Image
    from test_rgb_simulation import pump
    for name in ('CONFIG_PATH', 'PLATE_STATE_PATH', 'STORAGE_STATE_PATH'):
        monkeypatch.setattr(picklist_app, name, tmp_path / (name + '.json'))
    picklist_app.CONFIG_PATH.write_text(json.dumps({
        'image_encoding_scheme': 'mono_8_tiles_18x14',
        'image_active_columns': ['05', '04', '03', '02', '01']}))
    app = picklist_app.PicklistApp()
    app.withdraw()
    try:
        view = app.rgb_image_view
        view.optimize_enabled.set(False)
        path = tmp_path / 'grayscale.png'
        Image.fromarray(np.tile(np.arange(0, 256, 4, dtype=np.uint8), (64, 1))).save(path)
        view.image_path.set(str(path))
        view._load()
        assert view._simulation_cancel is not None
        pump(app, lambda: view._simulation_cancel is None, timeout=30)
        assert len(view.designs) == 252
        assert view.reconstruction.metadata['alignment_backend'] == 'chunked'
        assert view.alignment_example is not None
        assert view.ideal_reconstruction is not None
        assert 'Chunked alignment' in view.alignment_summary.get()
        assert 'expected alignment result' in view.prediction_frame['text']
        assert view.simulation_status.get() == 'Simulation complete'
        mean = view._alignment_model(view.designs).mean
        view.alignment_values['origami_per_design'].set('3')
        view._apply_alignment_settings()
        pump(app, lambda: view._simulation_cancel is None, timeout=30)
        assert view._alignment_model(view.designs).mean is mean
        assert view.alignment_example.metadata['alignment']['origami_per_design'] == 3
        assert view.simulation_progress['value'] == 100
    finally:
        app.destroy()
