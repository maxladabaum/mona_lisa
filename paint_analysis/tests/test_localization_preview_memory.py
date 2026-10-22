import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from origami_analysis import render_localization_preview


def test_chunked_mapped_preview_matches_full_histogram(tmp_path, monkeypatch):
    points = np.lib.format.open_memmap(tmp_path / 'points.npy', mode='w+', dtype=np.float32, shape=(250_003, 2))
    points[:] = np.random.default_rng(13).uniform(-30, 60, points.shape)
    points[100_000] = (np.nan, 2)
    points[200_000] = (3, np.inf)
    points[0] = (-30, -30)
    points[-1] = (60, 60)
    original = np.histogram2d
    finite = np.asarray(points)[np.isfinite(points).all(axis=1)]
    density, _, _ = original(finite[:, 0], finite[:, 1], bins=(90, 90), range=((-30,60),(-30,60)))
    density = gaussian_filter(density, sigma=1, mode='constant')
    expected = np.clip(density.T / (density.max() * .5), 0, 1)
    sizes = []
    def bounded(x, y, **kwargs):
        sizes.append(len(x))
        assert len(x) <= 100_000
        return original(x, y, **kwargs)
    monkeypatch.setattr(np, 'histogram2d', bounded)
    actual = render_localization_preview(points)
    np.testing.assert_allclose(actual['contrast'], expected)
    assert len(sizes) == 3
    assert actual['extent'] == (-30, 60, -30, 60)


def test_large_mapped_source_has_bounded_python_allocations(tmp_path):
    import tracemalloc
    points = np.lib.format.open_memmap(tmp_path / 'large.npy', mode='w+', dtype=np.float32, shape=(2_000_000, 2))
    points[:] = 1
    tracemalloc.start()
    try:
        preview = render_localization_preview(points, max_pixels=100)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 16_000_000, f'Preview allocated {peak} bytes for a mapped source'
    assert preview['contrast'].size <= 100


def test_degenerate_and_elongated_extents_respect_pixel_limit():
    result = render_localization_preview(np.array([[0, 0], [1e7, 0]]), max_pixels=50, blur_nm=0)
    assert result['contrast'].size <= 50
    assert np.isfinite(result['contrast']).all()
    assert result['extent'][3] > result['extent'][2]
    with pytest.raises(ValueError, match='finite'):
        render_localization_preview(np.array([[np.nan, 0]]))
