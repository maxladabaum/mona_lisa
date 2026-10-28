import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from origami_composite import image_region_preview_points, RegionPreviewCancelled, _progress_heartbeat, pose_basis
from test_origami_composite import footprint, settings, payload
from test_composite_export_progress import window_method
from test_composite_session import Variable


def test_chunk_scan_matches_old_spatial_query_for_rotated_example(tmp_path):
    rng = np.random.default_rng(182)
    source = np.memmap(tmp_path / 'source.dat', mode='w+', shape=(10003, 2), dtype=np.float32)
    source[:] = rng.uniform(-30, 30, source.shape)
    source[-2:] = [6, 2]  # preserve duplicate localizations
    corners = footprint((2, -3), 73)
    grid = np.array([[-3, -2], [3, 2]])
    cfg = settings()
    messages = []
    actual, count = image_region_preview_points(source, corners, grid, cfg, messages.append,
                                                chunk_size=137, max_points=len(source))
    center, basis = pose_basis(corners)
    radius = max(np.linalg.norm(grid, axis=1).max(), np.hypot(cfg.left_nm, cfg.top_nm) + np.hypot(cfg.width_nm, cfg.height_nm))
    ids = sorted(cKDTree(source).query_ball_point(center, radius))
    expected = (source[ids] - center) @ basis
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert count == len(ids)
    assert '100%' in messages[-1]


def test_dense_preview_is_bounded_reproducible_and_counts_all_matches():
    source = np.random.default_rng(83).uniform(-1, 1, (12345, 2))
    args = (source, footprint(), np.array([[0., 0.]]), settings())
    first, count = image_region_preview_points(*args, max_points=100, chunk_size=137)
    second, other_count = image_region_preview_points(*args, max_points=100, chunk_size=137)
    assert first.shape == (100, 2)
    assert count == other_count == len(source)
    np.testing.assert_array_equal(first, second)
    source_points = set(map(tuple, source))
    assert all(tuple(point) in source_points for point in first)


def test_preview_stops_between_chunks_and_rejects_nonfinite_source():
    source = np.zeros((301, 2))
    cancelled = threading.Event()
    def progress(text):
        if '100/301' in text:
            cancelled.set()
    with pytest.raises(RegionPreviewCancelled):
        image_region_preview_points(source, footprint(), np.zeros((1, 2)), settings(), progress,
                                    cancelled.is_set, chunk_size=100)
    source[-1] = np.nan
    with pytest.raises(ValueError, match='finite'):
        image_region_preview_points(source, footprint(), np.zeros((1, 2)), settings(), chunk_size=100)


@pytest.mark.parametrize('cancel', [False, True])
def test_preview_returns_while_scanning_and_uses_selected_template_geometry(cancel):
    entered, release = threading.Event(), threading.Event()
    threads, shown = [], []
    a = payload([footprint()], [False])
    b = payload([footprint()], [True])
    a['params'] = {'grid': np.array([[100., 100.]])}
    b['params'] = {'grid': np.array([[2., 2.]])}
    source = np.array([[6., 2.]])
    results = dict(a=a, b=b)
    data = SimpleNamespace(origami_multi_template_results=results, origami_source_points_nm=source)
    window = SimpleNamespace(busy=False, preview_cancel=None, _closed=False, data_owner=data,
                             settings=settings, first_params=lambda: a['params'], full_grid=lambda params: params['grid'],
                             events=queue.Queue(), status=Variable(''), after=lambda *args: None, poll=lambda: None,
                             show_region_preview=shown.append)
    def exclusions(results, report):
        assert list(results) == ['b']
        return dict(b=np.array([[3., 3.]]))
    window.exclusions = exclusions
    def scan(source, corners, grid, cfg, report, cancelled):
        np.testing.assert_array_equal(grid, b['params']['grid'])
        entered.set()
        assert release.wait(2)
        if cancelled():
            raise RegionPreviewCancelled()
        return source.copy(), len(source)
    def make_thread(**kwargs):
        thread = threading.Thread(**kwargs)
        threads.append(thread)
        return thread
    namespace = dict(threading=SimpleNamespace(Event=threading.Event, Thread=make_thread), np=np,
                     _progress_heartbeat=_progress_heartbeat, image_region_preview_points=scan,
                     RegionPreviewCancelled=RegionPreviewCancelled,
                     messagebox=SimpleNamespace(showerror=lambda *args, **kwargs: pytest.fail(str(args))))
    preview = window_method('preview_region', namespace)
    try:
        preview(window)
        assert entered.wait(2)
        assert window.busy  # click handler returned while scanning is still blocked
        if cancel:
            window_method('cancel_region_preview', {})(window)
    finally:
        release.set()
        for thread in threads:
            thread.join(2)
    poll = window_method('poll', dict(queue=queue, record_diagnostic=lambda *args, **kwargs: None,
                                     messagebox=namespace['messagebox']))
    poll(window)
    assert not window.busy and window.preview_cancel is None
    if cancel:
        assert not shown and 'cancelled' in window.status.get()
    else:
        assert shown[0]['example'] == 'b'
        np.testing.assert_array_equal(shown[0]['sites'], [[3., 3.]])


def test_stale_preview_is_not_displayed():
    cfg = settings()
    data = SimpleNamespace(origami_multi_template_results={}, origami_source_points_nm=np.zeros((1, 2)))
    window = SimpleNamespace(settings=lambda: cfg, data_owner=data, status=Variable(''))
    preview = dict(settings=cfg, results={}, source=data.origami_source_points_nm)
    method = window_method('show_region_preview', dict(tk=SimpleNamespace(TclError=RuntimeError)))
    method(window, preview)
    assert 'Inputs changed' in window.status.get()
