"""Alignment edits are committed explicitly and simulations never touch Tk workers."""
from dataclasses import replace
from threading import Event, get_ident
from time import monotonic, sleep
from unittest.mock import Mock, patch

import _tkinter
import numpy as np
import pytest
from PIL import Image

from alignment_reconstruction import AlignmentModel, AlignmentSettings
from rgb_simulation import cached_alignment_model, simulate_alignment
from test_alignment_reconstruction import designs

LIGHT = dict(linear_light=False, normalize_designs=False, compensate_area=False)


def test_count_change_reuses_geometry_and_optimizer_matrix():
    ds, cache = designs(), {}
    settings = AlignmentSettings(origami_per_design=2)
    first = cached_alignment_model(cache, ds, settings, 7.5, 1., LIGHT)
    target = np.zeros_like(first.expected(ds).density)
    first.optimize(ds, target)
    with patch('rgb_simulation.AlignmentModel', side_effect=AssertionError('rebuilt geometry')):
        second = cached_alignment_model(cache, ds, replace(settings, origami_per_design=20), 7.5, 1., LIGHT)
    assert second.mean is first.mean
    assert second._operators['gram'] is first._operators['gram']
    assert first.settings.origami_per_design == 2
    assert second.settings.origami_per_design == 20
    assert first.evaluate(ds, target)['alignment_variance'] == pytest.approx(
        10 * second.evaluate(ds, target)['alignment_variance'])


def test_separable_covariance_matches_full_kernel():
    from alignment_reconstruction import ERROR_SAMPLES, TRAIN_SEED, transformed_sites
    ds, settings = designs(), AlignmentSettings()
    model = AlignmentModel(ds, settings)
    errors = np.random.default_rng(TRAIN_SEED).normal(size=(ERROR_SAMPLES//2, 3))
    errors = np.concatenate((errors, -errors)) * [settings.translation_x_nm, settings.translation_y_nm, settings.rotation_deg]
    kernels = [model.kernel(transformed_sites(ds[0], settings, error)) for error in errors]
    mean = np.mean(kernels, axis=0)
    covariance = np.mean([k.T @ k for k in kernels], axis=0) - mean.T @ mean
    np.testing.assert_allclose(model.mean, mean, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(model.covariances[0], covariance, rtol=1e-10, atol=1e-12)


def test_batched_example_matches_individual_origami():
    from alignment_reconstruction import transformed_sites
    ds, settings = designs(), AlignmentSettings(origami_per_design=35)
    model = AlignmentModel(ds, settings)
    rng = np.random.default_rng(42)
    density = np.zeros((model.height, model.width, 3))
    for design in ds:
        weights = np.array([s['replacement_fraction'] for s in design['image_sites']])
        errors = rng.normal(size=(35, 3)) * [settings.translation_x_nm, settings.translation_y_nm, settings.rotation_deg]
        for error in errors:
            density[:, :, design['channel']] += (model.kernel(transformed_sites(design, settings, error)) @ weights).reshape(model.height, model.width) / 35
    np.testing.assert_allclose(model.example(ds, 42).density, density, rtol=1e-12, atol=1e-14)


def pump(app, condition, timeout=10):
    deadline = monotonic() + timeout
    while not condition():
        assert monotonic() < deadline, 'Timed out waiting for background simulation'
        app.tk.dooneevent(_tkinter.DONT_WAIT)
        sleep(.001)


@pytest.fixture
def view(tmp_path, monkeypatch):
    import picklist_app
    for name in ('CONFIG_PATH', 'PLATE_STATE_PATH', 'STORAGE_STATE_PATH'):
        monkeypatch.setattr(picklist_app, name, tmp_path / (name + '.json'))
    app = picklist_app.PicklistApp()
    app.withdraw()
    view = app.rgb_image_view
    view.optimize_enabled.set(False)
    view.tile_count.set('1')
    view._tile_count_changed()
    path = tmp_path / 'image.png'
    Image.new('RGB', (6, 8), (128, 100, 64)).save(path)
    view.image_path.set(str(path))
    view._load()
    try:
        yield app, view
    finally:
        app.destroy()


def test_typing_and_focus_changes_do_not_simulate(view):
    app, view = view
    previous = view.reconstruction
    with patch.object(view, '_volumes_changed') as run:
        for value in ('', '1', '10', '100'):
            view.alignment_values['origami_per_design'].set(value)
            app.tk.dooneevent(_tkinter.DONT_WAIT)
        run.assert_not_called()
        assert view._alignment_settings().origami_per_design == 20
        assert view.reconstruction is previous
        view._apply_alignment_settings()
        run.assert_called_once_with(background=True)
        assert view._alignment_settings().origami_per_design == 100
    with patch.object(view, '_volumes_changed') as run:
        view.alignment_values['origami_per_design'].set('1.5')
        view._apply_alignment_settings()
        run.assert_not_called()
        assert view._alignment_settings().origami_per_design == 100
        assert 'Check alignment settings' in view.simulation_status.get()


def test_background_progress_responsiveness_and_stale_results(view, monkeypatch):
    app, view = view
    started, release = Event(), Event()
    main_thread = get_ident()
    calls = []

    def work(*args):
        assert get_ident() != main_thread
        calls.append(args[1].origami_per_design)
        if len(calls) == 1:
            args[-1](12, 'Working…')
            started.set()
            assert release.wait(5)
        return simulate_alignment(*args)

    monkeypatch.setattr('rgb_image_view.simulate_alignment', work)
    comparison = Mock()
    comparison.winfo_exists.return_value = True
    view.comparison_window = comparison
    try:
        view.alignment_values['origami_per_design'].set('7')
        view._apply_alignment_settings()
        pump(app, started.is_set)
        pump(app, lambda: view.simulation_progress['value'] == 12)
        heartbeat = []
        app.after(1, lambda: heartbeat.append(True))
        pump(app, lambda: bool(heartbeat))
        assert str(view.prepare_button['state']) == 'disabled'
        first_cancel = view._simulation_cancel
        view.alignment_values['origami_per_design'].set('9')
        view._apply_alignment_settings()
        assert first_cancel.is_set()
        release.set()
        pump(app, lambda: view._simulation_cancel is None)
        assert calls == [7, 9]
        assert view.alignment_example.metadata['alignment']['origami_per_design'] == 9
        assert view.simulation_progress['value'] == 100
        assert view.simulation_status.get() == 'Simulation complete'
        assert str(view.prepare_button['state']) == 'normal'
        comparison.mark_stale.assert_called_once()
        comparison.refresh.assert_not_called()
    finally:
        view.comparison_window = None
        release.set()


def test_worker_error_restores_controls(view, monkeypatch):
    app, view = view
    monkeypatch.setattr('rgb_image_view.simulate_alignment', lambda *args: (_ for _ in ()).throw(ValueError('test failure')))
    view._apply_alignment_settings()
    pump(app, lambda: view._simulation_cancel is None)
    assert 'test failure' in view.simulation_status.get()
    assert str(view.prepare_button['state']) == 'normal'
    assert view.designs == []


def test_worker_optimization_matches_synchronous_model(tmp_path):
    ds, settings = designs(), AlignmentSettings(origami_per_design=3)
    path = tmp_path / 'target.png'
    Image.new('RGB', (24, 32), (100, 80, 60)).save(path)
    from image_encoding import FIT_MODES
    from gaussian_optimization import fitted_target
    cache, updates = {}, []
    result = simulate_alignment(ds, settings, 7.5, 1., LIGHT, str(path), FIT_MODES[0],
                                True, .1, 42, cache, Event(), lambda n, label: updates.append(n))
    model = cached_alignment_model(cache, ds, settings, 7.5, 1., LIGHT)
    target = fitted_target(str(path), FIT_MODES[0], model.size, linear_light=False)
    expected, _ = model.optimize(ds, target, regularization_strength=.1)
    assert result['designs'] == expected
    np.testing.assert_allclose(result['reconstruction'].density, model.expected(expected).density)
    np.testing.assert_allclose(result['example'].density, model.example(expected, 42).density)
    assert result['metrics']['validation_used_for_selection'] is False
    assert updates == sorted(updates)
    assert updates[-1] == 100


def test_destroy_cancels_pending_work(view, monkeypatch):
    app, view = view
    started, finished = Event(), Event()

    def work(*args):
        started.set()
        args[-2].wait(5)
        finished.set()
        from concurrent.futures import CancelledError
        raise CancelledError()

    monkeypatch.setattr('rgb_image_view.simulate_alignment', work)
    view._apply_alignment_settings()
    pump(app, started.is_set)
    cancelled = view._simulation_cancel
    view.destroy()
    assert cancelled.is_set()
    assert finished.wait(2)
