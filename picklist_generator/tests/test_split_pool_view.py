import pytest
from test_rgb_synthesis import app


@pytest.fixture(autouse=True)
def ideal_reference_defaults(monkeypatch):
    """Existing plot tests explicitly exercise the unrestricted probability mode."""
    import split_pool_view
    original = split_pool_view.SplitPoolWindow.__init__
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.physical_enabled.set(False)
    monkeypatch.setattr(split_pool_view.SplitPoolWindow, '__init__', initialize)


def test_analysis_subtab_preserves_state_and_closes_with_owner(app):
    import tkinter as tk
    view = app.rgb_image_view
    app.rgb_subtabs.select(view._split_pool_page)
    app.update()
    window = view._split_pool_window
    assert window.winfo_exists()
    assert not isinstance(window, tk.Toplevel)
    assert window.winfo_toplevel() is app
    assert app.rgb_subtabs.tab(view._split_pool_page, 'text') == 'Split–pool encoding analysis'
    assert window.source.get() == 'Current RGB designs'
    window.basis_k.set('8')
    app.rgb_subtabs.select(0); app.update()
    assert window.winfo_exists()
    view._open_split_pool()
    assert view._split_pool_window is window
    assert window.basis_k.get() == '8'
    assert app.rgb_subtabs.select() == str(view._split_pool_page)
    view.destroy()
    assert window.cancel.is_set()
    assert not window.winfo_exists()


def test_visualize_in_memory_and_switch_plots(app, monkeypatch):
    import numpy as np
    from unittest.mock import Mock
    import split_pool_view
    from split_pool_encoding import datasets, experiment

    class ImmediateThread:
        def __init__(self, target, **kwargs):
            self.target = target
        def start(self):
            self.target()

    monkeypatch.setattr(split_pool_view, 'Thread', ImmediateThread)
    choose_directory = Mock(side_effect=AssertionError('Visualization must not ask to save'))
    monkeypatch.setattr(split_pool_view.filedialog, 'askdirectory', choose_directory)
    monkeypatch.setattr(experiment, 'run_dataset', Mock(side_effect=AssertionError('No disk export during visualization')))
    monkeypatch.setitem(experiment.DEFAULTS, 'alternations', 2)
    monkeypatch.setitem(experiment.DEFAULTS, 'maxiter', 15)
    monkeypatch.setitem(experiment.DEFAULTS, 'measurement_repeats', 2)
    d = np.array([[.1, .2, .3], [.2, .3, .4], [.3, .4, .5], [.4, .5, .6]])
    monkeypatch.setattr(datasets, 'generate', lambda *a, **k: (d, {}))
    view = app.rgb_image_view
    view._open_split_pool(); window = view._split_pool_window
    window.source.set('Synthetic benchmarks'); window.restarts.set('1'); window.k_sweep.set('4')
    window.start(sweep=True)
    window.after_cancel(window._poll_id); window._poll_id = None
    window.poll()
    assert not window.running
    assert set(window.results) == {'pca', 'additive', 'on_only', 'on_off'}
    assert 'Analysis ready' in window.status.get()
    choose_directory.assert_not_called()
    experiment.run_dataset.assert_not_called()
    window.model.set('on_off')
    for plot in window.plot_picker['values']:
        window.plot.set(plot); window.draw(); window.canvas.draw()
        assert window.figure.axes
        assert 'RMSE' in window.summary.get()
    window.plot.set('Analog canvas'); window.address.set(999); window.draw()
    assert window.address.get() == 3
    window.plot.set('Sampling error'); window.model.set('pca'); window.draw()
    assert 'not a physical' in window.figure.axes[0].texts[0].get_text()
    window.close()


@pytest.mark.parametrize("design_count", [1, 3, 4, 21])
def test_full_rgb_monte_carlo_controls_do_not_refit(app, monkeypatch, design_count):
    from unittest.mock import Mock
    import split_pool_view
    from split_pool_encoding import experiment, optimization
    from test_split_pool_image import make_image_fixture

    class ImmediateThread:
        def __init__(self, target, **kwargs): self.target = target
        def start(self): self.target()

    monkeypatch.setattr(split_pool_view, 'Thread', ImmediateThread)
    monkeypatch.setitem(experiment.DEFAULTS, 'alternations', 1)
    monkeypatch.setitem(experiment.DEFAULTS, 'maxiter', 10)
    monkeypatch.setitem(experiment.DEFAULTS, 'measurement_repeats', 2)
    fit = Mock(wraps=optimization.fit); monkeypatch.setattr(optimization, 'fit', fit)
    owner = app.rgb_image_view
    from copy import deepcopy
    originals, _ = make_image_fixture(3)
    owner.designs = []
    for color in ('Red', 'Green', 'Blue'):
        group = [d for d in originals if d['color'] == color]
        for index in range(design_count):
            design = deepcopy(group[index % len(group)])
            design.update(name=f'{color}_{index}', barcode_code=index)
            owner.designs.append(design)
    owner._open_split_pool(); window = owner._split_pool_window
    window.restarts.set('1'); window.k_sweep.set('4'); window.start(sweep=True)
    window.after_cancel(window._poll_id); window._poll_id = None; window.poll()
    assert fit.call_count == 4  # Four models, each fitted once across all colors.
    window.plot.set('Dashboard'); window.draw()
    if window._poll_id:
        window.after_cancel(window._poll_id); window._poll_id = None
    window.poll()
    assert window.plot.get() == 'Dashboard'
    assert set(window.dashboard_sections) == {'Complete synthesis and pooling'}
    assert len(window.figure.axes) == 6
    assert len(window.figure.axes[0].images) == 1
    assert len(window.figure.axes[-2].lines) == 2
    assert 'All colors' in window.summary.get()
    capacity = 2**max(1, (3*design_count-1).bit_length())
    for result in window.results.values():
        addresses = [int(a) for fitted in result['color_fits'].values() for a in fitted['address_indices']]
        assert len(addresses) == len(set(addresses)) == 3*design_count
        assert all(fitted['address_capacity'] == capacity for fitted in result['color_fits'].values())
    assert f'{design_count}/{capacity} addresses used' in window.summary.get()
    assert len(window.figure.axes[0].patches) == 3
    if design_count > 1:
        first_bounds = window.figure.axes[0].patches[0].get_bbox().bounds
        window.location_picker.current(1); window._location_changed()
        assert window.address.get() == 1
        assert window.figure.axes[0].patches[0].get_bbox().bounds != first_bounds
        assert 'Red_1' in window.figure.axes[0].get_title()
        window.location_picker.current(0); window._location_changed()
        assert fit.call_count == 4
    initial_key = (window._revision, 'additive', 300, 42)
    initial_expected = window.image_cache[initial_key][0][0][1].tobytes()
    window.sample_count.set('50'); window.new_example()
    if window._poll_id:
        window.after_cancel(window._poll_id); window._poll_id = None
    window.poll()
    new_key = (window._revision, 'additive', 50, 43)
    assert window.image_cache[new_key][0][0][1].tobytes() == initial_expected
    assert fit.call_count == 4
    assert len(window.figure.axes) == 6
    window.plot.set('Image Monte Carlo'); window.draw()
    assert 'N=50' in window.figure.axes[2].get_title()
    import numpy as np
    pca = window.results['pca']['color_fits']['Red']
    raw = pca['prediction'].copy(); raw_mse = pca['mse']
    window.model.set('pca'); window.draw()
    if window._poll_id:
        window.after_cancel(window._poll_id); window._poll_id = None
    window.poll()
    assert len(window.figure.axes) == 3
    assert all(len(ax.images) == 1 for ax in window.figure.axes)
    assert window.figure.axes[1].get_title() == 'PCA reference (clipped)'
    assert 'Raw PCA RMSE' in window.summary.get()
    assert 'independent of K' in window.summary.get()
    np.testing.assert_array_equal(pca['prediction'], raw)
    assert pca['mse'] == raw_mse
    assert fit.call_count == 4
    window.close()


def test_xy_selector_tracks_target_across_channel_and_address_assignment(app):
    import numpy as np
    from split_pool_encoding.datasets import from_designs
    from split_pool_encoding.evaluation import metrics
    owner = app.rgb_image_view
    owner.designs = [dict(color=color, name=f'{color}_X{x}_Y0', barcode_code=x,
                         x_shift=x, y_shift=0, image_sites=[{'name': 'a'}],
                         site_fractions={'a': value+x*.1})
                     for color, value in [('Red', .1), ('Blue', .6)] for x in (0, 1)]
    owner._open_split_pool(); window = owner._split_pool_window
    window.image_designs = owner.designs
    fits = {}
    for color in ('Red', 'Blue'):
        data, metadata = from_designs(owner.designs, color)
        window.detail_datasets[color] = data; window.detail_metadata[color] = metadata
        pi = np.array([1, 0])
        p = data[pi]+.025
        fits[color] = dict(model='additive', prediction=p, assignment=pi,
                           address_indices=np.array([1, 3]), metrics=metrics(data[pi], p),
                           runtime=0, history=[], noise=[])
    window.data = window.detail_datasets['Red']
    window.results = {'additive': {**fits['Red'], 'color_fits': fits}}
    window._update_locations()
    assert tuple(window.location_picker['values']) == ('X 0, Y 0', 'X 1, Y 0')
    window.plot.set('Analog canvas')
    window.location_picker.current(1); window._location_changed()
    assert window.plot.get() == 'Analog canvas'
    ax = window.figure.axes[0]
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [.2])
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [.225])
    assert 'Address 1' in ax.get_title() and 'Red_X1_Y0' in ax.get_title()
    window.color.set('Blue'); window._channel_changed()
    assert window.location.get() == 'X 1, Y 0'
    ax = window.figure.axes[0]
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [.7])
    np.testing.assert_allclose(ax.lines[1].get_ydata(), [.725])
    assert 'Blue_X1_Y0' in ax.get_title()
    window.close()


@pytest.mark.parametrize("trigger", ["button", "selector", "while_running"])
def test_k_sweep_and_select_without_refitting(app, monkeypatch, trigger):
    import numpy as np
    import split_pool_view
    from split_pool_encoding import datasets, experiment, optimization
    from unittest.mock import Mock

    class ImmediateThread:
        def __init__(self, target, **kwargs): self.target = target
        def start(self): self.target()

    monkeypatch.setattr(split_pool_view, 'Thread', ImmediateThread)
    monkeypatch.setitem(experiment.DEFAULTS, 'alternations', 1)
    monkeypatch.setitem(experiment.DEFAULTS, 'maxiter', 10)
    monkeypatch.setitem(experiment.DEFAULTS, 'measurement_repeats', 1)
    monkeypatch.setattr(datasets, 'generate', lambda *a, **k: (np.array([[.1, .6], [.2, .4], [.7, .2], [.8, .3]]), {}))
    spy = Mock(wraps=optimization.fit); monkeypatch.setattr(optimization, 'fit', spy)
    app.rgb_image_view._open_split_pool(); window = app.rgb_image_view._split_pool_window
    window.source.set('Synthetic benchmarks'); window.k_sweep.set('1,2,Unrestricted'); window.restarts.set('1')
    initial_calls = 0
    if trigger == 'while_running':
        window.basis_k.set('4'); window.start(sweep=False)
        initial_calls = spy.call_count
        window.plot.set('RMSE vs K'); window._plot_changed()
        assert spy.call_count == initial_calls
        window.after_cancel(window._poll_id); window._poll_id = None; window.poll()
        assert not window.running
        assert set(window.k_results) == {'4'}
        assert spy.call_count == initial_calls
        window.sweep_button.invoke()
    elif trigger != 'button':
        window.basis_k.set('4'); window.start(sweep=False)
        window.after_cancel(window._poll_id); window._poll_id = None; window.poll()
        assert set(window.k_results) == {'4'}
        initial_calls = spy.call_count
        window.plot.set('RMSE vs K')
        window._plot_changed()
        assert not window.running
        assert spy.call_count == initial_calls
        assert 'Click Fit RMSE vs K' in window.summary.get()
        window.sweep_button.invoke()
    else:
        window.start(sweep=True)
    window.after_cancel(window._poll_id); window._poll_id = None; window.poll()
    assert not window.running
    assert set(window.k_results) == {'1', '2', 'Unrestricted'}
    assert window.progress_value.get() == 100
    assert window.progress_text.get() == '100% • fits complete'
    assert spy.call_count == initial_calls+12
    assert 'Completed 3/3 K values' in window.summary.get()
    np.testing.assert_array_equal(window.figure.axes[0].get_xticks(), [1, 2])
    window._plot_changed()
    assert spy.call_count == initial_calls+12
    assert window.plot.get() == 'RMSE vs K'
    ax = window.figure.axes[0]
    for line in ax.lines[:1]:
        np.testing.assert_array_equal(line.get_xdata(), [1, 2])
        expected = [window.k_results[k]['additive']['metrics']['rmse'] for k in ('1', '2')]
        np.testing.assert_allclose(line.get_ydata(), expected)
    window.basis_k.set('2'); window._select_k()
    assert window.results['on_off']['basis_k'] == 2
    assert spy.call_count == initial_calls+12
    window.close()


@pytest.mark.parametrize('extra_rounds', [0, 1])
@pytest.mark.parametrize('response', ['Concentration-dependent capture (C50)', 'RGB ratio (complete occupancy)'])
def test_physical_preview_uses_two_stage_recipes_and_stock_settings(app, monkeypatch, extra_rounds, response):
    import numpy as np
    import split_pool_view
    from split_pool_encoding import experiment
    from split_pool_encoding.models import transitions
    from test_split_pool_image import make_image_fixture

    class ImmediateThread:
        def __init__(self, target, **kwargs): self.target = target
        def start(self): self.target()

    monkeypatch.setattr(split_pool_view, 'Thread', ImmediateThread)
    monkeypatch.setitem(experiment.DEFAULTS, 'alternations', 1)
    monkeypatch.setitem(experiment.DEFAULTS, 'maxiter', 10)
    monkeypatch.setitem(experiment.DEFAULTS, 'measurement_repeats', 1)
    owner = app.rgb_image_view; owner.designs, _ = make_image_fixture(3)
    owner._open_split_pool(); window = owner._split_pool_window
    window.model.set('on_off')
    window.physical_enabled.set(True); window.restarts.set('1'); window.basis_k.set('2')
    window.extra_rounds.set(str(extra_rounds))
    window.split_response.set(response); window._response_changed()
    for key, value in dict(vector_volume_nl=10000, vector_amount_fmol=10000,
                           branch_volume_nl=2000, branch_budget_nl=1000, capture_c50_nm=50).items():
        window.recipe_variables[key].set(str(value))
    original_settings = window._read_recipe_settings()
    window.recipe_variables['vector_volume_nl'].set('20000')
    diluted = window._read_recipe_settings()
    assert diluted.vector_max_nm == original_settings.vector_max_nm/2
    window.recipe_variables['vector_volume_nl'].set('10000')
    window.start()
    window.after_cancel(window._poll_id); window._poll_id = None; window.poll()
    assert not window.running
    assert window.model.get() == 'on_off'
    assert '25 nL at both stages' in window.summary.get()
    assert len(window.figure.axes) == 6
    for fit in window.results['on_off']['color_fits'].values():
        recipe = fit['recipe']
        assert recipe['response_model'] == ('rgb_ratio' if response.startswith('RGB') else 'capture')
        expected_rounds = max(1, (len(owner.designs)-1).bit_length())+extra_rounds
        assert fit['address_bits'] == expected_rounds
        assert fit['extra_rounds'] == extra_rounds
        assert recipe['branch_transfers_nl'].shape == (2*expected_rounds, 2)
        assert recipe['u'].shape[0] == expected_rounds
        assert recipe['settings']['on_stock_nm'] == float(owner.replacement_stock.get())
        assert recipe['settings']['off_stock_nm'] == float(owner.base_stock.get())
        assert not np.any(recipe['source_transfers_nl'] % 25)
        assert not np.any(recipe['branch_transfers_nl'] % 25)
        expected = transitions(recipe['u'], recipe['v'])[1][fit['address_indices']]
        np.testing.assert_allclose(fit['prediction'], expected)
    window.plot.set('Transfer recipe'); window.draw()
    assert 'realized recipe' in window.summary.get()
    assert len(window.figure.axes) == 6  # Separate ON, OFF, and branch heatmaps with colorbars.
    plots = [ax for ax in window.figure.axes if ax.images]
    assert [t.get_text() for t in plots[2].get_yticklabels()][:2] == ['Round 1 · branch 0', 'Round 1 · branch 1']
    assert plots[0].get_yticklabels()[0].get_text() == 'K1'
    window.plot.set('Well trajectory'); window.address.set(1); window.draw(); window.canvas.draw()
    detail = window._detail_result(window.results['on_off'])
    assigned = int(np.flatnonzero(detail['assignment'] == 1)[0])
    code = format(int(detail['address_indices'][assigned]), f"0{detail['address_bits']}b")
    expected = detail['recipe']['branch_transfers_nl'][2*np.arange(len(code))+np.array(list(code), int)]
    active = np.flatnonzero(expected.sum(axis=0) > 0)
    assert f'Digital code {code}' in window.figure._suptitle.get_text()
    assert window.color.get() in window.figure._suptitle.get_text()
    heatmaps = [ax for ax in window.figure.axes if ax.images]
    if len(active):
        np.testing.assert_array_equal(heatmaps[0].images[0].get_array(), expected[:, active].T)
        assert 'nL' in heatmaps[0].format_coord(0, 0)
    window._location_changed()
    assert window.plot.get() == 'Well trajectory'
    window.model.set('additive'); window.plot.set('Image Monte Carlo'); window.draw()
    assert 'no ON/OFF stock recipe' in window.figure.axes[0].texts[0].get_text()
    window.close()


@pytest.mark.parametrize('model', ['pca', 'on_off', 'additive'])
def test_synthetic_monte_carlo_displays_matrices_and_resamples(app, model):
    import numpy as np
    from split_pool_encoding.evaluation import metrics
    owner = app.rgb_image_view; owner._open_split_pool(); window = owner._split_pool_window
    d = np.array([[.1, .2, .3], [.2, .3, .4], [.3, .4, .5], [.4, .5, .6]])
    pi = np.array([2, 0, 3, 1]); prediction = d[pi]+.05
    window.data = d; window.metadata = None
    result = dict(model=model, prediction=prediction, assignment=pi, runtime=0,
                  metrics=metrics(d[pi], prediction), color_fits=None, physical_mode=True)
    if model == 'on_off':
        result['recipe'] = {'realized': True}
    window.results = {model: result}; window.model.set(model); window.plot.set('Image Monte Carlo')
    window.draw(); window.canvas.draw()
    images = [ax.images[0] for ax in window.figure.axes if ax.images]
    assert len(images) == 3
    np.testing.assert_allclose(images[0].get_array(), d)
    np.testing.assert_allclose(images[1].get_array(), d+.05)
    sample = np.asarray(images[2].get_array()).copy()
    assert 'Matrix Monte Carlo' in window.summary.get()
    if model == 'on_off':
        assert '25 nL' in window.summary.get()
    window.new_example()
    new_images = [ax.images[0] for ax in window.figure.axes if ax.images]
    assert not np.array_equal(sample, new_images[2].get_array())
    np.testing.assert_allclose(new_images[1].get_array(), d+.05)
    window.sample_count.set('0'); window.draw()
    assert 'positive integer' in window.figure.axes[0].texts[0].get_text()
    window.close()


def test_progress_bar_keeps_partial_progress_on_cancellation(app):
    owner = app.rgb_image_view; owner._open_split_pool(); window = owner._split_pool_window
    assert window.progress_value.get() == 0
    window.running = True
    window.events.put(('fit_progress', (37.5, 'K=2 • on_off • Restart 1/3')))
    window.cancel.set()
    window.events.put(('error', 'Analysis cancelled.'))
    window.poll()
    assert not window.running
    assert window.progress_value.get() == 37.5
    assert 'cancelled' in window.progress_text.get()
    window.close()


def test_k_vector_overview_and_clickable_profile(app):
    import numpy as np
    from types import SimpleNamespace
    from split_pool_encoding.optimization import fit
    from split_pool_encoding.evaluation import metrics
    owner = app.rgb_image_view; owner._open_split_pool(); window = owner._split_pool_window
    window.data = np.array([[.1, .2, .3], [.2, .3, .4], [.3, .4, .5], [.4, .5, .6]])
    window.metadata = None
    result = fit(window.data, 'on_off', basis_k=2, restarts=1, alternations=1, maxiter=5)
    result['metrics'] = metrics(window.data[result['assignment']], result['prediction'])
    concentrations = np.array([[[10., 20.], [30., 40.], [50., 60.]], [[70., 80.], [90., 100.], [110., 120.]]])
    result['recipe'] = {'vector_concentrations_nm': concentrations}
    window.results = {'on_off': result}; window.model.set('on_off'); window.plot.set('K vectors')
    window.draw(); window.canvas.draw()
    assert 'K1–K2 of 2' in window.summary.get()
    assert len(window._vector_axes) == 4
    np.testing.assert_array_equal(window._vector_axes[0].images[0].get_array(), concentrations[0, :, 0].reshape(3, 1))
    np.testing.assert_array_equal(window._vector_axes[1].images[0].get_array(), concentrations[0, :, 1].reshape(3, 1))
    window._pick_vector(SimpleNamespace(inaxes=window._vector_axes[2], ydata=0))
    assert window.vector_index.get() == 2
    detail = [ax for ax in window.figure.axes if ax.images][:2]
    np.testing.assert_array_equal(detail[0].images[0].get_array(), concentrations[1, :, 0].reshape(3, 1))
    np.testing.assert_array_equal(detail[1].images[0].get_array(), concentrations[1, :, 1].reshape(3, 1))
    window.vector_index.set(999); window.draw()
    assert window.vector_index.get() == 2
    window.plot.set('Target and reconstruction'); window.draw()
    assert str(window.vector_picker['state']) == 'disabled'
    window.close()


def test_branch_walkthrough_renders_code_mixtures_and_canvas(app):
    import numpy as np
    from split_pool_encoding.optimization import fit
    from split_pool_encoding.evaluation import metrics
    owner = app.rgb_image_view; owner._open_split_pool(); window = owner._split_pool_window
    window.data = np.random.default_rng(4).uniform(size=(4, 5)); window.metadata = None
    result = fit(window.data, 'on_off', basis_k=3, restarts=1, alternations=1, maxiter=5)
    result['metrics'] = metrics(window.data[result['assignment']], result['prediction'])
    window.results = {'on_off': result}; window.model.set('on_off')
    window.plot.set('Branch walkthrough'); window.address.set(2)
    window.draw(); window.canvas.draw()
    images = [ax.images[0].get_array() for ax in window.figure.axes if ax.images]
    assert len(images) == 4
    assert images[0].shape == (2, 3)
    np.testing.assert_allclose(images[3][0], result['prediction'][np.flatnonzero(result['assignment'] == 2)[0]])
    assert 'Digital code' in window.figure._suptitle.get_text()
    window.close()


def test_extra_rounds_invalid_or_edited_does_not_launch_fit(app, monkeypatch):
    from unittest.mock import Mock
    import split_pool_view
    from test_split_pool_image import make_image_fixture
    owner = app.rgb_image_view; owner.designs, _ = make_image_fixture(3)
    owner._open_split_pool(); window = owner._split_pool_window
    thread = Mock(); monkeypatch.setattr(split_pool_view, 'Thread', thread)
    error = Mock(); monkeypatch.setattr(split_pool_view.messagebox, 'showerror', error)
    window.extra_rounds.set('2')
    assert 'extra' in window.round_info.get()
    thread.assert_not_called()
    for value in ('-1', '1.5', '12'):
        window.extra_rounds.set(value); window.start()
        assert not window.running
    assert error.call_count == 3
    thread.assert_not_called()
    window.close()


def test_independent_well_mode_end_to_end_rgb_and_plots(app, monkeypatch):
    import numpy as np
    import split_pool_view
    from split_pool_encoding import experiment
    from test_split_pool_image import make_image_fixture
    class ImmediateThread:
        def __init__(self, target, **kwargs): self.target = target
        def start(self): self.target()
    monkeypatch.setattr(split_pool_view, 'Thread', ImmediateThread)
    monkeypatch.setitem(experiment.DEFAULTS, 'alternations', 1)
    monkeypatch.setitem(experiment.DEFAULTS, 'maxiter', 5)
    monkeypatch.setitem(experiment.DEFAULTS, 'measurement_repeats', 1)
    owner = app.rgb_image_view; owner.designs, _ = make_image_fixture(40)
    for design in owner.designs:
        for i, site in enumerate(design['image_sites']):
            site.update(row=i//5, column=i%5)
    owner._open_split_pool(); window = owner._split_pool_window
    window.synthesis_mode.set('Independent wells'); window._mode_changed()
    assert str(window.round_picker['state']) == 'disabled'
    assert str(window.assignment_check['state']) == 'disabled'
    assert window.physical_enabled.get()
    window.extra_rounds.set('invalid ignored'); window.restarts.set('1'); window.basis_k.set('4')
    window.start(); window.after_cancel(window._poll_id); window._poll_id = None; window.poll()
    assert not window.running
    assert set(window.results) == {'on_off'}
    assert window.plot.get() == 'Dashboard'
    assert set(window.dashboard_sections) == {'Complete synthesis and pooling'}
    assert len(window.figure.axes) == 6
    section_widgets = tuple(window.dashboard_frame.winfo_children())
    window.draw()
    assert tuple(window.dashboard_frame.winfo_children()) == section_widgets
    window.plot.set('K vectors'); window.draw()
    assert window._vector_cells
    assert all(ax.images[0].get_array().shape == (8, 5) for ax in window.figure.axes if ax.images)
    window.plot.set('K stock preparation'); window.draw()
    assert all(ax.images[0].get_array().shape == (8, 5) for ax in window.figure.axes if ax.images)
    window.plot.set('Canvas construction'); window.draw()
    construction = window.figure
    original_axes = tuple(construction.axes)
    window._move_build(window._build_count)
    assert window.build_step.get() == window._build_count
    assert 'capture complete' in window.build_status.get()
    assert tuple(construction.axes) == original_axes
    construction_images = [ax.images[0].get_array() for ax in construction.axes if ax.images]
    assert len(construction_images) == 9
    selected_fit = window._detail_result(window.results['on_off'])
    np.testing.assert_allclose(construction_images[7], selected_fit['prediction'][0].reshape(8, 5))
    window.color.set('Green'); window._channel_changed()
    assert window.build_step.get() == 1
    result = window.results['on_off']
    addresses = []
    for color, fit in result['color_fits'].items():
        assert fit['synthesis_mode'] == 'independent_wells'
        np.testing.assert_allclose(fit['prediction'], fit['recipe']['u'])
        assert fit['recipe']['branch_transfers_nl'].shape == (4, 4)
        addresses.extend(fit['address_indices'])
    assert len(set(addresses)) == 12
    window.model.set('on_off'); window.plot.set('Image Monte Carlo'); window.draw()
    assert 'Expected independent-well image' in [ax.get_title() for ax in window.figure.axes]
    for plot in window.plot_picker['values']:
        window.plot.set(plot); window.draw(); window.canvas.draw()
        assert window.figure.axes
        if plot == 'Canvas construction':
            stage_axes = tuple(window.figure.axes)
            stage_images = tuple(ax.images[0] for ax in stage_axes if ax.images)
            window._move_build(1); window.canvas.draw()
            assert tuple(window.figure.axes) == stage_axes
            assert tuple(ax.images[0] for ax in stage_axes if ax.images) == stage_images
    window.close()


def test_poll_batches_model_updates_into_one_dashboard_refresh(app, monkeypatch):
    from unittest.mock import Mock
    owner = app.rgb_image_view; owner._open_split_pool(); window = owner._split_pool_window
    draw = Mock(); monkeypatch.setattr(window, 'draw', draw)
    window.running = True
    for model in ('on_off', 'on_only', 'pca'):
        window.events.put(('result', ('4', {'model': model})))
    window.events.put(('done', 'Analysis ready.'))
    window.poll()
    assert not window.running
    assert set(window.results) == {'on_off', 'on_only', 'pca'}
    assert draw.call_count == 1
    window.close()


@pytest.mark.parametrize('model', ['additive', 'on_only', 'on_off', 'pca'])
@pytest.mark.parametrize('source', ['Current RGB designs', 'Synthetic benchmarks'])
def test_visualize_fits_only_selected_model(app, monkeypatch, model, source):
    from unittest.mock import Mock
    import numpy as np
    import split_pool_view
    from split_pool_encoding import datasets, optimization, experiment
    from test_split_pool_image import make_image_fixture

    class ImmediateThread:
        def __init__(self, target, **kwargs): self.target = target
        def start(self): self.target()

    monkeypatch.setattr(split_pool_view, 'Thread', ImmediateThread)
    monkeypatch.setattr(datasets, 'generate', lambda *a, **k: (np.ones((4, 3))*.5, {}))
    owner = app.rgb_image_view
    owner.designs, _ = make_image_fixture(3)
    window = owner._split_pool_window
    monkeypatch.setattr(window, 'draw', lambda: None)
    monkeypatch.setitem(experiment.DEFAULTS, 'alternations', 1)
    monkeypatch.setitem(experiment.DEFAULTS, 'maxiter', 5)
    monkeypatch.setitem(experiment.DEFAULTS, 'measurement_repeats', 1)
    window.restarts.set('1')
    fit = Mock(wraps=optimization.fit)
    monkeypatch.setattr(optimization, 'fit', fit)
    assert model in window.model_picker['values']
    window.source.set(source); window.model.set(model)
    window.start()
    window.after_cancel(window._poll_id); window._poll_id = None; window.poll()
    assert fit.call_count == 1
    assert {call.args[1] for call in fit.call_args_list} == {model}
    assert set(window.results) == {model}
    assert window.model.get() == model
    assert set(window.model_picker['values']) == {'additive', 'on_only', 'on_off', 'pca'}


def test_exact_k_matrix_preview_draws_reference_and_blind_fit(app):
    import numpy as np
    from split_pool_encoding.datasets import generate
    window = app.rgb_image_view._split_pool_window
    window.data, window.synthetic_truth = generate('exact_k_4', bits=3, positions=8)
    order = np.arange(len(window.data))[::-1]
    prediction = window.data[order] * .9
    result = dict(model='on_off', prediction=prediction, assignment=order)
    window.figure.clear()
    window.draw_matrix_example(result, 300, 42)
    window.figure.canvas.draw()
    panels = [ax for ax in window.figure.axes if ax.images]
    assert len(panels) == 4
    assert 'Known recipe' in panels[1].get_title()
    np.testing.assert_allclose(panels[1].images[0].get_array(), window.data, atol=1e-15)
    np.testing.assert_allclose(panels[2].images[0].get_array(), window.data*.9)
    assert 'blind search' in window.summary.get()
