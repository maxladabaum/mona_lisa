"""Save-dialog lifecycle and close protection without a Tk display."""
import ast
from pathlib import Path
import queue
from types import SimpleNamespace

import pytest

from test_composite_export_progress import window_method


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('loading', [False, True])
def test_modal_session_dialog_stays_until_terminal_event(fail, loading):
    calls, statuses, errors = [], [], []

    class Widget:
        def __init__(self, *args, **kwargs):
            self.options = kwargs
        def pack(self, **kwargs): pass
        def title(self, value): pass
        def transient(self, owner): pass
        def resizable(self, *args): pass
        def protocol(self, name, callback): self.close_callback = callback
        def grab_set(self): calls.append('grab')
        def focus_set(self): pass
        def start(self): calls.append('start')
        def stop(self): calls.append('stop')
        def lift(self): calls.append('lift')
        def bell(self): pass
        def grab_release(self): calls.append('release')
        def destroy(self): calls.append('destroy')

    window = SimpleNamespace(busy=True, events=queue.Queue(), status=SimpleNamespace(set=statuses.append),
                             after=lambda *args: 'timer', poll=lambda: None)
    guard = window_method('prevent_close_during_session_io', {})
    window.prevent_close_during_session_io = lambda: guard(window)
    finish = window_method('finish_session_dialog', {})
    window.finish_session_dialog = lambda: finish(window)
    show = window_method('show_session_dialog', dict(tk=SimpleNamespace(Toplevel=Widget),
                                                 ttk=SimpleNamespace(Label=Widget, Progressbar=Widget)))
    show(window, loading=loading)
    assert window.session_io_busy
    assert calls == ['start', 'grab']
    assert window.session_dialog.close_callback() is True
    close = window_method('close', {})
    close(window)  # Returns before cleanup; no data owner or Tk dependencies needed.
    assert window.session_io_busy
    assert 'destroy' not in calls
    poll = window_method('poll', dict(queue=queue, record_diagnostic=lambda *args, **kwargs: None,
                                     messagebox=SimpleNamespace(showerror=lambda *args, **kwargs: errors.append(args))))
    window.events.put(('progress', 'Writing source coordinates…'))
    poll(window)
    assert window.session_io_busy
    assert statuses[-1] == 'Writing source coordinates…'
    window.update_idletasks = lambda: None
    applied = []
    window.apply_configuration = applied.append
    loaded = SimpleNamespace(origami_composite_configuration={'saved': True})
    terminal = ('session_loaded', loaded) if loading else ('session_saved', 'image.paintrecon')
    window.events.put(('session_error', 'disk full') if fail else terminal)
    poll(window)
    assert not window.session_io_busy
    assert not window.busy
    assert calls[-3:] == ['stop', 'release', 'destroy']
    assert bool(errors) == fail
    assert guard(window) is False
    assert applied == ([{'saved': True}] if loading and not fail else [])


def test_main_app_close_is_blocked_during_session_save():
    path = Path(__file__).parents[1] / 'paint_analysis_gui.py'
    method = next(node for node in ast.walk(ast.parse(path.read_text()))
                  if isinstance(node, ast.FunctionDef) and node.name == '_on_close')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    calls = []
    composite = SimpleNamespace(prevent_close_during_session_io=lambda: calls.append('blocked') or True)
    app = SimpleNamespace(origami_composite_window=composite)
    namespace['_on_close'](app)
    assert calls == ['blocked']


def test_configuration_failure_dismisses_loading_dialog_and_reports_error():
    calls = []
    def apply(config):
        raise ValueError('invalid settings')
    window = SimpleNamespace(busy=True, events=queue.Queue(), status=SimpleNamespace(set=lambda value: calls.append(value)),
                             update_idletasks=lambda: None, apply_configuration=apply,
                             finish_session_dialog=lambda: calls.append('dismissed'),
                             after=lambda *args: 'timer', poll=lambda: None)
    window.events.put(('session_loaded', SimpleNamespace(origami_composite_configuration={})))
    poll = window_method('poll', dict(queue=queue, record_diagnostic=lambda *args, **kwargs: None,
                                     messagebox=SimpleNamespace(showerror=lambda *args, **kwargs: calls.append('error shown'))))
    poll(window)
    assert 'dismissed' in calls
    assert 'error shown' in calls
    assert not window.busy
    assert not window._bulk_mapping_update
