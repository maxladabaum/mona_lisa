"""Standalone session entry must not require an analysis or source dataset."""
import ast
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

import pytest

from test_composite_export_progress import window_method


def main_method(name, namespace):
    path = Path(__file__).parents[1] / 'paint_analysis_gui.py'
    method = next(node for node in ast.walk(ast.parse(path.read_text()))
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('selected', ['saved.paintrecon', ''])
def test_main_entry_opens_session_without_analysis(selected):
    calls = []
    app = SimpleNamespace(_file_dialog_initial_dir=lambda: Path('/tmp'),
                          _remember_file_dialog_dir=lambda path: calls.append(('remember', path)))
    def open_window():
        calls.append(('open', None))
        app.origami_composite_window = SimpleNamespace(load_session=lambda path: calls.append(('load', path)))
    app._open_origami_composite = open_window
    method = main_method('open_reconstruction_session', dict(
        Path=Path, filedialog=SimpleNamespace(askopenfilename=lambda **kwargs: selected),
        messagebox=SimpleNamespace(showinfo=lambda *args: pytest.fail(str(args)))))
    method(app)
    assert calls == ([('remember', Path(selected)), ('open', None), ('load', selected)] if selected else [])


def test_main_entry_does_not_interrupt_active_save():
    notices = []
    app = SimpleNamespace(origami_composite_window=SimpleNamespace(winfo_exists=lambda: True, busy=True))
    method = main_method('open_reconstruction_session', dict(
        messagebox=SimpleNamespace(showinfo=lambda *args: notices.append(args))))
    method(app)
    assert len(notices) == 1


@pytest.mark.parametrize('fail', [False, True])
def test_explicit_session_path_loads_in_worker_without_second_dialog(fail):
    workers, loaded_paths = [], []
    entered, release = threading.Event(), threading.Event()
    owner = object()
    window = SimpleNamespace(busy=False, _closed=False, events=queue.Queue(), status=SimpleNamespace(set=lambda text: None))
    window.show_session_dialog = lambda **kwargs: None
    def load(path, progress):
        loaded_paths.append(path)
        entered.set()
        assert release.wait(2)
        if fail:
            raise ValueError('invalid session')
        return owner
    def thread(**kwargs):
        worker = threading.Thread(**kwargs)
        workers.append(worker)
        return worker
    method = window_method('load_session', dict(load_reconstruction_session=load,
                                               threading=SimpleNamespace(Thread=thread)))
    try:
        method(window, 'saved.paintrecon')
        assert entered.wait(2)
        assert window.busy
        assert window.events.empty()
    finally:
        release.set()
        for worker in workers:
            worker.join(2)
    assert loaded_paths == ['saved.paintrecon']
    assert window.events.get_nowait() == (('session_error', 'invalid session') if fail else ('session_loaded', owner))
