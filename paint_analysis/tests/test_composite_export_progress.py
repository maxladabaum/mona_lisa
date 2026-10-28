"""Exercise GUI export scheduling without requiring a Tk display."""
import ast
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

import pytest


def window_method(name, namespace):
    path = Path(__file__).parents[1] / 'origami_composite_ui.py'
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('fail', [False, True])
def test_export_returns_while_worker_runs_and_reports_terminal_event(fail):
    entered, release = threading.Event(), threading.Event()
    workers, statuses = [], []
    source, results = object(), object()
    owner = SimpleNamespace(origami_source_points_nm=source, origami_multi_template_results=results,
                            origami_identification_baseline=None)
    window = SimpleNamespace(busy=False, result={'rgb': 'snapshot'}, result_inputs=(results, source),
                             owner=owner, data_owner=owner, events=queue.Queue(), _closed=False,
                             status=SimpleNamespace(set=statuses.append))

    def export(result, path, progress):
        assert result is window.result
        entered.set()
        assert release.wait(2)
        progress('Saving report (4/4)…')
        if fail:
            raise ValueError('disk full')

    def thread(**kwargs):
        worker = threading.Thread(**kwargs)
        workers.append(worker)
        return worker

    method = window_method('export', dict(
        threading=SimpleNamespace(Thread=thread), export_composite=export,
        filedialog=SimpleNamespace(asksaveasfilename=lambda **kwargs: 'test.png'),
        messagebox=SimpleNamespace(showerror=lambda *args, **kwargs: pytest.fail(str(args)))))
    try:
        method(window)
        assert entered.wait(2)
        assert window.busy
        assert window.events.empty()  # export returned while its worker is blocked
    finally:
        release.set()
        for worker in workers:
            worker.join(2)
    assert window.events.get_nowait() == ('progress', 'Saving report (4/4)…')
    terminal = window.events.get_nowait()
    assert terminal == (('export_error', 'disk full') if fail else ('exported', None))
    window.events.put(terminal)
    window.after = lambda *args: 'timer'
    window.poll = lambda: None
    errors = []
    poll = window_method('poll', dict(queue=queue, record_diagnostic=lambda *args, **kwargs: None,
                                    messagebox=SimpleNamespace(showerror=lambda *args, **kwargs: errors.append(args))))
    poll(window)
    assert not window.busy
    assert ('Export failed' if fail else 'Saved PNG') in statuses[-1]
    assert bool(errors) == fail
