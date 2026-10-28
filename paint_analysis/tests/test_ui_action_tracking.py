from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk
from unittest.mock import Mock

import pytest
import paint_analysis_gui as gui


@pytest.fixture
def root():
    previous = tk._default_root
    window = tk.Tk()
    window.withdraw()
    yield window
    window.destroy()
    tk._default_root = previous


def test_tcl_combobox_popdown_is_logged_without_widget_object(root, monkeypatch):
    recorder = Mock()
    monkeypatch.setattr(gui, 'record_diagnostic', recorder)
    combo = ttk.Combobox(root, values=('Aligned density', 'Origami type counts'))
    popup = root.tk.call('ttk::combobox::PopdownWindow', str(combo))
    path = f'{popup}.f.l'
    assert root.tk.call('winfo', 'class', path) == 'Listbox'
    gui.PaintAnalysisApp._record_ui_action(root, SimpleNamespace(widget=path, type='ButtonPress'))
    recorder.assert_called_once_with('ui_action', widget=path, widget_class='Listbox',
                                     label='', input_event='ButtonPress')


def test_normal_buttons_still_record_label(root, monkeypatch):
    recorder = Mock()
    monkeypatch.setattr(gui, 'record_diagnostic', recorder)
    button = ttk.Button(root, text='Plot Only')
    gui.PaintAnalysisApp._record_ui_action(root, SimpleNamespace(widget=button, type='ButtonPress'))
    assert recorder.call_args.kwargs['label'] == 'Plot Only'


def test_destroyed_widgets_and_logging_errors_do_not_escape(root, monkeypatch):
    recorder = Mock(side_effect=RuntimeError('diagnostics unavailable'))
    monkeypatch.setattr(gui, 'record_diagnostic', recorder)
    gui.PaintAnalysisApp._record_ui_action(root, SimpleNamespace(widget='.destroyed', type='ButtonPress'))
    recorder.assert_not_called()
    button = ttk.Button(root, text='Plot Only')
    gui.PaintAnalysisApp._record_ui_action(root, SimpleNamespace(widget=button, type='ButtonPress'))
    recorder.assert_called_once()


def test_closed_composite_releases_source_and_queued_results():
    import queue
    import weakref
    import numpy as np
    from origami_composite_ui import CompositeWindow
    owner = SimpleNamespace()
    source = np.arange(100)
    reference = weakref.ref(source)
    events = queue.Queue()
    events.put(('result', source))
    window = SimpleNamespace(owner=owner, data_owner=owner, configuration=lambda: {}, result=source,
                             result_inputs=(source,), events=events, poll_after_id='poll',
                             after_cancel=Mock(), destroy=Mock())
    owner.origami_composite_window = window
    del source
    CompositeWindow.close(window)
    assert reference() is None
    assert owner.origami_composite_window is None
    assert window._closed
    assert events.empty()
