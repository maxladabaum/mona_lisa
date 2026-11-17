from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from split_pool_plot_canvas import StablePlotCanvasMixin


class CountingCanvas:
    def __init__(self):
        self.figure = Figure()
        self.resize_count = 0
    def resize(self, event):
        self.resize_count += 1


class ScrollCanvas(StablePlotCanvasMixin, CountingCanvas):
    pass


def test_scrolling_does_not_trigger_plot_resize_work():
    canvas = ScrollCanvas()
    for position in range(1000):
        canvas.resize(SimpleNamespace(width=1000, height=800, x=0, y=-position))
    assert canvas.resize_count == 1
    canvas.resize(SimpleNamespace(width=900, height=800))
    assert canvas.resize_count == 2
    canvas.figure.set_dpi(200)
    canvas.resize(SimpleNamespace(width=900, height=800))
    assert canvas.resize_count == 3


class StageCanvas(StablePlotCanvasMixin, FigureCanvasAgg):
    reuse_layout = True


def test_stage_updates_reuse_layout_but_resize_and_new_axes_recompute():
    figure = Figure(layout='constrained'); canvas = StageCanvas(figure)
    ax = figure.subplots(); artist = ax.imshow([[0, 1], [1, 0]])
    engine = figure.get_layout_engine()
    original = engine.execute
    engine.execute = Mock(wraps=original)
    canvas.draw()
    for stage in range(4):
        artist.set_data([[stage/4, 1], [0, stage/4]])
        ax.set_title(f'Stage {stage}')
        canvas.draw()
    assert engine.execute.call_count == 1
    figure.set_size_inches(8, 6); canvas.draw()
    assert engine.execute.call_count == 2
    figure.clear(); figure.subplots().plot([0, 1]); canvas.draw()
    assert engine.execute.call_count == 3
    canvas.reuse_layout = False; canvas.draw()
    assert engine.execute.call_count == 4


def test_layout_engine_is_restored_if_rendering_fails(monkeypatch):
    figure = Figure(layout='constrained'); canvas = StageCanvas(figure)
    figure.subplots().plot([0, 1]); canvas.draw()
    engine = figure.get_layout_engine()
    monkeypatch.setattr(FigureCanvasAgg, 'draw', Mock(side_effect=RuntimeError('draw failed')))
    with pytest.raises(RuntimeError):
        canvas.draw()
    assert figure.get_layout_engine() is engine
