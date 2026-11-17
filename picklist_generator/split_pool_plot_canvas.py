"""Avoid expensive plot resizing when Tk only moves a scrolled widget."""


class StablePlotCanvasMixin:
    reuse_layout = False

    def resize(self, event):
        # Tk Configure includes position changes, not just actual resizes.
        geometry = (event.width, event.height, self.figure.dpi)
        if geometry == getattr(self, '_last_plot_geometry', None):
            return
        if event.width <= 0 or event.height <= 0:
            return
        super().resize(event)
        self._last_plot_geometry = geometry

    def draw(self):
        # Construction stages change image values and fixed-format labels.
        # Keep their first computed layout until the axes or canvas size change.
        signature = (tuple(self.figure.axes), tuple(self.figure.get_size_inches()), self.figure.dpi)
        engine = self.figure.get_layout_engine()
        reuse = (self.reuse_layout and engine is not None
                 and signature == getattr(self, '_last_plot_layout', None))
        if reuse:
            self.figure.set_layout_engine('none')
        try:
            super().draw()
            self._last_plot_layout = signature
        finally:
            if reuse:
                self.figure.set_layout_engine(engine)


def plot_canvas(figure, master):
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    class StablePlotCanvas(StablePlotCanvasMixin, FigureCanvasTkAgg):
        pass

    return StablePlotCanvas(figure, master=master)
