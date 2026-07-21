from pyqtgraph import PlotWidget
import pyqtgraph as pg
from PySide6.QtWidgets import QSizePolicy

pg.ViewBox.suggestPadding = lambda *_: 0.02


class PGWidget(PlotWidget):
    def __init__(self, parent=None):
        super().__init__(
            parent=parent,
            axisItems={"top": pg.AxisItem(orientation="top", showValues=False),
                       "right": pg.AxisItem(orientation="right", showValues=False)})
        # PlotWidget's size hint must never establish a panel resize floor. The
        # containing layout owns its physical pixel size in both dimensions.
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        self.x = []
        self.y = []

        self.scale_x = 1

        self._replace = False

        pen = pg.mkPen(color=(0, 0, 255), width=1)
        self._pen = pen
        self._data_line = self.plot(self.x, self.y, pen=pen)

        self._cache = list()

    def reset(self, default=0, length=0):
        self.x = []
        self.y = []

        for idx in range(length):
            self.x.append(idx)
            self.y.append(default)

        self._data_line.setData(self.x, self.y)

    def replace(self, values):
        self.x = []
        self.y = []

        self.x.extend(range(len(values)))
        self.y.extend(values)

        self._data_line.setData(self.x, self.y)

    def update_plot(self, values):
        add_len = len(values)

        if add_len >= 500:
            self.x = []
            self.y = []
            values = values[add_len - 500:]
            add_len = len(values)
        elif len(self.x) + add_len > 500:
            self.x = self.x[add_len:]
            self.y = self.y[add_len:]

        first_x = -1 if len(self.x) == 0 else int(self.x[-1] * self.scale_x)

        x_list = list(range(first_x + 1, first_x + add_len + 1))

        if self.scale_x != 1:
            x_list_2 = [float(x) / self.scale_x for x in x_list]
        else:
            x_list_2 = x_list

        self.x.extend(x_list_2)
        self.y.extend(values)

        self._data_line.setData(self.x, self.y)

    def cache_data(self, values):
        self._cache += values
        self._replace = False

    def replace_cache(self, values):
        self._cache = values
        self._replace = True

    def use_cache(self):
        if len(self._cache) > 0:
            values = self._cache
            self._cache = list()
            if self._replace:
                self.replace(values)
            else:
                self.update_plot(values)
