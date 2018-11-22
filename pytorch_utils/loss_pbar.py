import time
from progressbar import ProgressBar, FormatLabel, UnknownLength
from progressbar.widgets import WidgetBase
import progressbar as pbar


class LossWidget(WidgetBase):
    def __init__(self, get_value_fun):
        WidgetBase.__init__(self)
        self.get_value_fun = get_value_fun

    def __call__(self, progress, data, width=None):
        return self.get_value_fun()


def pbar_loss(n_epochs, loss_getter):

    widget_loss = LossWidget(loss_getter)

    widgets = [
        '[',
        widget_loss,
        ']',
        ' [',
        pbar.Timer(),
        '] ',
        pbar.Bar(),
        ' (',
        pbar.ETA(),
        ') ',
    ]

    return ProgressBar(widgets=widgets, max_value=n_epochs)
