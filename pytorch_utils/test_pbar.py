import time
from progressbar import ProgressBar, FormatLabel, UnknownLength
import progressbar as pbar

def pbar_loss(n_epochs, loss_getter):
    
    widget_loss = FormatLabel('Loss: %(value)s')
    widget_loss.mapping = {**widget_loss.mapping,
                           'value': ('value', loss_getter)}

    widgets = [
        '[', widget_loss, ']',
        ' [', pbar.Timer(), '] ',
        pbar.Bar(),
        ' (', pbar.ETA(), ') ',
    ]

    return ProgressBar(widgets=widgets, max_value=UnknownLength)
