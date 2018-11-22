from pytorch_utils.loss_pbar import LossWidget, pbar_loss
import time

n_epochs = 200


class DummyLossGetter():
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1
        return self.count


with pbar_loss(n_epochs, DummyLossGetter()) as bar:
    for i in range(n_epochs):
        time.sleep(0.1)
        bar.update(i)
