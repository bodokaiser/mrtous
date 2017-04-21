import torch
import numpy as np

from skimage.exposure import equalize_hist
from skimage.filters import threshold_li

class Clip:

    def __init__(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, x):
        return np.clip(x, self.vmin, self.vmax) - self.vmin


class ToTensor:

    def __call__(self, x):
        return torch.from_numpy(x.astype(np.float32)).unsqueeze(0)


class HistNormalize:

    def __init__(self, nbins=256, mask=True):
        self.mask = mask
        self.nbins = nbins

    def __call__(self, x):
        try:
            threshold = max(0, threshold_li(x))
        except ValueError:
            return np.zeros_like(x)
        mask = x > threshold if self.mask else np.ones_like(x)

        return np.multiply(equalize_hist(x, self.nbins, mask), mask)