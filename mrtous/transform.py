import numpy as np

from skimage.filters import threshold_otsu

class RegionCrop(object):

    def __call__(self, mr: np.ndarray, us: np.ndarray):
        if np.any(mr) and np.any(us):
            mask = us > threshold_otsu(us)

            x = np.where(np.any(mask, 0))[0][[0, -1]]
            y = np.where(np.any(mask, 1))[0][[0, -1]]

            if np.abs(np.diff(x)[0]) < 10 or np.abs(np.diff(y)[0]) < 10:
                # "mark" samples which are too small to be filtered
                return np.zeros_like(us), np.zeros_like(mr)

            mr = mr[y[0]:y[1], x[0]:x[1]]
            us = us[y[0]:y[1], x[0]:x[1]]

        return mr, us