import numpy as np

from skimage.filters import threshold_otsu

class ROICrop(object):
    def __call__(self, mr, us):
        if np.any(mr) and np.any(us):
            mask = us > threshold_otsu(us)
            xrange = np.where(np.any(mask, 0))[0][[0, -1]]
            yrange = np.where(np.any(mask, 1))[0][[0, -1]]
            mr = mr[xrange[0]:xrange[1], yrange[0]:yrange[1]]
            us = us[xrange[0]:xrange[1], yrange[0]:yrange[1]]
        return mr, us