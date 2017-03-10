import random
import numpy as np
import scipy as sp

from skimage import filters, transform

class RegionCrop(object):

    def __call__(self, mr, us):
        if np.any(mr) and np.any(us):
            mask = us > filters.threshold_otsu(us)

            x = np.where(np.any(mask, 0))[0][[0, -1]]
            y = np.where(np.any(mask, 1))[0][[0, -1]]

            if np.abs(np.diff(x)[0]) < 10 or np.abs(np.diff(y)[0]) < 10:
                # "mark" samples which are too small to be filtered
                return np.zeros_like(us), np.zeros_like(mr)

            mr = mr[y[0]:y[1], x[0]:x[1]]
            us = us[y[0]:y[1], x[0]:x[1]]

        return mr, us

class RandomFlip(object):

    def __call__(self, image):
        if random.random() > .5:
            image = np.flipud(image)
        if random.random() > .5
            image = np.fliplr(image)
        return image

class RandomRotate(object):

    def __call__(self, image):
        return sp.ndimage.rotate(image, np.random.choice(90), reshape=False)

class RandomZoom(object):

    def __call__(self, image):
        zoom = random.uniform(1.0, 1.5)

        height, width = image.shape
        zheight = int(np.round(zoom*height))
        zwidth = int(np.round(zoom*width))

        top = (zheight - height) // 2
        left = (zwidth - width) // 2

        image = image[top:top+zheight, left:left+zwidth]
        image = sp.ndimage.zoom(image, zoom)

        trim_top = (image.shape[0] - height) // 2
        trim_left = (image.shape[1] - width) // 2

        return image[trim_top:trim_top+height, trim_left:trim_left+width]