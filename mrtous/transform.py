import random
import numpy as np
import scipy as sp

import skimage.filters
import skimage.transform

class Normalize(object):

    def __init__(self, vrange):
        self.vrange = vrange

    def __call__(self, image):
        image -= np.min(self.vrange)
        image /= np.sum(np.abs(self.vrange))
        return image

class CenterCrop(object):

    def __init__(self, size):
        self.width = size
        self.height = size

    def __call__(self, image):
        xlen, ylen = image.shape

        xoff = xlen // 2 - self.width // 2
        yoff = ylen // 2 - self.height // 2

        return image[xoff:xoff+self.width, yoff:yoff+self.height]

class BinaryMask(object):

    def __call__(self, image):
        if np.any(image):
            return image > skimage.filters.threshold_otsu(image)
        return np.zeros_like(image).astype(bool)

class RandomFlip(object):

    def __call__(self, image):
        if random.random() > .5:
            image = np.flipud(image)
        if random.random() > .5:
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