import skimage
import skimage.io

from torch import Tensor
from torch.autograd import Variable

def imsave(filename, image):
    if image.is_cuda:
        image = image.cpu()
    if isinstance(image, Variable):
        image = image.data
    if isinstance(image, Tensor):
        image = image.numpy()

    skimage.io.imsave(filename, skimage.img_as_uint(image))
