import argparse
import math
import numpy as np
import os

import skimage
import skimage.util

from mrtous import util
from mrtous.dataset import MnibiteNative

def image_to_patches(image, size):
    patches = skimage.util.view_as_windows(image, size, int(math.ceil(.3*size)))
    return np.reshape(patches, [-1, size, size])

def main(args):
    dataset = MnibiteNative(
        os.path.join(args.datadir, f'{args.dataset:02d}_mr.mnc'),
        os.path.join(args.datadir, f'{args.dataset:02d}_us.mnc'))

    targetdir = os.path.join(args.targetdir, f'{args.dataset:02d}')
    targetsum = args.threshold*args.targetsize**2

    os.makedirs(targetdir, exist_ok=True)

    for i in range(len(dataset)):
        mr_image, us_image = dataset[i]

        mr_patches = image_to_patches(mr_image[0].numpy(), args.targetsize)
        us_patches = image_to_patches(us_image[0].numpy(), args.targetsize)

        indices, = np.where(us_patches.sum() > targetsum)

        for j in indices:
            util.save_image(os.path.join(targetdir,
                f'{i+j:04d}_mr.tif'), mr_patches[j])
            util.save_image(os.path.join(targetdir,
                f'{i+j:04d}_us.tif'), us_patches[j])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, required=True)
    parser.add_argument('--datadir', type=str, nargs='?', default='mnibite')
    parser.add_argument('--targetdir', type=str, nargs='?', default='.')
    parser.add_argument('--targetsize', type=int, nargs='?', default=25)
    parser.add_argument('--threshold', type=float, nargs='?', default=.2)

    main(parser.parse_args())