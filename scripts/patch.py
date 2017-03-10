import os
import sys
import math
import argparse
import numpy as np

import skimage
import skimage.io
import skimage.util

from mrtous import dataset

def image_to_patches(image, size):
    patches = skimage.util.view_as_windows(image, size, int(math.ceil(.3*size)))
    return np.reshape(patches, [-1, size, size])

def main(args):
    mnibites = [
        dataset.MNIBITENative(args.datadir, args.dataset, axis='x'),
        dataset.MNIBITENative(args.datadir, args.dataset, axis='y'),
        dataset.MNIBITENative(args.datadir, args.dataset, axis='z'),
    ]

    targetdir = os.path.join(args.targetdir, f'{args.dataset:02d}')
    targetsum = args.threshold*args.targetsize**2

    os.makedirs(targetdir, exist_ok=True)

    for mnibite in mnibites:
        axis = mnibite.axis

        for _, (mr_image, us_image) in enumerate(mnibite):
            mr_patches = image_to_patches(mr_image, args.targetsize)
            us_patches = image_to_patches(us_image, args.targetsize)

            indices, = np.where(us_patches.sum((1, 2)) > targetsum)

            for index in indices:
                skimage.io.imsave(os.path.join(targetdir,
                    f'{index}_{axis}_mr.tif'), mr_patches[index])
                skimage.io.imsave(os.path.join(targetdir,
                    f'{index}_{axis}_us.tif'), us_patches[index])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, required=True)
    parser.add_argument('--datadir', type=str, nargs='?', default='mnibite')
    parser.add_argument('--targetdir', type=str, nargs='?', default='.')
    parser.add_argument('--targetsize', type=int, nargs='?', default=25)
    parser.add_argument('--threshold', type=float, nargs='?', default=.5)

    main(parser.parse_args())