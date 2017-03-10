import os
import sys
import argparse
import numpy as np
import skimage as sk

from mrtous import dataset
from skimage import io, util, exposure

def image_to_patches(image, size):
    stride = int(np.ceil(.3*size))
    patches = util.view_as_windows(image, size, stride)
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
                mr_patch = sk.img_as_uint(
                    exposure.rescale_intensity(mr_patches[index],
                        out_range='float'))
                us_patch = sk.img_as_uint(
                    exposure.rescale_intensity(us_patches[index],
                        out_range='float'))

                io.imsave(os.path.join(targetdir, f'{index}_{axis}_mr.png'),
                    mr_patch, plugin='freeimage')
                io.imsave(os.path.join(targetdir, f'{index}_{axis}_us.png'),
                    us_patch, plugin='freeimage')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, required=True)
    parser.add_argument('--datadir', type=str, nargs='?', default='mnibite')
    parser.add_argument('--targetdir', type=str, nargs='?', default='.')
    parser.add_argument('--targetsize', type=int, nargs='?', default=25)
    parser.add_argument('--threshold', type=float, nargs='?', default=.5)

    main(parser.parse_args())