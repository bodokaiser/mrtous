import h5py
import numpy as np
import os

import skimage
import skimage.io

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from mrtous.transform import Normalize, CenterCrop

class Minc2Z(Dataset):

    def __init__(self, filename):
        self.hdf = h5py.File(filename, 'r')

    @property
    def volume(self):
        return self.hdf['minc-2.0/image/0/image']

    @property
    def vrange(self):
        return self.volume.attrs['valid_range']

    def __getitem__(self, index):
        return self.volume[index].astype(np.float64)

    def __len__(self):
        return self.volume.shape[0]

class Minc2Y(Minc2Z):

    def __getitem__(self, index):
        return self.volume[:, index].astype(np.float64)

    def __len__(self):
        return self.volume.shape[1]

class Minc2X(Minc2Z):

    def __getitem__(self, index):
        return self.volume[:, :, index].astype(np.float64)

    def __len__(self):
        return self.volume.shape[2]

class MnibiteNative(Dataset):

    def __init__(self, root, id):
        self.mr = Minc2Z(os.path.join(root, f'{id:02d}_mr.mnc'))
        self.us = Minc2Z(os.path.join(root, f'{id:02d}_us.mnc'))
        assert len(self.mr) == len(self.us)

        self.input_transform = Compose([
            Normalize(self.mr.vrange),
            CenterCrop(300),
        ])
        self.target_transform = Compose([
            Normalize(self.us.vrange),
            CenterCrop(300),
        ])

    def __getitem__(self, index):
        mr, us = self.mr[index], self.us[index]

        if self.input_transform is not None:
            mr = self.input_transform(mr)
        if self.target_transform is not None:
            us = self.target_transform(us)

        return mr, us

    def __len__(self):
        return len(self.mr)

class MnibiteFolder(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.mr_fnames = []
        self.us_fnames = []

        for fname in os.listdir(root):
            fname = os.path.join(root, fname)
            if fname.endswith('_mr.tif'):
                self.mr_fnames.append(fname)
            if fname.endswith('_us.tif'):
                self.us_fnames.append(fname)
        assert(len(self.mr_fnames) == len(self.us_fnames))

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        mr = skimage.io.imread(self.mr_fnames[index])
        us = skimage.io.imread(self.us_fnames[index])

        if self.input_transform is not None:
            mr = self.input_transform(mr)
        if self.target_transform is not None:
            us = self.target_transform(us)

        return mr.astype(np.float64), us.astype(np.float64)

    def __len__(self):
        return len(self.mr_fnames)