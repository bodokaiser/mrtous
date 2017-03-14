import h5py
import numpy as np
import os

import skimage
import skimage.io

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from mrtous.transform import Normalize, CenterCrop, ExpandDim

class Minc2z(Dataset):

    def __init__(self, filename):
        self.hdf = h5py.File(filename, 'r')

    @property
    def volume(self):
        return self.hdf['minc-2.0/image/0/image']

    @property
    def vrange(self):
        return self.volume.attrs['valid_range']

    def __getitem__(self, index):
        return self.volume[index]

    def __len__(self):
        return self.volume.shape[0]

class Minc2y(Minc2z):

    def __getitem__(self, index):
        return self.volume[:, index]

    def __len__(self):
        return self.volume.shape[1]

class Minc2x(Minc2z):

    def __getitem__(self, index):
        return self.volume[:, :, index]

    def __len__(self):
        return self.volume.shape[2]

class MnibiteNative(Dataset):

    def __init__(self, mr_filename, us_filename,
        input_transform=None, target_transform=None):
        self.mr = Minc2z(mr_filename)
        self.us = Minc2z(us_filename)
        assert len(self.mr) == len(self.us)

        if input_transform is None:
            input_transform = Compose([
                Normalize(self.mr.vrange),
                CenterCrop(300),
                ExpandDim(2),
                ToTensor(),
            ])
        if target_transform is None:
            target_transform = Compose([
                Normalize(self.us.vrange),
                CenterCrop(300),
                ExpandDim(2),
                ToTensor(),
            ])
        self.input_transform = input_transform
        self.target_transform = target_transform

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
            if fname.endswith('mr.tif'):
                self.mr_fnames.append(fname)
            if fname.endswith('us.tif'):
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

        return mr.astype(np.float32), us.astype(np.float32)

    def __len__(self):
        return len(self.mr_fnames)