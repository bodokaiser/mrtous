import h5py
import numpy as np
import os

import skimage
import skimage.io

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from mrtous.transform import Normalize, CenterCrop, ExpandDim

class Concat(Dataset):

    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)

    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                return self.datasets[i][index]
        raise IndexError(f'{index} exceeds {self.length}')

    def __len__(self):
        return self.length

class Minc2(Dataset):

    def __init__(self, filename):
        self.hdf = h5py.File(filename, 'r')

    @property
    def volume(self):
        return self.hdf['minc-2.0/image/0/image']

    @property
    def vrange(self):
        return self.volume.attrs['valid_range']

    @property
    def xlength(self):
        return self.volume.shape[2]

    @property
    def ylength(self):
        return self.volume.shape[1]

    @property
    def zlength(self):
        return self.volume.shape[0]

    def __getitem__(self, index):
        if index < self.zlength:
            return self.volume[index]
        if index < self.zlength + self.ylength:
            return self.volume[:, index - self.zlength]
        if index < self.zlength + self.ylength + self.xlength:
            return self.volume[:, :, index - self.zlength - self.ylength]
        raise IndexError(f'index {index} exceeds {len(self)}')

    def __len__(self):
        return sum(self.volume.shape)

class MnibiteNative(Dataset):

    def __init__(self, mr_filename, us_filename,
        input_transform=None, target_transform=None):
        self.mr = Minc2(mr_filename)
        self.us = Minc2(us_filename)
        assert len(self.mr) == len(self.us)

        if input_transform is None:
            input_transform = Compose([
                Normalize(self.mr.vrange),
                CenterCrop(320),
                ExpandDim(2),
                ToTensor(),
            ])
        if target_transform is None:
            target_transform = Compose([
                Normalize(self.us.vrange),
                CenterCrop(320),
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