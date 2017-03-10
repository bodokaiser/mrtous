import os
import h5py
import numpy as np

import skimage
import skimage.io

from mrtous import transform
from torch.utils import data
from torchvision import transforms

class MINC2(data.Dataset):

    AXES = ['x', 'y', 'z']

    def __init__(self, filename, axis='z'):
        if not axis in self.AXES:
            raise ValueError('axis must be "x", "y" or "z"')
        self.axis = axis

        with h5py.File(filename, 'r') as f:
            self.volume = f['minc-2.0/image/0/image']
            self.vrange = f['minc-2.0/image/0/image'].attrs['valid_range']
            self.length = f['minc-2.0/dimensions/'+axis+'space'].attrs['length']
            self.volume = np.array(self.volume, np.float64)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.axis == self.AXES[2]:
            return self.volume[index]
        if self.axis == self.AXES[1]:
            return self.volume[:, index]
        if self.axis == self.AXES[0]:
            return self.volume[:, :, index]

class MNIBITENative(data.Dataset):

    def __init__(self, root, id, axis='z'):
        self.mr = MINC2(os.path.join(root, f'{id:02d}_mr.mnc'), axis)
        self.us = MINC2(os.path.join(root, f'{id:02d}_us.mnc'), axis)
        assert len(self.mr) == len(self.us)

        self.axis = axis

        self.input_transform = transforms.Compose([
            transform.Normalize(self.mr.vrange),
            transform.CenterCrop(300),
        ])
        self.target_transform = transforms.Compose([
            transform.Normalize(self.us.vrange),
            transform.CenterCrop(300),
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

class MNIBITEFolder(data.Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        if type(root) is str:
            root = [root]

        self.mr_fnames = []
        self.us_fnames = []

        for root in root:
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