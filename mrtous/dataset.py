import os
import h5py
import numpy as np
import skimage as sk

from skimage import io, util
from torch.utils.data import Dataset

def normalize(value, vrange):
    return (np.array(value, np.float32)-np.min(vrange)) / np.sum(np.abs(vrange))

class MINC2(Dataset):

    AXES = ['x', 'y', 'z']

    def __init__(self, filename, axis='z'):
        if not axis in self.AXES:
            raise ValueError('axis must be "x", "y" or "z"')
        self.axis = axis

        with h5py.File(filename, 'r') as f:
            self.volume = f['minc-2.0/image/0/image']
            self.vrange = f['minc-2.0/image/0/image'].attrs['valid_range']
            self.length = f['minc-2.0/dimensions/'+axis+'space'].attrs['length']
            self.volume = normalize(self.volume, self.vrange)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.axis == self.AXES[2]:
            return self.volume[index]
        if self.axis == self.AXES[1]:
            return np.flipud(self.volume[:, index])
        if self.axis == self.AXES[0]:
            return np.flipud(self.volume[:, :, index])

class MNIBITENative(Dataset):

    def __init__(self, root, id, transform=None, axis='z'):
        self.mr = MINC2(os.path.join(root, f'{id:02d}_mr.mnc'), axis)
        self.us = MINC2(os.path.join(root, f'{id:02d}_us.mnc'), axis)
        self.transform = transform

    def __getitem__(self, index):
        mr = self.mr[index]
        us = self.us[index]
        if self.transform is not None:
            mr, us = self.transform(mr, us)
        return mr, us

    def __len__(self):
        return len(self.mr)

class MNIBITEFolder(Dataset):

    def __init__(self, root, transform=None, target_transform=None):
        if type(root) is str:
            root = [root]

        self.mr_fnames = []
        self.us_fnames = []

        for root in root:
            for fname in os.listdir(root):
                fname = os.path.join(root, fname)
                if fname.endswith('_mr.png'):
                    self.mr_fnames.append(fname)
                if fname.endswith('_us.png'):
                    self.us_fnames.append(fname)

        assert(len(self.mr_fnames) == len(self.us_fnames))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        mr = sk.img_as_float(io.imread(self.mr_fnames[index]))
        us = sk.img_as_float(io.imread(self.us_fnames[index]))

        if self.transform is not None:
            mr = self.transform(mr)
        if self.target_transform is not None:
            us = self.target_transform(us)

        return mr.astype(np.float32), us.astype(np.float32)

    def __len__(self):
        return len(self.mr_fnames)