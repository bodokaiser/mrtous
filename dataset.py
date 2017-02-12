import os
import h5py
import numpy as np

from torch.utils import data

class MINC2(data.Dataset):
    def __init__(self, filename):
        self.minc = h5py.File(filename, 'r')
        self.volume = self.minc['minc-2.0/image/0/image']
        self.vrange = self.minc['minc-2.0/image/0/image'].attrs['valid_range']
        self.xlength = self.minc['minc-2.0/dimensions/xspace'].attrs['length']
        self.ylength = self.minc['minc-2.0/dimensions/yspace'].attrs['length']
        self.zlength = self.minc['minc-2.0/dimensions/zspace'].attrs['length']

    def __getitem__(self, index):
        slice = np.array(self.volume[index])
        return slice / np.sum(np.abs(self.vrange))

    def __len__(self):
        return self.zlength

class MNIBITE(data.Dataset):
    def __init__(self, directory, id, transform=None):
        self.mr_minc2 = MINC2(os.path.join(directory, '{}_mr.mnc'.format(id)))
        self.us_minc2 = MINC2(os.path.join(directory, '{}_us.mnc'.format(id)))
        self.transform = transform

    def __getitem__(self, index):
        mr = self.mr_minc2[index]
        us = self.us_minc2[index]
        if self.transform is not None:
            mr = self.transform(mr)
            us = self.transform(us)
        return mr, us

    def __len__(self):
        return len(self.mr_minc2)