import os
import h5py
import numpy as np

from torch.utils.data import Dataset, DataLoader

class MINC2(Dataset):
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

def format_path(directory, filename, params):
    return os.path.join(directory, filename.format(params))

class MNIBITE(Dataset):
    def __init__(self, directory, id, transform=None):
        self.mr = MINC2(format_path(directory, '{}_mr.mnc', id))
        self.us = MINC2(format_path(directory, '{}_us.mnc', id))
        self.transform = transform

    def __getitem__(self, index):
        mr = self.mr[index]
        us = self.us[index]
        if self.transform is not None:
            mr, us = self.transform(mr, us)
        return mr, us

    def __len__(self):
        return len(self.mr)