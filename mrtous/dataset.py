import os
import h5py
import numpy as np

from torch.utils.data import Dataset

class MINC2(Dataset):

    def __init__(self, filename):
        with h5py.File(filename, 'r') as f:
            self.volume = f['minc-2.0/image/0/image']
            self.vrange = f['minc-2.0/image/0/image'].attrs['valid_range']
            self.xlength = f['minc-2.0/dimensions/xspace'].attrs['length']
            self.ylength = f['minc-2.0/dimensions/yspace'].attrs['length']
            self.zlength = f['minc-2.0/dimensions/zspace'].attrs['length']
            self.volume = np.array(self.volume, np.float32)
            self.volume -= np.min(self.vrange)
            self.volume /= np.sum(np.abs(self.vrange))

    def __len__(self):
        return self.xlength + self.ylength + self.zlength

    def __getitem__(self, index):
        if index < self.zlength:
            return self.volume[index]
        if index < self.ylength + self.zlength:
            return np.flipud(self.volume[:, index-self.zlength])
        if index < self.xlength + self.ylength + self.zlength:
            return np.flipud(self.volume[:, :, index-self.ylength-self.zlength])

        raise IndexError('invalid index')

class MNIBITE(Dataset):

    def __init__(self, dir, id, transform=None):
        self.mr = MINC2(os.path.join(dir, f'{id:02d}_mr.mnc'))
        self.us = MINC2(os.path.join(dir, f'{id:02d}_us.mnc'))
        self.transform = transform

    def __getitem__(self, index):
        mr = self.mr[index]
        us = self.us[index]
        if self.transform is not None:
            mr, us = self.transform(mr, us)
        return mr, us

    def __len__(self):
        return len(self.mr)