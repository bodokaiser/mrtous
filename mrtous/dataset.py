import os
import h5py
import numpy as np

from torch.utils.data import Dataset

class MINC2(Dataset):

    def __init__(self, filename: str):
        self.minc = h5py.File(filename, 'r')
        self.volume = self.minc['minc-2.0/image/0/image']
        self.vrange = self.minc['minc-2.0/image/0/image'].attrs['valid_range']
        self.xlength = self.minc['minc-2.0/dimensions/xspace'].attrs['length']
        self.ylength = self.minc['minc-2.0/dimensions/yspace'].attrs['length']
        self.zlength = self.minc['minc-2.0/dimensions/zspace'].attrs['length']

    def __getitem__(self, index: int):
        slice = np.array(self.volume[index]) / np.sum(np.abs(self.vrange))
        return slice.astype(np.float32)

    def __len__(self) -> int:
        return self.zlength

class MNIBITE(Dataset):

    def __init__(self, dir: str, id: int, transform=None):
        self.mr = MINC2(os.path.join(dir, f'{id:02d}_mr.mnc'))
        self.us = MINC2(os.path.join(dir, f'{id:02d}_us.mnc'))
        self.transform = transform

    def __getitem__(self, index: int):
        mr = self.mr[index]
        us = self.us[index]
        if self.transform is not None:
            mr, us = self.transform(mr, us)
        return mr, us

    def __len__(self) -> int:
        return len(self.mr)