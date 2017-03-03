import os
import h5py
import numpy as np

from torch.utils.data import Dataset

class MINC2(Dataset):

    def __init__(self, filename: str):
        with h5py.File(filename, 'r') as f:
            self.volume = f['minc-2.0/image/0/image']
            self.vrange = f['minc-2.0/image/0/image'].attrs['valid_range']
            self.xlength = f['minc-2.0/dimensions/xspace'].attrs['length']
            self.ylength = f['minc-2.0/dimensions/yspace'].attrs['length']
            self.zlength = f['minc-2.0/dimensions/zspace'].attrs['length']
            self.volume = np.array(self.volume, np.float32)
            self.volume -= np.min(self.vrange)
            self.volume /= np.sum(np.abs(self.vrange))

    def __getitem__(self, index: int):
        return self.volume[index]

    def __len__(self) -> int:
        return self.zlength

    def __iter__(self):
        return self.volume.__iter__()

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