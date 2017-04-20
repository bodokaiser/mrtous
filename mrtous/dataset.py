import h5py
import numpy as np

from torch.utils.data import Dataset

class MINC2(Dataset):

    def __init__(self, filename, transform=None):
        with h5py.File(filename, 'r') as f:
            self.volume = f['minc-2.0/image/0/image'][:]
        self.transform = transform

    def __getitem__(self, index):
        slice = self.volume[index]
        if self.transform is not None:
            slice = self.transform(slice)
        return slice

    def __len__(self):
        return len(self.volume)


class MNIBITE(Dataset):

    def __init__(self, mr_filename, us_filename,
        input_transform=None, target_transform=None):
        self.mr = MINC2(mr_filename, input_transform)
        self.us = MINC2(us_filename, target_transform)
        assert len(self.mr) == len(self.us), 'mr, us have different length'
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        return self.mr[index], self.us[index]

    def __len__(self):
        return len(self.mr)