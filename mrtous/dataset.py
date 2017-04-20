import h5py
import os

from torch.utils.data import Dataset

class MINC2(Dataset):

    def __init__(self, filename, transform=None):
        self.filename = filename
        self.transform = transform

    def __getitem__(self, index):
        with h5py.File(self.filename, 'r') as f:
            slice = f['minc-2.0/image/0/image'][index]
        if self.transform is not None:
            slice = self.transform(slice)

        return slice

    def __len__(self):
        with h5py.File(self.filename, 'r') as f:
            return len(f['minc-2.0/image/0/image'])


class MNIBITE(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        mr_root, us_root = os.path.join(root, 'mr'), os.path.join(root, 'us')

        filenames = [f for f in os.listdir(mr_root) if f.endswith('mnc')]
        filenames.sort()

        self.mr = [MINC2(os.path.join(mr_root, f), input_transform)
            for f in filenames]
        self.us = [MINC2(os.path.join(us_root, f), target_transform)
            for f in filenames]

        self.lengths = [len(mr) for mr in self.mr]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, slice):
        index = 0
        offset = 0

        for length in self.lengths:
            if slice < length + offset:
                slice -= offset
                break

            offset += length
            index += 1

        return self.mr[index][slice], self.us[index][slice]

    def __len__(self):
        return sum(self.lengths)