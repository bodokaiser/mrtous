import os
import h5py
import numpy as np
import skimage as sk

from skimage import io, util
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

class MNIBITENative(Dataset):

    def __init__(self, root, id, transform=None):
        self.mr = MINC2(os.path.join(root, f'{id:02d}_mr.mnc'))
        self.us = MINC2(os.path.join(root, f'{id:02d}_us.mnc'))
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

    def __init__(self, root):
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

    def __getitem__(self, index):
        mr = sk.img_as_float(io.imread(self.mr_fnames[index]))
        us = sk.img_as_float(io.imread(self.us_fnames[index]))

        return mr.astype(np.float32), us.astype(np.float32)

    def __len__(self):
        return len(self.mr_fnames)