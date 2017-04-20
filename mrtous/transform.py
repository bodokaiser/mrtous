import torch

class ToTensor:

    def __call__(self, x):
        return torch.from_numpy(x).transpose(2, 0).transpose(1, 2).float()