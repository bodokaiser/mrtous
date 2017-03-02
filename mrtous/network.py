import torch
import torch.nn as nn

class Basic(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 3, padding=1)
        self.conn = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.conn(x)
        return x