import torch
import torch.nn as nn

def init_weight(layer):
    layer.weight.data.normal_(0.5, 0.2)

class Basic(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 3, padding=1)
        self.conn = nn.Conv2d(3, 1, 1)

        init_weight(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.conn(x)
        return x