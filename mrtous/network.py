import torch
import torch.nn as nn

def normal_init(model):
    classname = model.__class__.__name__

    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.5, 0.3)

class Basic(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 3, 3, padding=1)
        self.conn = nn.Conv2d(3, 1, 1)

        self.apply(normal_init)

    def forward(self, x):
        x = self.conv(x)
        x = self.conn(x)
        return x