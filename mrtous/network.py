import torch
import torch.nn as nn
import torch.nn.functional as ff

class Basic(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 3, 3, 1, 1)
        self.conn = nn.Conv2d(3, 1, 1)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.conn(outputs)

        return outputs

class UNetConv(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_size, out_size, 3, padding=1),
            nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)

        return outputs

class UNetEncode(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv = UNetConv(in_size, out_size)
        self.down = nn.MaxPool2d(2, 1)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.down(outputs)

        return outputs

class UNetDecode(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv = UNetConv(in_size, out_size)
        self.up = nn.ConvTranspose2d(in_size, out_size, 2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)

        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2*[offset // 2, offset // 2 + 1]

        outputs1 = ff.pad(inputs1, padding)

        return self.conv(torch.cat([outputs1, outputs2], 1))

class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encode1 = UNetEncode(1, 32)
        self.center = UNetConv(32, 64)
        self.decode1 = UNetDecode(64, 32)
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, inputs):
        encode1 = self.encode1(inputs)
        center = self.center(encode1)
        decode1 = self.decode1(encode1, center)

        return self.final(decode1)