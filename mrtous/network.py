import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

def center_crop(x, size):
    crop_h = torch.FloatTensor([x.size()[2]]).sub(size[0]).div(-2)
    crop_w = torch.FloatTensor([x.size()[3]]).sub(size[1]).div(-2)

    return F.pad(x, [
        crop_w.ceil().int()[0], crop_w.floor().int()[0],
        crop_h.ceil().int()[0], crop_h.floor().int()[0],
    ])

class One(nn.Module):

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.encoder = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.decoder(x)
        x = self.encoder(x)

        return x


class Two(nn.Module):

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        dec = self.decoder(x)
        enc = self.encoder(dec)

        return self.final(torch.cat([dec, enc], 1))


class UNetConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.decode1 = nn.Sequential(
            UNetConv(1, 64),
            nn.MaxPool2d(2, 2),
        )
        self.decode2 = nn.Sequential(
            UNetConv(64, 128),
            nn.MaxPool2d(2, 2),
        )
        self.decode3 = nn.Sequential(
            UNetConv(128, 256),
            nn.MaxPool2d(2, 2),
        )
        self.decode4 = nn.Sequential(
            UNetConv(256, 512),
            nn.MaxPool2d(2, 2),
        )
        self.encode4 = nn.Sequential(
            UNetConv(512, 1024),
            nn.ConvTranspose2d(1024, 512, 2, 2),
        )
        self.encode3 = nn.Sequential(
            UNetConv(1024, 512),
            nn.ConvTranspose2d(512, 256, 2, 2),
        )
        self.encode2 = nn.Sequential(
            UNetConv(512, 256),
            nn.ConvTranspose2d(256, 128, 2, 2),
        )
        self.encode1 = nn.Sequential(
            UNetConv(256, 128),
            nn.ConvTranspose2d(128, 64, 2, 2),
        )
        self.final = nn.Sequential(
            UNetConv(128, 64),
            nn.Conv2d(64, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

    def concat(self, x, y):
        return torch.cat([x, center_crop(y, x.size()[2:])], 1)

    def forward(self, x):
        dec1 = self.decode1(x)
        dec2 = self.decode2(dec1)
        dec3 = self.decode3(dec2)
        dec4 = self.decode4(dec3)
        enc4 = self.encode4(dec4)
        enc3 = self.encode3(self.concat(enc4, dec4))
        enc2 = self.encode2(self.concat(enc3, dec3))
        enc1 = self.encode1(self.concat(enc2, dec2))
        fin = self.final(self.concat(enc1, dec1))

        return center_crop(fin, x.size()[2:])


class USNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, indices):
        x = F.max_unpool2d(x, indices, 2, 2)

        return self.encode(x)


class USNet(nn.Module):

    def __init__(self):
        super().__init__()

        modules = list(models.vgg16_bn().features.children())
        modules[0] = nn.Conv2d(1, 64, 3, padding=1)

        self.decode1 = nn.Sequential(*modules[:5])
        self.decode2 = nn.Sequential(*modules[6:12])
        self.decode3 = nn.Sequential(*modules[13:22])
        self.decode4 = nn.Sequential(*modules[23:32])
        self.encode4 = USNetEnc(512, 256)
        self.encode3 = USNetEnc(256, 128)
        self.encode2 = USNetEnc(128, 64)
        self.encode1 = USNetEnc(64, 64)
        self.final = nn.Conv2d(64, 1, 1)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x):
        dec1, ind1 = self.pool(self.decode1(x))
        dec2, ind2 = self.pool(self.decode2(dec1))
        dec3, ind3 = self.pool(self.decode3(dec2))
        dec4, ind4 = self.pool(self.decode4(dec3))
        enc4 = self.encode4(dec4, ind4)
        enc3 = self.encode3(enc4, ind3)
        enc2 = self.encode2(enc3, ind2)
        enc1 = self.encode1(enc2, ind1)

        return self.final(enc1)