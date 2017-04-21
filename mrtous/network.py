import torch
import torch.nn as nn

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