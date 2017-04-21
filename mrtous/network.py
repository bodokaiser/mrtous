import torch
import torch.nn as nn

class Simple(nn.Module):

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