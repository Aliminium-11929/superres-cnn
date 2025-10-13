import torch
import torch.nn as nn

class SRCNN(nn.Module):
    """
    Simplified SRCNN baseline for 2x image super-resolution.
    Input and output: grayscale images (1 channel).
    """
    def __init__(self):
        super(SRCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.layers(x)
