import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    """
    Enhanced SRCNN for 2x image super-resolution.
    Improved with deeper architecture and better initialization.
    """
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        
        # Feature extraction layer
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        
        # Non-linear mapping layers (deeper than original)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        
        # Reconstruction layer
        self.conv4 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Bicubic upsampling first
        x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        
        return x


class EDSR_Lite(nn.Module):
    """
    Lightweight EDSR-style residual network for better performance.
    Uses residual blocks and skip connections.
    """
    def __init__(self, num_channels=1, num_features=64, num_blocks=8, scale=2):
        super(EDSR_Lite, self).__init__()
        
        self.scale = scale
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_features) for _ in range(num_blocks)
        ])
        
        # Final convolution before upsampling
        self.conv_before_upsample = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
        )
        
        # Final output layer
        self.conv_last = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial feature extraction
        x = self.conv_first(x)
        residual = x
        
        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Add skip connection
        x = self.conv_before_upsample(x) + residual
        
        # Upsample and reconstruct
        x = self.upsample(x)
        x = self.conv_last(x)
        
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers.
    """
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual  # Skip connection
        return out