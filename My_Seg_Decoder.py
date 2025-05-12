import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel attention using Squeeze-and-Excitation
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()

        # Global context pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Bottleneck fully connected layers
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),  # Reduce channels
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),  # Restore channels
            nn.Sigmoid()  # Output gating weights [0, 1]
        )

    def forward(self, x):
        # Apply learned scaling to each channel
        scale = self.fc(self.pool(x))
        return x * scale

# Final segmentation decoder
class ImprovedSegDecoder(nn.Module):
    def __init__(self, in_channels: int = 384, num_classes: int = 1):
        super(ImprovedSegDecoder, self).__init__()

        # First upsampling block: 32x32 → 64x64, channels: 384 → 192
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            SEBlock(in_channels // 2)
        )

        # Second upsampling block: 64x64 → 128x128, channels: 192 → 96
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            SEBlock(in_channels // 4)
        )

        # Third upsampling block: 128x128 → 256x256, channels: 96 → 48
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            SEBlock(in_channels // 8)
        )

        # Final 1x1 convolution to predict the mask
        self.final_conv = nn.Conv2d(in_channels // 8, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 384, 32, 32)
        Returns:
            mask: Tensor of shape (B, 1, 256, 256)
        """

        # Upsample progressively
        x = self.up1(x)  # → (B, 192, 64, 64)
        x = self.up2(x)  # → (B, 96, 128, 128)
        x = self.up3(x)  # → (B, 48, 256, 256)

        # Final projection to 1-channel binary mask
        mask = self.final_conv(x)  # → (B, 1, 256, 256)

        return mask
