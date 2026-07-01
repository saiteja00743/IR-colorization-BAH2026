"""
models.py – Neural network architectures for the IR Colorization Pipeline
=========================================================================
Three models:
  1. SuperResolutionNet   – SRCNN-style network to upscale 200m TIR → 100m TIR
  2. SegmentationNet      – Lightweight U-Net to predict 6-class land-cover masks
  3. ColorizationNet      – Semantic-guided U-Net/generator to produce RGB from TIR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    """Conv → BatchNorm → ReLU block."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Basic residual block with two 3×3 convs."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(channels, channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class UpBlock(nn.Module):
    """2× bilinear upsample then ConvBNReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBNReLU(in_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # Handle potential size mismatch due to integer arithmetic
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# 1. Super-Resolution Network (SRCNN-inspired with residuals)
# ---------------------------------------------------------------------------

class SuperResolutionNet(nn.Module):
    """
    Upscales a single-channel 200m TIR patch (256×256) to a 512×512 TIR patch.
    
    Input:  (B, 1, 256, 256)  – 200m TIR
    Output: (B, 1, 512, 512)  – 100m TIR
    """
    def __init__(self, num_residual=8, num_features=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(1, num_features, 9, 1, 4),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_residual)])
        self.tail = nn.Sequential(
            # Sub-pixel convolution for 2× upscaling
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),  # → (B, num_features, H*2, W*2)
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 1, 3, 1, 1),
        )

    def forward(self, x):
        # Bicubic upsample as residual base
        base   = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        feat   = self.head(x)
        feat   = self.body(feat)
        detail = self.tail(feat)
        return base + detail


# ---------------------------------------------------------------------------
# 2. Segmentation Network (U-Net)
# ---------------------------------------------------------------------------

NUM_CLASSES = 6  # Water, Vegetation, Urban, Road/Barren, Agriculture, Other


class SegmentationNet(nn.Module):
    """
    Lightweight U-Net for 6-class land-cover segmentation.

    Input:  (B, 1, 512, 512)  – 100m TIR
    Output: (B, 6, 512, 512)  – class logits
    """
    def __init__(self, in_channels=1, num_classes=NUM_CLASSES):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(ConvBNReLU(in_channels, 32), ConvBNReLU(32, 32))
        self.enc2 = nn.Sequential(ConvBNReLU(32, 64), ConvBNReLU(64, 64))
        self.enc3 = nn.Sequential(ConvBNReLU(64, 128), ConvBNReLU(128, 128))
        self.enc4 = nn.Sequential(ConvBNReLU(128, 256), ConvBNReLU(256, 256))

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(ConvBNReLU(256, 512), ConvBNReLU(512, 512))

        # Decoder
        self.dec4 = UpBlock(512 + 256, 256)
        self.dec3 = UpBlock(256 + 128, 128)
        self.dec2 = UpBlock(128 + 64,  64)
        self.dec1 = UpBlock(64 + 32,   32)

        self.out  = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))
        d4 = self.dec4(bn, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out(d1)


# ---------------------------------------------------------------------------
# 3. Colorization Network (Semantic-guided U-Net)
# ---------------------------------------------------------------------------

class ColorizationNet(nn.Module):
    """
    Generates a 3-channel RGB image from 100m TIR and its semantic mask.

    Input:  (B, 1+6, 512, 512)  – concat of TIR + soft segmentation probs
    Output: (B, 3, 512, 512)    – synthesized RGB (values in [0, 1])
    """
    def __init__(self, in_channels=7):  # 1 TIR + 6 class probs
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(ConvBNReLU(in_channels, 64), ConvBNReLU(64, 64))
        self.enc2 = nn.Sequential(ConvBNReLU(64, 128), ConvBNReLU(128, 128))
        self.enc3 = nn.Sequential(ConvBNReLU(128, 256), ConvBNReLU(256, 256))
        self.enc4 = nn.Sequential(ConvBNReLU(256, 512), ConvBNReLU(512, 512))

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ConvBNReLU(512, 1024),
            ConvBNReLU(1024, 512),
        )

        # Decoder
        self.dec4 = UpBlock(512 + 512, 256)
        self.dec3 = UpBlock(256 + 256, 128)
        self.dec2 = UpBlock(128 + 128, 64)
        self.dec1 = UpBlock(64 + 64,   32)

        self.out  = nn.Sequential(
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid(),          # outputs in [0, 1]
        )

    def forward(self, tir, seg_probs):
        x  = torch.cat([tir, seg_probs], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))
        d4 = self.dec4(bn, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out(d1)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_models(device='cpu'):
    """Instantiate and return all three models on the given device."""
    sr   = SuperResolutionNet().to(device)
    seg  = SegmentationNet().to(device)
    col  = ColorizationNet().to(device)
    return sr, seg, col


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sr, seg, col = build_models(device)

    dummy_tir_200 = torch.randn(1, 1, 256, 256).to(device)
    dummy_tir_512 = torch.randn(1, 1, 512, 512).to(device)

    print("=== SuperResolutionNet ===")
    out_sr = sr(dummy_tir_200)
    print(f"  Input:  {dummy_tir_200.shape}")
    print(f"  Output: {out_sr.shape}")

    print("\n=== SegmentationNet ===")
    out_seg_logits = seg(dummy_tir_512)
    print(f"  Input:  {dummy_tir_512.shape}")
    print(f"  Output: {out_seg_logits.shape}")

    seg_probs = torch.softmax(out_seg_logits, dim=1)

    print("\n=== ColorizationNet ===")
    out_rgb = col(dummy_tir_512, seg_probs)
    print(f"  Input:  tir={dummy_tir_512.shape}, seg={seg_probs.shape}")
    print(f"  Output: {out_rgb.shape}")
    print("\nAll model shapes verified. OK")
