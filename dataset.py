"""
dataset.py – PyTorch Dataset for loading .npy patch pairs
===========================================================
Loads triplets of:
  - tir_200m.npy      → input to SR model (256×256, single band)
  - tir_100m_512.npy  → SR ground truth (512×512, single band)
  - rgb_100m_512.npy  → Colorization target (512×512, 3 bands)
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


def normalize_tir(arr):
    """Normalize a TIR array to [0, 1] using percentile clipping."""
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    arr = np.clip(arr, p2, p98)
    arr = (arr - p2) / (p98 - p2 + 1e-6)
    return arr.astype(np.float32)


def normalize_rgb(arr):
    """Normalize an RGB array to [0, 1] using per-channel percentile clipping."""
    out = np.zeros_like(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        # (C, H, W)
        for c in range(arr.shape[0]):
            p2, p98 = np.percentile(arr[c], 2), np.percentile(arr[c], 98)
            out[c] = np.clip((arr[c] - p2) / (p98 - p2 + 1e-6), 0, 1)
    else:
        # (H, W, C)
        for c in range(arr.shape[-1]):
            p2, p98 = np.percentile(arr[..., c], 2), np.percentile(arr[..., c], 98)
            out[..., c] = np.clip((arr[..., c] - p2) / (p98 - p2 + 1e-6), 0, 1)
    return out


class IRPatchDataset(Dataset):
    """
    Scans a patches root directory for sample_XXX folders containing:
      tir_200m.npy, tir_100m_512.npy, rgb_100m_512.npy

    Args:
        patches_root: str  – path to output/patches or a product sub-folder
        augment:      bool – apply random horizontal/vertical flip
    """
    def __init__(self, patches_root: str, augment: bool = True):
        self.augment = augment
        self.samples = []

        # Support flat layout (sample_000/...) or product/sample_000/...
        pattern = os.path.join(patches_root, '**', 'tir_200m.npy')
        npy_files = glob.glob(pattern, recursive=True)

        for tir200_path in npy_files:
            sample_dir = os.path.dirname(tir200_path)
            tir100_path = os.path.join(sample_dir, 'tir_100m_512.npy')
            rgb_path    = os.path.join(sample_dir, 'rgb_100m_512.npy')
            if os.path.exists(tir100_path) and os.path.exists(rgb_path):
                self.samples.append((tir200_path, tir100_path, rgb_path))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No valid sample triplets found under '{patches_root}'. "
                "Ensure tir_200m.npy, tir_100m_512.npy, and rgb_100m_512.npy exist."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tir200_path, tir100_path, rgb_path = self.samples[idx]

        tir200 = np.load(tir200_path).astype(np.float32)
        tir100 = np.load(tir100_path).astype(np.float32)
        rgb    = np.load(rgb_path).astype(np.float32)

        # Squeeze band dim if shape is (1, H, W)
        if tir200.ndim == 3 and tir200.shape[0] == 1:
            tir200 = tir200[0]
        if tir100.ndim == 3 and tir100.shape[0] == 1:
            tir100 = tir100[0]

        # Normalize
        tir200 = normalize_tir(tir200)
        tir100 = normalize_tir(tir100)

        # RGB: shape (C, H, W) or (H, W, C)
        if rgb.ndim == 3 and rgb.shape[-1] in (3, 4):
            # (H, W, C) → (C, H, W)
            rgb = rgb.transpose(2, 0, 1)
        rgb = normalize_rgb(rgb)[:3]  # keep first 3 bands

        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                tir200 = np.fliplr(tir200).copy()
                tir100 = np.fliplr(tir100).copy()
                rgb    = rgb[:, :, ::-1].copy()
            if np.random.rand() > 0.5:
                tir200 = np.flipud(tir200).copy()
                tir100 = np.flipud(tir100).copy()
                rgb    = rgb[:, ::-1, :].copy()

        # Add channel dim for TIR
        tir200 = torch.from_numpy(tir200).unsqueeze(0)  # (1, 256, 256)
        tir100 = torch.from_numpy(tir100).unsqueeze(0)  # (1, 512, 512)
        rgb    = torch.from_numpy(rgb.astype(np.float32))            # (3, 512, 512)

        return {
            'tir_200m':     tir200,
            'tir_100m_gt':  tir100,
            'rgb_gt':       rgb,
        }
