"""
train.py   Training script for SR, Segmentation, and Colorization models
=========================================================================
Usage examples:

  # Train Super-Resolution
  python train.py --model sr --patches_dir output/patches --epochs 50

  # Train Segmentation (requires seg labels or uses pseudo-labels from thresholding)
  python train.py --model seg --patches_dir output/patches --epochs 30

  # Train Colorization (requires pre-trained SR model)
  python train.py --model col --patches_dir output/patches --epochs 50 --sr_weights weights/sr_best.pth
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from models import SuperResolutionNet, SegmentationNet, ColorizationNet, NUM_CLASSES
from dataset import IRPatchDataset


# ---------------------------------------------------------------------------
# Perceptual Loss (simple feature-level MSE)
# ---------------------------------------------------------------------------

class CombinedSRLoss(nn.Module):
    """L1 + MS-SSIM inspired: L1 + gradient loss for SR."""
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def gradient_loss(self, pred, target):
        def grad(t):
            dx = t[:, :, :, 1:] - t[:, :, :, :-1]
            dy = t[:, :, 1:, :] - t[:, :, :-1, :]
            return dx, dy
        pdx, pdy = grad(pred)
        tdx, tdy = grad(target)
        return self.l1(pdx, tdx) + self.l1(pdy, tdy)

    def forward(self, pred, target):
        return self.l1(pred, target) + 0.1 * self.gradient_loss(pred, target)


# ---------------------------------------------------------------------------
# Pseudo segmentation labels from TIR band thresholding
# ---------------------------------------------------------------------------

def make_pseudo_seg_labels(tir_batch: torch.Tensor) -> torch.Tensor:
    """
    Creates rough 6-class pseudo labels from TIR intensity.
    Classes (rough mapping):
      0=Water (very cold), 1=Vegetation (cool), 2=Agriculture (mild-cool),
      3=Barren (mild-warm), 4=Urban (warm), 5=Industrial/Hot (very hot)
    """
    B, C, H, W = tir_batch.shape
    labels = torch.zeros(B, H, W, dtype=torch.long, device=tir_batch.device)
    t = tir_batch[:, 0]  # (B, H, W) values in [0, 1]
    labels[t < 0.15] = 0
    labels[(t >= 0.15) & (t < 0.30)] = 1
    labels[(t >= 0.30) & (t < 0.45)] = 2
    labels[(t >= 0.45) & (t < 0.60)] = 3
    labels[(t >= 0.60) & (t < 0.80)] = 4
    labels[t >= 0.80] = 5
    return labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_sr(args, device):
    """Train Super-Resolution network."""
    print("\n[START] Training Super-Resolution Network")
    dataset = IRPatchDataset(args.patches_dir, augment=True)
    n_val   = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=device.type == 'cuda')
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model     = SuperResolutionNet().to(device)
    criterion = CombinedSRLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler(enabled=device.type == 'cuda')

    os.makedirs(args.weights_dir, exist_ok=True)
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': []}
    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        t_loss = 0.0
        for batch in train_dl:
            tir200 = batch['tir_200m'].to(device)
            tir100 = batch['tir_100m_gt'].to(device)
            optimizer.zero_grad()
            with autocast(enabled=device.type == 'cuda'):
                pred = model(tir200)
                loss = criterion(pred, tir100)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item()

        scheduler.step()
        t_loss /= max(1, len(train_dl))

        # --- Val ---
        model.eval()
        v_loss = 0.0
        v_psnr = 0.0
        with torch.no_grad():
            for batch in val_dl:
                tir200 = batch['tir_200m'].to(device)
                tir100 = batch['tir_100m_gt'].to(device)
                pred   = model(tir200)
                v_loss += criterion(pred, tir100).item()
                v_psnr += psnr(pred.clamp(0, 1), tir100)
        v_loss /= max(1, len(val_dl))
        v_psnr /= max(1, len(val_dl))

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_psnr'].append(v_psnr)

        print(f"  Epoch {epoch:3d}/{args.epochs} | train_loss={t_loss:.4f} | val_loss={v_loss:.4f} | val_PSNR={v_psnr:.2f}dB")

        if v_psnr > best_psnr:
            best_psnr = v_psnr
            path = os.path.join(args.weights_dir, 'sr_best.pth')
            torch.save(model.state_dict(), path)
            print(f"    [OK] Saved best SR model (PSNR={best_psnr:.2f}dB)   {path}")

    # Save final
    torch.save(model.state_dict(), os.path.join(args.weights_dir, 'sr_final.pth'))
    _save_history(history, args.weights_dir, 'sr_history.json')
    return history


def train_seg(args, device):
    """Train Segmentation network using pseudo-labels."""
    print("\n[START] Training Segmentation Network (pseudo-labels from TIR)")
    dataset  = IRPatchDataset(args.patches_dir, augment=True)
    n_val    = max(1, int(0.15 * len(dataset)))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model     = SegmentationNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler(enabled=device.type == 'cuda')

    os.makedirs(args.weights_dir, exist_ok=True)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss = 0.0
        for batch in train_dl:
            tir100 = batch['tir_100m_gt'].to(device)
            labels = make_pseudo_seg_labels(tir100)
            optimizer.zero_grad()
            with autocast(enabled=device.type == 'cuda'):
                logits = model(tir100)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item()
        scheduler.step()
        t_loss /= max(1, len(train_dl))

        model.eval()
        v_loss = 0.0
        v_acc  = 0.0
        with torch.no_grad():
            for batch in val_dl:
                tir100 = batch['tir_100m_gt'].to(device)
                labels = make_pseudo_seg_labels(tir100)
                logits = model(tir100)
                v_loss += criterion(logits, labels).item()
                preds   = logits.argmax(dim=1)
                v_acc  += (preds == labels).float().mean().item()
        v_loss /= max(1, len(val_dl))
        v_acc  /= max(1, len(val_dl))

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        print(f"  Epoch {epoch:3d}/{args.epochs} | train_loss={t_loss:.4f} | val_loss={v_loss:.4f} | val_acc={v_acc:.3f}")

        if v_acc > best_acc:
            best_acc = v_acc
            path = os.path.join(args.weights_dir, 'seg_best.pth')
            torch.save(model.state_dict(), path)
            print(f"    [OK] Saved best Seg model (acc={best_acc:.3f})   {path}")

    torch.save(model.state_dict(), os.path.join(args.weights_dir, 'seg_final.pth'))
    _save_history(history, args.weights_dir, 'seg_history.json')
    return history


def train_col(args, device):
    """Train Colorization network (requires SR weights to generate 100m TIR)."""
    print("\n[START] Training Colorization Network")
    dataset  = IRPatchDataset(args.patches_dir, augment=True)
    n_val    = max(1, int(0.15 * len(dataset)))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load pre-trained SR model to generate 100m TIR if no direct 100m GT given
    sr_model = SuperResolutionNet().to(device)
    sr_weights = args.sr_weights or os.path.join(args.weights_dir, 'sr_best.pth')
    if os.path.exists(sr_weights):
        sr_model.load_state_dict(torch.load(sr_weights, map_location=device))
        print(f"  Loaded SR weights from {sr_weights}")
    sr_model.eval()

    # Load pre-trained Seg model
    seg_model = SegmentationNet().to(device)
    seg_weights = os.path.join(args.weights_dir, 'seg_best.pth')
    if os.path.exists(seg_weights):
        seg_model.load_state_dict(torch.load(seg_weights, map_location=device))
        print(f"  Loaded Seg weights from {seg_weights}")
    seg_model.eval()

    col_model = ColorizationNet().to(device)
    optimizer = optim.AdamW(col_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.L1Loss()
    scaler    = GradScaler(enabled=device.type == 'cuda')

    os.makedirs(args.weights_dir, exist_ok=True)
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': []}
    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        col_model.train()
        t_loss = 0.0
        for batch in train_dl:
            tir200  = batch['tir_200m'].to(device)
            rgb_gt  = batch['rgb_gt'].to(device)
            tir100_gt = batch['tir_100m_gt'].to(device)

            with torch.no_grad():
                tir100_sr = sr_model(tir200).clamp(0, 1)
                # Use the better of SR output and GT when GT available
                tir100    = tir100_gt  # use GT for training stability
                seg_logits = seg_model(tir100)
                seg_probs  = torch.softmax(seg_logits, dim=1)

            optimizer.zero_grad()
            with autocast(enabled=device.type == 'cuda'):
                rgb_pred = col_model(tir100, seg_probs)
                loss     = criterion(rgb_pred, rgb_gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item()

        scheduler.step()
        t_loss /= max(1, len(train_dl))

        col_model.eval()
        v_loss = 0.0
        v_psnr = 0.0
        with torch.no_grad():
            for batch in val_dl:
                tir200    = batch['tir_200m'].to(device)
                rgb_gt    = batch['rgb_gt'].to(device)
                tir100_gt = batch['tir_100m_gt'].to(device)
                seg_logits = seg_model(tir100_gt)
                seg_probs  = torch.softmax(seg_logits, dim=1)
                rgb_pred   = col_model(tir100_gt, seg_probs)
                v_loss += criterion(rgb_pred, rgb_gt).item()
                v_psnr += psnr(rgb_pred.clamp(0, 1), rgb_gt)

        v_loss /= max(1, len(val_dl))
        v_psnr /= max(1, len(val_dl))

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_psnr'].append(v_psnr)

        print(f"  Epoch {epoch:3d}/{args.epochs} | train_loss={t_loss:.4f} | val_loss={v_loss:.4f} | val_PSNR={v_psnr:.2f}dB")

        if v_psnr > best_psnr:
            best_psnr = v_psnr
            path = os.path.join(args.weights_dir, 'col_best.pth')
            torch.save(col_model.state_dict(), path)
            print(f"    [OK] Saved best Col model (PSNR={best_psnr:.2f}dB)   {path}")

    torch.save(col_model.state_dict(), os.path.join(args.weights_dir, 'col_final.pth'))
    _save_history(history, args.weights_dir, 'col_history.json')
    return history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_history(history: dict, weights_dir: str, filename: str):
    path = os.path.join(weights_dir, filename)
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  [STATS] History saved to {path}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train IR Colorization Pipeline Models')
    parser.add_argument('--model',       choices=['sr', 'seg', 'col', 'all'],
                        default='all', help='Which model to train')
    parser.add_argument('--patches_dir', default='output/patches',
                        help='Root directory containing patch subfolders')
    parser.add_argument('--weights_dir', default='weights',
                        help='Directory to save model weights')
    parser.add_argument('--epochs',      type=int,   default=30)
    parser.add_argument('--batch_size',  type=int,   default=2)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--sr_weights',  type=str,   default=None,
                        help='Path to pre-trained SR weights (for colorization training)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    start = time.time()
    if args.model in ('sr', 'all'):
        train_sr(args, device)
    if args.model in ('seg', 'all'):
        train_seg(args, device)
    if args.model in ('col', 'all'):
        train_col(args, device)

    elapsed = time.time() - start
    print(f"\n[OK] Training complete in {elapsed/60:.1f} minutes.")


if __name__ == '__main__':
    main()
