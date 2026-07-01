"""
pipeline.py   End-to-end inference: 200m TIR   SR   Segmentation   Colorization
==================================================================================
Usage:
  # Run on the demo patch
  python pipeline.py --input_dir output/patches/demo/sample_006

  # Run on a full product patch directory
  python pipeline.py --input_dir output/patches/demo --product_id demo
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import tifffile
import cv2
from pathlib import Path

from models import SuperResolutionNet, SegmentationNet, ColorizationNet
from dataset import normalize_tir, normalize_rgb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEG_CLASS_COLORS = np.array([
    [0,   0, 255],    # 0 Water          Blue
    [0, 128,   0],    # 1 Vegetation     Dark Green
    [0, 255,   0],    # 2 Agriculture    Lime
    [128, 64, 0],     # 3 Barren         Brown
    [255, 0,   0],    # 4 Urban          Red
    [255, 165, 0],    # 5 Hot/Industr.   Orange
], dtype=np.uint8)

SEG_CLASS_NAMES = ['Water', 'Vegetation', 'Agriculture', 'Barren', 'Urban', 'Hot/Industrial']


def load_model(ModelClass, weights_path, device):
    model = ModelClass().to(device)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"  [OK] Loaded weights: {weights_path}")
    else:
        print(f"  [WARN]   No weights at {weights_path}   using random init (for demo only)")
    model.eval()
    return model


def arr_to_tensor(arr: np.ndarray, device) -> torch.Tensor:
    """Convert (H, W) numpy to (1, 1, H, W) torch tensor."""
    t = torch.from_numpy(arr.astype(np.float32))
    return t.unsqueeze(0).unsqueeze(0).to(device)


def save_tif_bgr(arr: np.ndarray, path: str):
    """Save RGB numpy (H, W, 3) as TIFF in BGR order (Blue=L1, Green=L2, Red=L3)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Reorder to BGR for the submission format
    bgr = arr[..., ::-1].copy()
    # Scale to uint16 for satellite data
    bgr_u16 = (bgr * 65535).astype(np.uint16)
    # (H, W, 3)   (3, H, W) for tifffile
    tifffile.imwrite(path, bgr_u16.transpose(2, 0, 1))
    print(f"  [SAVE] Saved TIF: {path}")


def save_png_preview(arr: np.ndarray, path: str):
    """Save a uint8 PNG preview."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    preview = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Single-sample inference
# ---------------------------------------------------------------------------

def run_inference(tir200_arr: np.ndarray, sr_model, seg_model, col_model, device):
    """
    Full forward pass through the 3-stage pipeline.

    Args:
        tir200_arr: (H, W) numpy float32 array, raw or normalized 200m TIR

    Returns:
        dict with keys:
          'tir_200m_norm'     normalized 200m TIR (H, W)
          'tir_100m_sr'       SR output (H*2, W*2)
          'seg_mask'          argmax class map (H*2, W*2)
          'seg_color'         colorized seg mask (H*2, W*2, 3) uint8
          'rgb_pred'          predicted RGB (H*2, W*2, 3) float32 [0,1]
    """
    tir200_norm = normalize_tir(tir200_arr)
    tir200_t    = arr_to_tensor(tir200_norm, device)

    with torch.no_grad():
        # Stage 1   Super-Resolution
        tir100_t = sr_model(tir200_t).clamp(0, 1)

        # Stage 2   Segmentation
        seg_logits = seg_model(tir100_t)
        seg_probs  = torch.softmax(seg_logits, dim=1)
        seg_mask   = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Stage 3   Colorization
        rgb_t   = col_model(tir100_t, seg_probs).clamp(0, 1)

    tir100_np = tir100_t.squeeze().cpu().numpy()       # (H, W)
    rgb_np    = rgb_t.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

    # Colorize seg mask
    seg_color = SEG_CLASS_COLORS[seg_mask]

    return {
        'tir_200m_norm': tir200_norm,
        'tir_100m_sr':   tir100_np,
        'seg_mask':      seg_mask,
        'seg_color':     seg_color,
        'rgb_pred':      rgb_np,
    }


# ---------------------------------------------------------------------------
# Directory-level inference
# ---------------------------------------------------------------------------

def process_sample_dir(sample_dir: str, sr_model, seg_model, col_model,
                        device, output_base: str, product_id: str):
    """Process a single sample_XXX directory."""
    tir200_path = os.path.join(sample_dir, 'tir_200m.npy')
    if not os.path.exists(tir200_path):
        print(f"  [WARN]   Skipping {sample_dir}: tir_200m.npy not found")
        return

    tir200_arr = np.load(tir200_path).astype(np.float32)
    if tir200_arr.ndim == 3:
        tir200_arr = tir200_arr[0]

    results = run_inference(tir200_arr, sr_model, seg_model, col_model, device)

    # Paths
    sr_tif_path  = os.path.join(output_base, 'tir_superresolved_100m', f'{product_id}.tif')
    col_tif_path = os.path.join(output_base, 'colorized_tir_100m',     f'{product_id}.tif')
    preview_dir  = os.path.join(output_base, 'previews', product_id)

    # Save mandatory TIF outputs
    tir100_np = results['tir_100m_sr']
    tir100_u16 = (tir100_np * 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(sr_tif_path), exist_ok=True)
    tifffile.imwrite(sr_tif_path, tir100_u16)
    print(f"  [SAVE] SR TIF: {sr_tif_path}")

    save_tif_bgr(results['rgb_pred'], col_tif_path)

    # Save PNG previews
    os.makedirs(preview_dir, exist_ok=True)
    # 200m TIR
    tir200_u8 = (np.clip(results['tir_200m_norm'], 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(preview_dir, '1_tir_200m.png'), tir200_u8)
    # SR 100m
    tir100_u8 = (np.clip(results['tir_100m_sr'], 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(preview_dir, '2_tir_sr_100m.png'), tir100_u8)
    # Seg mask
    cv2.imwrite(os.path.join(preview_dir, '3_seg_mask.png'),
                cv2.cvtColor(results['seg_color'], cv2.COLOR_RGB2BGR))
    # Colorized
    save_png_preview(results['rgb_pred'], os.path.join(preview_dir, '4_colorized_rgb.png'))

    print(f"  [IMG]   Previews saved: {preview_dir}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='End-to-end IR Colorization Inference')
    parser.add_argument('--input_dir',   default='output/patches/demo/sample_006',
                        help='Path to a sample directory OR a directory containing sample dirs')
    parser.add_argument('--output_dir',  default='output/model_outputs',
                        help='Output directory (mandatory submission structure)')
    parser.add_argument('--product_id',  default=None,
                        help='Product ID for output filename (default: inferred from dir name)')
    parser.add_argument('--weights_dir', default='weights',
                        help='Directory containing model weights')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[SETUP] Device: {device}")

    # Load models
    sr_model  = load_model(SuperResolutionNet, os.path.join(args.weights_dir, 'sr_best.pth'),  device)
    seg_model = load_model(SegmentationNet,    os.path.join(args.weights_dir, 'seg_best.pth'), device)
    col_model = load_model(ColorizationNet,    os.path.join(args.weights_dir, 'col_best.pth'), device)

    input_dir  = args.input_dir
    output_dir = args.output_dir

    # Determine if input_dir is a sample dir or a product dir
    if os.path.exists(os.path.join(input_dir, 'tir_200m.npy')):
        # Single sample
        product_id = args.product_id or Path(input_dir).name
        print(f"\n[SAMPLE] Processing sample: {input_dir}")
        process_sample_dir(input_dir, sr_model, seg_model, col_model,
                           device, output_dir, product_id)
    else:
        # Directory of samples
        sample_dirs = sorted([
            d for d in Path(input_dir).rglob('*')
            if d.is_dir() and (d / 'tir_200m.npy').exists()
        ])
        if not sample_dirs:
            print(f"[ERR] No tir_200m.npy found under {input_dir}")
            return
        for sample_dir in sample_dirs:
            product_id = args.product_id or f"{sample_dir.parent.name}_{sample_dir.name}"
            print(f"\n[SAMPLE] Processing sample: {sample_dir}")
            process_sample_dir(str(sample_dir), sr_model, seg_model, col_model,
                               device, output_dir, product_id)

    print(f"\n[OK] Inference complete. Results in: {output_dir}")


if __name__ == '__main__':
    main()
