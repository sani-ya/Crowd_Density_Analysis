"""
evaluate.py — Evaluation script for trained crowd counting models.

Usage:
    python evaluate.py --config config.yaml
    python evaluate.py --config config.yaml --checkpoint ./checkpoints/best_model.pth
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import cv2
import yaml

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Core Metrics
# ──────────────────────────────────────────────────────────────

def compute_metrics(model, loader, device):
    """
    Compute MAE and MSE over an entire dataloader.

    MAE (Mean Absolute Error)  — average per-image count error.
    MSE (Mean Squared Error)   — penalises large errors more; use for outlier sensitivity.

    Returns: (mae, mse) both as Python floats.
    """
    model.eval()
    mae_list, mse_list = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            gt_counts = batch['count'].numpy()   # (B,)

            pred_maps = model(images)            # (B, 1, H', W')

            # Upsample density map to original size if needed
            pred_counts = pred_maps.sum(dim=[1, 2, 3]).cpu().numpy()  # sum over spatial

            for pred, gt in zip(pred_counts, gt_counts):
                err = abs(pred - gt)
                mae_list.append(err)
                mse_list.append(err ** 2)

    mae = float(np.mean(mae_list))
    mse = float(np.sqrt(np.mean(mse_list)))   # RMSE convention in crowd counting
    return mae, mse


# ──────────────────────────────────────────────────────────────
#  Per-image evaluation with optional heatmap saving
# ──────────────────────────────────────────────────────────────

def evaluate_and_save(model, loader, device, output_dir: str):
    """
    Run inference on all images, save density heatmaps and a results CSV.
    """
    model.eval()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            paths  = batch['path']
            gt_counts = batch['count'].numpy()

            pred_maps = model(images)
            pred_counts = pred_maps.sum(dim=[1, 2, 3]).cpu().numpy()

            for i, (path, pred, gt) in enumerate(zip(paths, pred_counts, gt_counts)):
                img_name = Path(path).stem
                err = abs(pred - gt)

                # Save heatmap overlay
                density = pred_maps[i, 0].cpu().numpy()
                heatmap_path = os.path.join(output_dir, f'{img_name}_density.png')
                save_density_heatmap(density, heatmap_path)

                results.append({
                    'image': img_name,
                    'gt_count': float(gt),
                    'pred_count': float(pred),
                    'abs_error': float(err),
                })
                logger.info(f"  {img_name}: GT={gt:.0f}  Pred={pred:.1f}  Err={err:.1f}")

    # Save CSV
    import csv
    csv_path = os.path.join(output_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'gt_count', 'pred_count', 'abs_error'])
        writer.writeheader()
        writer.writerows(results)

    errors = [r['abs_error'] for r in results]
    mae = np.mean(errors)
    mse = np.sqrt(np.mean([e**2 for e in errors]))
    logger.info(f"\n{'='*40}")
    logger.info(f"FINAL  MAE: {mae:.2f}   RMSE: {mse:.2f}")
    logger.info(f"Results saved to {output_dir}")
    return mae, mse


def save_density_heatmap(density: np.ndarray, path: str,
                          colormap: int = cv2.COLORMAP_JET):
    """
    Convert a raw density map to a colourized heatmap PNG.
    High-density regions appear red; low-density blue.
    """
    d_norm = density / (density.max() + 1e-8)   # normalise to [0, 1]
    d_uint8 = (d_norm * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(d_uint8, colormap)
    cv2.imwrite(path, heatmap)


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate crowd counting model')
    p.add_argument('--config', type=str, default='config.yaml')
    p.add_argument('--checkpoint', type=str, help='Override checkpoint path')
    p.add_argument('--output_dir', type=str, help='Override output directory')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s')

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if args.checkpoint:
        cfg['evaluation']['checkpoint'] = args.checkpoint
    if args.output_dir:
        cfg['evaluation']['prediction_dir'] = args.output_dir

    device_str = cfg['system'].get('device', 'auto')
    device = torch.device('cuda' if (device_str == 'auto' and torch.cuda.is_available())
                          else device_str if device_str != 'auto' else 'cpu')

    from dataset import build_dataloaders
    from model import build_model

    _, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    ckpt = torch.load(cfg['evaluation']['checkpoint'], map_location=device)
    model.load_state_dict(ckpt['model_state'])
    logger.info(f"Loaded checkpoint: {cfg['evaluation']['checkpoint']}")

    if cfg['evaluation'].get('save_predictions', False):
        evaluate_and_save(model, val_loader, device,
                          cfg['evaluation']['prediction_dir'])
    else:
        mae, mse = compute_metrics(model, val_loader, device)
        logger.info(f"MAE: {mae:.2f}   RMSE: {mse:.2f}")
