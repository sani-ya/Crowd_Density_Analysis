"""
train.py — Training loop for crowd density estimation.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --dataset_path ./data/ShanghaiTech/part_B --epochs 200
    python train.py --config config.yaml --resume ./checkpoints/epoch_50.pth
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

import yaml

from dataset import build_dataloaders
from model import build_model
from evaluate import compute_metrics

# ──────────────────────────────────────────────────────────────
#  Logging setup
# ──────────────────────────────────────────────────────────────

def setup_logging(cfg: dict):
    log_cfg = cfg.get('logging', {})
    log_file = log_cfg.get('log_file', './logs/train.log')
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, log_cfg.get('level', 'INFO'))
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ]
    logging.basicConfig(level=level, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=handlers)
    return logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Optional TensorBoard / WandB
# ──────────────────────────────────────────────────────────────

def build_writer(cfg: dict):
    log_cfg = cfg.get('logging', {})
    writer = None

    if log_cfg.get('tensorboard', False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = log_cfg.get('tensorboard_dir', './runs')
            writer = SummaryWriter(log_dir=tb_dir)
            logging.getLogger(__name__).info(f"TensorBoard logging → {tb_dir}")
        except ImportError:
            logging.getLogger(__name__).warning("tensorboard not installed; skipping.")

    if log_cfg.get('wandb', False):
        try:
            import wandb
            wandb.init(
                project=log_cfg.get('wandb_project', 'crowd-analytics'),
                entity=log_cfg.get('wandb_entity'),
                config=cfg,
            )
        except ImportError:
            logging.getLogger(__name__).warning("wandb not installed; skipping.")

    return writer


# ──────────────────────────────────────────────────────────────
#  Loss functions
# ──────────────────────────────────────────────────────────────

def build_criterion(cfg: dict):
    loss_name = cfg['training']['loss_function'].lower()
    if loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'mae':
        return nn.L1Loss()
    elif loss_name == 'ssim_mse':
        # Combine MSE with structural similarity for better spatial accuracy
        mse = nn.MSELoss()
        def ssim_mse(pred, target):
            # Simple implementation: weighted sum of pixel MSE + gradient MSE
            grad_x_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            grad_x_t = target[:, :, :, 1:] - target[:, :, :, :-1]
            grad_y_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            grad_y_t = target[:, :, 1:, :] - target[:, :, :-1, :]
            return mse(pred, target) + 0.1 * (mse(grad_x_p, grad_x_t) + mse(grad_y_p, grad_y_t))
        return ssim_mse
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# ──────────────────────────────────────────────────────────────
#  Optimiser & Scheduler
# ──────────────────────────────────────────────────────────────

def build_optimizer(model, cfg: dict):
    tc = cfg['training']
    params = model.parameters()
    if tc['optimizer'].lower() == 'adam':
        return optim.Adam(params, lr=tc['learning_rate'],
                          weight_decay=tc['weight_decay'])
    elif tc['optimizer'].lower() == 'sgd':
        return optim.SGD(params, lr=tc['learning_rate'],
                         momentum=tc.get('momentum', 0.9),
                         weight_decay=tc['weight_decay'])
    raise ValueError(f"Unknown optimizer: {tc['optimizer']}")


def build_scheduler(optimizer, cfg: dict):
    tc = cfg['training']
    sched = tc.get('lr_scheduler', 'plateau')
    if sched == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='min',
                                  patience=tc.get('lr_patience', 10),
                                  factor=tc.get('lr_factor', 0.5))
                                  
    elif sched == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=tc['epochs'])
    return None


# ──────────────────────────────────────────────────────────────
#  Checkpoint helpers
# ──────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    start_epoch = ckpt.get('epoch', 0) + 1
    best_mae = ckpt.get('best_mae', float('inf'))
    if optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    logging.getLogger(__name__).info(
        f"Resumed from {path} | epoch {start_epoch} | best_mae {best_mae:.2f}")
    return start_epoch, best_mae


# ──────────────────────────────────────────────────────────────
#  Train one epoch
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, cfg, logger):
    model.train()
    total_loss = 0.0
    clip = cfg['training'].get('grad_clip', 1.0)

    for batch_idx, batch in enumerate(loader):
        images   = batch['image'].to(device, non_blocking=True)
        densities = batch['density'].to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(images)

        # Upsample prediction to match target density map size if needed
        if pred.shape != densities.shape:
            pred = torch.nn.functional.interpolate(
                pred, size=densities.shape[2:], mode='bilinear', align_corners=False)

        loss = criterion(pred, densities)
        loss.backward()

        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            logger.info(f"  Batch [{batch_idx+1}/{len(loader)}] loss: {loss.item():.6f}")

    return total_loss / len(loader)


# ──────────────────────────────────────────────────────────────
#  Main Training Loop
# ──────────────────────────────────────────────────────────────

def train(cfg: dict, override_args: argparse.Namespace = None):
    logger = setup_logging(cfg)
    writer = build_writer(cfg)

    # ── Seed & Device ────────────────────────────────────────
    seed = cfg['system'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_str = cfg['system'].get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    # ── Data ────────────────────────────────────────────────
    logger.info("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ───────────────────────────────────────────────
    logger.info(f"Building model: {cfg['model']['architecture']}")
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total_params:,}")

    # ── Optimiser & Loss ────────────────────────────────────
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # ── Resume ──────────────────────────────────────────────
    start_epoch = 0
    best_mae = float('inf')
    resume_path = cfg['training'].get('resume_checkpoint')
    if resume_path and os.path.isfile(resume_path):
        start_epoch, best_mae = load_checkpoint(resume_path, model, optimizer, device)

    tc = cfg['training']
    epochs = tc['epochs']
    ckpt_dir = tc['checkpoint_dir']
    save_every = tc.get('save_every_n_epochs', 10)
    save_best = tc.get('save_best_only', True)

    # ── Training loop ───────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch [{epoch+1}/{epochs}]  LR={optimizer.param_groups[0]['lr']:.2e}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg, logger)

        # Validation
        val_mae, val_mse = compute_metrics(model, val_loader, device)
        epoch_time = time.time() - t0

        logger.info(f"Train Loss: {train_loss:.6f} | "
                    f"Val MAE: {val_mae:.2f} | Val MSE: {val_mse:.2f} | "
                    f"Time: {epoch_time:.1f}s")

        # LR scheduling
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_mae)
            else:
                scheduler.step()

        # TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Metrics/MAE', val_mae, epoch)
            writer.add_scalar('Metrics/MSE', val_mse, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # WandB
        try:
            import wandb
            if wandb.run:
                wandb.log({'train_loss': train_loss, 'val_mae': val_mae,
                           'val_mse': val_mse, 'epoch': epoch})
        except ImportError:
            pass

        # Save best
        if val_mae < best_mae:
            best_mae = val_mae
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_mae': best_mae,
                'config': cfg,
            }, os.path.join(ckpt_dir, 'best_model.pth'))
            logger.info(f"  ✓ New best model saved (MAE={best_mae:.2f})")

        # Periodic save
        if not save_best and (epoch + 1) % save_every == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_mae': best_mae,
            }, os.path.join(ckpt_dir, f'epoch_{epoch+1:04d}.pth'))

    logger.info(f"\nTraining complete. Best MAE: {best_mae:.2f}")
    if writer:
        writer.close()


# ──────────────────────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train crowd counting model')
    p.add_argument('--config', type=str, default='config.yaml')
    p.add_argument('--dataset_path', type=str, help='Override dataset root path')
    p.add_argument('--batch_size', type=int, help='Override batch size')
    p.add_argument('--lr', type=float, help='Override learning rate')
    p.add_argument('--epochs', type=int, help='Override number of epochs')
    p.add_argument('--model_save_path', type=str, help='Override checkpoint directory')
    p.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    p.add_argument('--arch', type=str, help='Override model architecture (csrnet|mcnn)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides (dataset-agnostic: just change config, no code edits needed)
    if args.dataset_path:
        cfg['dataset']['root'] = args.dataset_path
    if args.batch_size:
        cfg['training']['batch_size'] = args.batch_size
    if args.lr:
        cfg['training']['learning_rate'] = args.lr
    if args.epochs:
        cfg['training']['epochs'] = args.epochs
    if args.model_save_path:
        cfg['training']['checkpoint_dir'] = args.model_save_path
    if args.resume:
        cfg['training']['resume_checkpoint'] = args.resume
    if args.arch:
        cfg['model']['architecture'] = args.arch

    train(cfg)
