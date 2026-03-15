"""
train_forecaster.py — Train the GRU/LSTM crowd forecaster.

Expects a CSV with column 'count' (one row per time step, e.g. 1/minute).
The model learns to predict the next N counts given M past counts.

Usage:
    python train_forecaster.py --config config.yaml
    python train_forecaster.py --config config.yaml --data ./data/my_counts.csv
"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


# ──────────────────────────────────────────────────────────────
#  Time-Series Dataset
# ──────────────────────────────────────────────────────────────

class CrowdTimeSeriesDataset(Dataset):
    """
    Sliding-window dataset from a CSV of crowd counts.

    Each sample: (seq_len counts) → (forecast_horizon counts)
    Counts are normalised to [0, 1] using the training max.
    """

    def __init__(self, counts: np.ndarray, seq_len: int, horizon: int):
        self.counts = counts.astype(np.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.max_val = self.counts.max() + 1e-6

    def __len__(self):
        return len(self.counts) - self.seq_len - self.horizon + 1

    def __getitem__(self, idx):
        x = self.counts[idx: idx + self.seq_len] / self.max_val
        y = self.counts[idx + self.seq_len: idx + self.seq_len + self.horizon] / self.max_val
        return (torch.tensor(x).unsqueeze(-1),   # (seq_len, 1)
                torch.tensor(y))                  # (horizon,)


# ──────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────

def train_forecaster(cfg: dict):
    fc = cfg['forecasting']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    import pandas as pd
    df = pd.read_csv(fc['data_path'])
    counts = df['count'].values.astype(np.float32)
    logger.info(f"Loaded {len(counts)} time steps from {fc['data_path']}")

    # Train/val split (80/20)
    split = int(0.8 * len(counts))
    train_counts = counts[:split]
    val_counts = counts[split:]

    seq_len = fc['sequence_length']
    horizon = fc['forecast_horizon']

    train_ds = CrowdTimeSeriesDataset(train_counts, seq_len, horizon)
    val_ds   = CrowdTimeSeriesDataset(val_counts, seq_len, horizon)

    train_loader = DataLoader(train_ds, batch_size=fc['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=fc['batch_size'], shuffle=False)

    # Model
    from model import build_forecaster
    model = build_forecaster(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=fc['learning_rate'])
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    ckpt_path = fc['checkpoint']
    os.makedirs(os.path.dirname(ckpt_path) if os.path.dirname(ckpt_path) else '.', exist_ok=True)

    for epoch in range(fc['epochs']):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{fc['epochs']}]  "
                        f"Train: {train_loss:.6f}  Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'max_val': train_ds.max_val,
            }, ckpt_path)

    logger.info(f"Forecaster training complete. Best val loss: {best_val_loss:.6f}")
    logger.info(f"Saved to: {ckpt_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--data', type=str)
    args = p.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if args.data:
        cfg['forecasting']['data_path'] = args.data

    train_forecaster(cfg)
