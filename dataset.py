"""
dataset.py — Dataset-agnostic crowd counting dataset.

Supports:
  - ShanghaiTech (Part A / Part B)
  - UCF_CC_50
  - UCF-QNRF
  - WorldExpo'10
  - Any custom dataset via CSV / JSON / MAT annotations

To add a NEW dataset: see README section "Adding a New Dataset".
"""

import os
import json
import glob
import logging
import numpy as np
import cv2
import scipy.io as sio
import scipy.ndimage
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Density Map Generator
# ──────────────────────────────────────────────────────────────

class DensityMapGenerator:
    """
    Converts raw point annotations (head positions) into Gaussian density maps.

    Two modes:
      - Fixed sigma: classic Gaussian, good for sparse crowds.
      - Adaptive sigma: sigma ∝ average distance to k nearest neighbours.
        Used in ShanghaiTech papers for dense/occluded scenes.
    """

    def __init__(self, sigma: float = 15.0, adaptive: bool = True, k: int = 3):
        self.sigma = sigma
        self.adaptive = adaptive
        self.k = k

    def generate(self, points: np.ndarray, map_h: int, map_w: int,
                 orig_h: int, orig_w: int) -> np.ndarray:
        """
        Args:
            points    : (N, 2) array of [col, row] head annotations in ORIGINAL image coords.
            map_h/w   : Output density map size (typically image_size / output_stride).
            orig_h/w  : Original image size (used to scale point coords).
        Returns:
            density   : (map_h, map_w) float32 density map.
                        Integral ≈ total head count.
        """
        density = np.zeros((map_h, map_w), dtype=np.float32)

        if len(points) == 0:
            return density

        # Scale points from original image space to density map space
        scale_x = map_w / orig_w
        scale_y = map_h / orig_h
        pts = points.copy().astype(np.float32)
        pts[:, 0] *= scale_x   # col
        pts[:, 1] *= scale_y   # row

        # Clamp to valid range
        pts[:, 0] = np.clip(pts[:, 0], 0, map_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, map_h - 1)

        for idx, pt in enumerate(pts):
            col, row = int(pt[0]), int(pt[1])

            if self.adaptive and len(pts) > self.k:
                sigma = self._adaptive_sigma(pt, pts, idx)
            else:
                sigma = self.sigma

            # Place a Gaussian blob at this point
            density = self._add_gaussian(density, row, col, sigma)

        return density

    def _adaptive_sigma(self, pt: np.ndarray, all_pts: np.ndarray,
                        idx: int) -> float:
        """Compute sigma as beta * average_distance_to_k_nearest_neighbours."""
        diffs = all_pts - pt
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        dists[idx] = np.inf           # exclude self
        nearest = np.sort(dists)[:self.k]
        avg_dist = nearest.mean()
        return max(avg_dist * 0.3, 1.0)   # beta = 0.3

    @staticmethod
    def _add_gaussian(density: np.ndarray, row: int, col: int,
                      sigma: float) -> np.ndarray:
        """Add a 2-D Gaussian to density map, cropped to boundaries."""
        h, w = density.shape
        # Kernel radius: 3σ on each side
        radius = max(1, int(3 * sigma))
        r0, r1 = max(0, row - radius), min(h, row + radius + 1)
        c0, c1 = max(0, col - radius), min(w, col + radius + 1)

        # Local coordinate grid
        y = np.arange(r0, r1) - row
        x = np.arange(c0, c1) - col
        xx, yy = np.meshgrid(x, y)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))

        # Normalise so sum = 1 (one head)
        k_sum = kernel.sum()
        if k_sum > 0:
            kernel /= k_sum

        density[r0:r1, c0:c1] += kernel
        return density


# ──────────────────────────────────────────────────────────────
#  Annotation Parsers
# ──────────────────────────────────────────────────────────────

def parse_mat_annotation(ann_path: str) -> np.ndarray:
    """
    Parse MATLAB .mat annotation (ShanghaiTech / UCF_CC_50 format).
    Returns (N, 2) array of [col, row].
    """
    mat = sio.loadmat(ann_path)
    # ShanghaiTech key: 'image_info' → nested structure
    if 'image_info' in mat:
        pts = mat['image_info'][0, 0][0, 0][0]   # shape (N, 2)
    # UCF_CC_50 key: 'annPoints'
    elif 'annPoints' in mat:
        pts = mat['annPoints']
    # Generic fallback: find first (N,2) numeric array
    else:
        pts = None
        for v in mat.values():
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 2:
                pts = v
                break
        if pts is None:
            raise ValueError(f"Cannot parse MAT file: {ann_path}")
    return pts.astype(np.float32)


def parse_csv_annotation(ann_path: str) -> np.ndarray:
    """
    Parse CSV annotation.
    Expected columns: x,y  OR  col,row  (header optional)
    Returns (N, 2) array of [col, row].
    """
    import csv
    points = []
    with open(ann_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0 and not row[0].replace('.', '').lstrip('-').isdigit():
                continue   # skip header
            if len(row) >= 2:
                points.append([float(row[0]), float(row[1])])
    return np.array(points, dtype=np.float32) if points else np.zeros((0, 2), dtype=np.float32)


def parse_json_annotation(ann_path: str) -> np.ndarray:
    """
    Parse JSON annotation.
    Supported schemas:
      {"points": [[x, y], ...]}
      [{"x": ..., "y": ...}, ...]
    Returns (N, 2) array of [col, row].
    """
    with open(ann_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'points' in data:
        pts = data['points']
    elif isinstance(data, list):
        pts = [[d.get('x', d.get('col', 0)), d.get('y', d.get('row', 0))] for d in data]
    else:
        raise ValueError(f"Unsupported JSON annotation format: {ann_path}")

    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 2), dtype=np.float32)


ANNOTATION_PARSERS = {
    'mat':  parse_mat_annotation,
    'csv':  parse_csv_annotation,
    'json': parse_json_annotation,
}


# ──────────────────────────────────────────────────────────────
#  Main Dataset Class
# ──────────────────────────────────────────────────────────────

class CrowdDataset(Dataset):
    """
    Dataset-agnostic crowd counting dataset.

    Works with any dataset by specifying:
      - image_dir   : folder of images
      - ann_dir     : folder of annotations
      - ann_format  : 'mat' | 'csv' | 'json'
      - ann_suffix  : annotation filename suffix (e.g. '_ann.mat')
      - img_suffix  : image filename suffix (e.g. '.jpg')

    The annotation file for an image is assumed to follow:
        {image_stem}{ann_suffix}.{ann_format}
    e.g. image IMG_001.jpg → annotation GT_IMG_001.mat
    """

    def __init__(
        self,
        image_dir: str,
        ann_dir: str,
        ann_format: str = 'mat',
        image_size: Tuple[int, int] = (512, 512),
        output_stride: int = 8,
        density_sigma: float = 15.0,
        adaptive_sigma: bool = True,
        k_nearest: int = 3,
        augment: bool = False,
        aug_config: Optional[Dict] = None,
        normalize_mean: List[float] = (0.485, 0.456, 0.406),
        normalize_std: List[float] = (0.229, 0.224, 0.225),
        ann_prefix: str = 'GT_',      # prefix added to image stem for annotation
        ann_suffix: str = '',         # suffix added to image stem for annotation
        img_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp'),
    ):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.ann_dir = Path(ann_dir)
        self.ann_format = ann_format.lower()
        self.image_size = image_size          # (H, W)
        self.output_stride = output_stride
        self.augment = augment
        self.aug_config = aug_config or {}
        self.ann_prefix = ann_prefix
        self.ann_suffix = ann_suffix

        assert self.ann_format in ANNOTATION_PARSERS, \
            f"ann_format must be one of {list(ANNOTATION_PARSERS.keys())}"

        self.density_gen = DensityMapGenerator(
            sigma=density_sigma,
            adaptive=adaptive_sigma,
            k=k_nearest,
        )

        # Normalisation transform
        self.normalize = T.Normalize(mean=normalize_mean, std=normalize_std)

        # Gather image paths
        self.image_paths = []
        for ext in img_extensions:
            self.image_paths.extend(sorted(self.image_dir.glob(f'*{ext}')))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {image_dir}")

        logger.info(f"[CrowdDataset] Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _annotation_path(self, img_path: Path) -> Path:
        """Derive annotation path from image path."""
        stem = img_path.stem
        ann_name = f"{self.ann_prefix}{stem}{self.ann_suffix}.{self.ann_format}"
        return self.ann_dir / ann_name

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        ann_path = self._annotation_path(img_path)

        # ── Load image ──────────────────────────────────────
        img = cv2.imread(str(img_path))
        if img is None:
            raise IOError(f"Could not load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # ── Load annotation ──────────────────────────────────
        if ann_path.exists():
            parser = ANNOTATION_PARSERS[self.ann_format]
            points = parser(str(ann_path))
        else:
            logger.warning(f"Annotation not found: {ann_path}. Using empty.")
            points = np.zeros((0, 2), dtype=np.float32)

        # ── Augmentation (on raw image + points) ─────────────
        if self.augment:
            img, points = self._augment(img, points, orig_h, orig_w)
            orig_h, orig_w = img.shape[:2]

        # ── Resize image ──────────────────────────────────────
        H, W = self.image_size
        img_resized = cv2.resize(img, (W, H))

        # ── Generate density map ──────────────────────────────
        map_h = H // self.output_stride
        map_w = W // self.output_stride
        density = self.density_gen.generate(points, map_h, map_w, orig_h, orig_w)

        # ── To tensors ────────────────────────────────────────
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        img_tensor = self.normalize(img_tensor)
        density_tensor = torch.from_numpy(density).unsqueeze(0).float()

        count = float(density_tensor.sum().item())   # predicted count = sum of density

        return {
            'image':   img_tensor,          # (3, H, W)
            'density': density_tensor,      # (1, H/s, W/s)
            'count':   torch.tensor(count, dtype=torch.float32),
            'path':    str(img_path),
        }

    def _augment(self, img: np.ndarray, points: np.ndarray,
                 h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentations to image and point annotations simultaneously."""
        # Random horizontal flip
        if self.aug_config.get('horizontal_flip', True) and np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()
            if len(points):
                points = points.copy()
                points[:, 0] = w - 1 - points[:, 0]

        # Random crop
        if self.aug_config.get('random_crop', True):
            ch, cw = self.aug_config.get('crop_size', [400, 400])
            ch, cw = min(ch, h), min(cw, w)
            if h > ch and w > cw:
                r0 = np.random.randint(0, h - ch)
                c0 = np.random.randint(0, w - cw)
                img = img[r0:r0 + ch, c0:c0 + cw]
                if len(points):
                    points = points.copy()
                    points[:, 0] -= c0
                    points[:, 1] -= r0
                    # Keep only points inside crop
                    mask = ((points[:, 0] >= 0) & (points[:, 0] < cw) &
                            (points[:, 1] >= 0) & (points[:, 1] < ch))
                    points = points[mask]

        # Color jitter (image only)
        if self.aug_config.get('color_jitter', True):
            alpha = 1.0 + (np.random.rand() - 0.5) * self.aug_config.get('brightness', 0.2) * 2
            beta  = 1.0 + (np.random.rand() - 0.5) * self.aug_config.get('contrast', 0.2) * 2
            img = np.clip(img.astype(np.float32) * alpha * beta, 0, 255).astype(np.uint8)

        return img, points


# ──────────────────────────────────────────────────────────────
#  DataLoader Factory
# ──────────────────────────────────────────────────────────────

def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders from config dict."""
    ds_cfg = cfg['dataset']
    root = Path(ds_cfg['root'])

    train_ds = CrowdDataset(
        image_dir=str(root / ds_cfg['train_image_dir']),
        ann_dir=str(root / ds_cfg['train_annotation_dir']),
        ann_format=ds_cfg['annotation_format'],
        image_size=tuple(ds_cfg['image_size']),
        output_stride=ds_cfg['output_stride'],
        density_sigma=ds_cfg['density_sigma'],
        adaptive_sigma=ds_cfg['adaptive_sigma'],
        k_nearest=ds_cfg['k_nearest'],
        augment=True,
        aug_config=ds_cfg.get('augmentation', {}),
        normalize_mean=ds_cfg['normalize_mean'],
        normalize_std=ds_cfg['normalize_std'],
    )

    val_ds = CrowdDataset(
        image_dir=str(root / ds_cfg['val_image_dir']),
        ann_dir=str(root / ds_cfg['val_annotation_dir']),
        ann_format=ds_cfg['annotation_format'],
        image_size=tuple(ds_cfg['image_size']),
        output_stride=ds_cfg['output_stride'],
        density_sigma=ds_cfg['density_sigma'],
        adaptive_sigma=ds_cfg['adaptive_sigma'],
        k_nearest=ds_cfg['k_nearest'],
        augment=False,
        normalize_mean=ds_cfg['normalize_mean'],
        normalize_std=ds_cfg['normalize_std'],
    )

    sys_cfg = cfg.get('system', {})
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=sys_cfg.get('num_workers', 4),
        pin_memory=sys_cfg.get('pin_memory', True),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['evaluation']['batch_size'],
        shuffle=False,
        num_workers=sys_cfg.get('num_workers', 4),
        pin_memory=sys_cfg.get('pin_memory', True),
    )

    return train_loader, val_loader
