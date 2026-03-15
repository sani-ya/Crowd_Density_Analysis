"""
run_app.py  ─  Fast Hybrid Crowd Analysis (Native OpenCV Window)
================================================================
Approach: Adaptive Weighted Average Fusion
  Both YOLO and CSRNet run on the FULL frame every time.
  The final count is a weighted blend of both:
    • Sparse crowd  (YOLO < SPARSE_THRESHOLD) → YOLO weight 80%, CSRNet 20%
    • Dense crowd   (YOLO > DENSE_THRESHOLD)  → YOLO weight 20%, CSRNet 80%
    • In between → smooth linear interpolation of weights

Rationale:
  YOLO is highly accurate for individuals but misses overlapping people.
  CSRNet estimates density blobs and is better for dense crowds.
  By dynamically shifting trust between them based on scene density,
  we get the best of both worlds in one single output count.

Speed optimisations
  • YOLO runs every frame at 50 % resolution  (very fast)
  • CSRNet runs every Nth frame, result is cached between runs
  • No Flask / HTTP overhead — single OpenCV window

Controls:
  Q  →  Quit
  S  →  Save current frame as PNG
"""

import os, sys, csv, time
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml
import json
import urllib.request
import threading
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  SPEED SETTINGS  (tweak these if still slow)
# ─────────────────────────────────────────────
YOLO_SCALE      = 0.5   # Run YOLO on half-resolution (0.5 = 50%)
CSRNET_SKIP     = 5     # Run CSRNet only every N frames (5 = 5× faster)

# ─────────────────────────────────────────────
#  ADAPTIVE WEIGHT THRESHOLD
# ─────────────────────────────────────────────
# Below this YOLO count  → output = YOLO only   (sparse crowd, YOLO is perfect)
# At or above this count → output = 20% YOLO + 80% CSRNet  (dense crowd, CSRNet sees what YOLO misses)
DENSITY_THRESHOLD = 15

# ─────────────────────────────────────────────

def load_models(cfg, device):
    from model import build_model
    print(f"[INFO] Loading CSRNet on {device}...", flush=True)
    csrnet = build_model(cfg).to(device).eval()
    ckpt = cfg['inference']['checkpoint_A']
    if os.path.exists(ckpt):
        csrnet.load_state_dict(torch.load(ckpt, map_location=device)['model_state'])
        print(f"[OK]   CSRNet loaded from {ckpt}", flush=True)
    else:
        print(f"[WARN] CSRNet checkpoint not found at {ckpt}. Count will be YOLO-only.", flush=True)
        csrnet = None

    print("[INFO] Loading YOLO...", flush=True)
    yolo = YOLO('yolov8n.pt')
    print("[OK]   YOLO loaded.", flush=True)
    return csrnet, yolo


def build_transform(cfg):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=cfg['dataset']['normalize_mean'],
            std=cfg['dataset']['normalize_std']
        )
    ])


def run_csrnet(model, frame, target_size, transform, device):
    """Run CSRNet on the full frame, return density map."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, target_size)
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        density = model(tensor)[0, 0].cpu().numpy()
    return density


def fuse_counts(yolo_count, csrnet_count, density_available):
    """
    Sharp two-mode fusion:
      SPARSE (yolo_count < 15):
        → Trust YOLO 100%. CSRNet is ignored.
          Reason: YOLO is highly accurate at finding individual people.

      DENSE (yolo_count >= 15):
        → Output = 0.20 * yolo_count + 0.80 * csrnet_count
          Reason: YOLO starts missing heavily overlapping bodies.
          CSRNet's density map is far more reliable in packed scenes.

    Returns: (final_count, w_yolo, w_csr, mode_label)
    """
    if yolo_count == 0 and density_available:
        # SUPER DENSE MODE — YOLO completely fails to detect people due to extreme density/occlusion
        final = 0.20 * yolo_count + 0.80 * csrnet_count
        return final, 0.20, 0.80, "DENSE (YOLO=0)"
    elif yolo_count < DENSITY_THRESHOLD or not density_available:
        # SPARSE MODE — YOLO wins outright
        return yolo_count, 1.0, 0.0, "SPARSE"
    else:
        # DENSE MODE — CSRNet takes majority control
        final = 0.20 * yolo_count + 0.80 * csrnet_count
        return final, 0.20, 0.80, "DENSE"


def process(frame, csrnet, yolo, transform, target_size, device,
            cached_density, frame_idx):
    h, w = frame.shape[:2]

    # ── 1. YOLO on FULL frame (every frame, half-res for speed) ─
    small   = cv2.resize(frame, (0, 0), fx=YOLO_SCALE, fy=YOLO_SCALE)
    results = yolo(small, verbose=False, conf=0.45)[0]
    person_boxes = []
    for box in results.boxes:
        if int(box.cls) == 0:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            # Scale back to original resolution
            person_boxes.append((
                int(x1 / YOLO_SCALE), int(y1 / YOLO_SCALE),
                int(x2 / YOLO_SCALE), int(y2 / YOLO_SCALE)
            ))
    yolo_count = float(len(person_boxes))

    # ── 2. CSRNet on FULL frame (every CSRNET_SKIP frames) ──────
    density = cached_density
    if csrnet is not None and (frame_idx % CSRNET_SKIP == 0):
        density = run_csrnet(csrnet, frame, target_size, transform, device)
        # No zeroing — CSRNet now works on the WHOLE frame

    # ── 3. Fuse counts with sharp threshold logic ────────────────
    csrnet_count = float(density.sum()) if density is not None else 0.0
    total_count, w_yolo, w_csr, density_mode = fuse_counts(
        yolo_count, csrnet_count, density is not None
    )

    # ── 4. Draw on frame ─────────────────────────────────────────
    out = frame.copy()

    # Heatmap only shown in DENSE mode (CSRNet is trusted)
    if density_mode == "DENSE" and density is not None and density.max() > 0:
        d_norm  = (density - density.min()) / (density.max() - density.min() + 1e-8)
        d_uint8 = (d_norm * 255).astype(np.uint8)
        hmap    = cv2.applyColorMap(d_uint8, cv2.COLORMAP_JET)
        hmap    = cv2.resize(hmap, (w, h))
        out = cv2.addWeighted(out, 0.60, hmap, 0.40, 0)

    # YOLO bounding boxes always shown
    for (x1, y1, x2, y2) in person_boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 80), 2)

    return out, total_count, yolo_count, csrnet_count, density, w_yolo, w_csr, density_mode


def draw_hud(frame, total, yolo_c, csrnet_c, fps, mode_str, w_yolo, w_csr, density_mode):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

    cv2.putText(frame, f"COUNT: {int(round(total))}",
                (20, 55), cv2.FONT_HERSHEY_DUPLEX, 1.7, (255, 255, 255), 2, cv2.LINE_AA)

    if density_mode == "SPARSE":
        sub = f"YOLO only ({int(yolo_c)} detected, sparse scene)"
        mode_color = (80, 255, 80)      # green
    else:
        sub = f"YOLO:{int(yolo_c)} x20%  +  CSRNet:{int(round(csrnet_c))} x80%  =  {int(round(total))}"
        mode_color = (80, 160, 255)     # blue

    cv2.putText(frame, sub,
                (300, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, mode_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"[{mode_str} | {density_mode}]   FPS:{fps:.1f}",
                (300, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)


def run(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csrnet, yolo = load_models(cfg, device)
    transform   = build_transform(cfg)
    target_size = tuple(cfg['dataset']['image_size'])

    # ── Menu ─────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("   HYBRID CROWD ANALYSIS  (Approach A Fusion)")
    print("="*50)
    print(" [1] LIVE WEBCAM")
    print(" [2] VIDEO / IMAGE FILE")
    print(" [3] EXIT")
    print("="*50)
    choice = input("\n>> Select (1-3): ").strip()
    if choice == '3':
        return

    if choice == '1':
        source, is_live = 0, True
    elif choice == '2':
        source = input(">> File path: ").strip().strip('"').strip("'")
        is_live = False
        if not os.path.exists(source):
            print(f"[ERROR] Not found: {source}")
            return
    else:
        return

    is_image = (not is_live and isinstance(source, str) and
                source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Cannot open source. If webcam, make sure no other app is using it.")
        return

    log_file = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(["time", "mode", "total", "yolo", "csrnet"])

    print(f"\n[RUNNING] Press Q to quit. Logging to {log_file}")

    mode_str      = "LIVE FUSED" if is_live else "FILE FUSED"
    cached_density = None
    frame_idx     = 0
    fps_start     = time.time()
    fps           = 0.0
    fps_alpha     = 0.1           # smoothing factor

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()

            out, total, yolo_c, csrnet_c, cached_density, w_yolo, w_csr, density_mode = process(
                frame, csrnet, yolo, transform, target_size, device,
                cached_density, frame_idx
            )
            frame_idx += 1

            elapsed = time.time() - t0
            fps = fps * (1 - fps_alpha) + (1.0 / max(elapsed, 1e-6)) * fps_alpha

            draw_hud(out, total, yolo_c, csrnet_c, fps, mode_str, w_yolo, w_csr, density_mode)
            cv2.imshow("Hybrid Crowd Analysis Dashboard", out)

            # Send stats to frontend (api.py)
            if frame_idx % 3 == 0:
                density_str = 'Critical' if total > 150 else 'Crowded' if total > 60 else 'Normal'
                stats = {
                    "count": int(round(total)),
                    "density": density_str,
                    "anomaly": total > 180,
                    "conf": int(w_csr * 100) if "DENSE" in density_mode else 100,
                    "mode": density_mode
                }
                def _post(s):
                    try:
                        req = urllib.request.Request("http://127.0.0.1:5000/api/update_stats", 
                                                     data=json.dumps(s).encode(),
                                                     headers={'Content-Type': 'application/json'},
                                                     method='POST')
                        urllib.request.urlopen(req, timeout=0.2)
                    except: pass
                threading.Thread(target=_post, args=(stats,), daemon=True).start()

            # Log
            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([
                    datetime.now().strftime("%H:%M:%S"),
                    mode_str,
                    int(round(total)),
                    int(yolo_c),
                    int(round(csrnet_c))
                ])

            if is_image:
                print(f"\n[RESULT] Estimated people: {int(round(total))}")
                print(">> Click the window and press ANY KEY to close.")
                cv2.waitKey(0)
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"capture_{datetime.now().strftime('%H%M%S')}.png"
                cv2.imwrite(save_path, out)
                print(f"[SAVED] {save_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[DONE] Log saved to {log_file}")


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    run(cfg)
