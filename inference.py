import os
import time
import csv
import sys
import traceback
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
#  UI Renderer (Clean Integers & Heatmap)
# ──────────────────────────────────────────────────────────────
def render_overlay(frame, density, count, mode_str, alpha=0.4):
    h, w = frame.shape[:2]
    
    # 1. Apply Heatmap if density map exists
    if density is not None and density.max() > 0:
        d_norm = (density - density.min()) / (density.max() - density.min() + 1e-8)
        d_uint8 = (d_norm * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(d_uint8, cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap, (w, h))
        frame = cv2.addWeighted(frame, 1 - alpha, heatmap_resized, alpha, 0)

    # 2. Status Bar
    cv2.rectangle(frame, (0, 0), (w, 75), (0, 0, 0), -1)
    display_count = int(round(count))
    
    cv2.putText(frame, f"TOTAL COUNT: {display_count}", (30, 50), 
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2)
    
    # Mode Color logic
    mode_color = (0, 255, 0) if "LIVE" in mode_str else (0, 165, 255)
    cv2.putText(frame, f"[{mode_str}]", (w - 350, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
    return frame

# ──────────────────────────────────────────────────────────────
#  Main Engine
# ──────────────────────────────────────────────────────────────
def run_inference(cfg: dict):
    # --- INTERACTIVE TERMINAL MENU ---
    print("\n" + "="*50, flush=True)
    print("      CROWD ANALYTICS SYSTEM - MASTER", flush=True)
    print("="*50, flush=True)
    print(" [1] LIVE WEBCAM DEMO", flush=True)
    print(" [2] PROCESS VIDEO/IMAGE FILE", flush=True)
    print(" [3] EXIT", flush=True)
    print("="*50, flush=True)
    
    sys.stdout.flush()
    choice = input("\n>> Select Option (1-3): ").strip()

    if choice == '3': return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from model import build_model 
    
    # Load Models
    print(f"\n[INFO] Loading AI Models on {device}...", flush=True)
    model_A = build_model(cfg).to(device).eval()
    model_A.load_state_dict(torch.load(cfg['inference']['checkpoint_A'], map_location=device)['model_state'])
    yolo_model = YOLO('yolov8n.pt').to(device)

    # Source Selection
    if choice == '1':
        source = 0
        is_live = True
    elif choice == '2':
        source = input("\n>> Enter path or Drag & Drop file: ").strip().replace('"', '').replace("'", "")
        is_live = False
        if not os.path.exists(source):
            print(f"[ERROR] Path not found: {source}")
            return
    else: return

    # Check if single image
    is_image = False
    if not is_live and isinstance(source, str):
        is_image = any(source.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp'])

    cap = cv2.VideoCapture(source)
    log_file = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    target_size = tuple(cfg['dataset']['image_size'])
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=cfg['dataset']['normalize_mean'], std=cfg['dataset']['normalize_std'])])

    print(f"\n[RUNNING] Window active. CSV logging to: {log_file}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # --- FUSED SPATIAL MASKING (APPROACH A) ---
            h_orig, w_orig = frame.shape[:2]
            # Split the screen horizontally at 50%
            split_y = int(h_orig * 0.5)

            # 1. YOLO for the Foreground (Bottom Half)
            yolo_results = yolo_model(frame, verbose=False, conf=0.45)[0]
            person_boxes = []
            for box in yolo_results.boxes:
                if int(box.cls) == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    # Use center Y of the bounding box to check if it's in the bottom half
                    cy = (y1 + y2) / 2
                    if cy >= split_y:
                        person_boxes.append((x1, y1, x2, y2))

            # 2. CSRNet for the Background (Top Half)
            img_input = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), target_size)
            input_tensor = transform(img_input).unsqueeze(0).to(device)
            with torch.no_grad():
                output_A = model_A(input_tensor)[0, 0].cpu().numpy()
            
            # Zero out the bottom half of the density map so we don't double count
            d_h, d_w = output_A.shape
            d_split_y = int(d_h * (split_y / h_orig))
            output_A[d_split_y:, :] = 0.0

            # 3. Combine counts
            csrnet_count = float(output_A.sum())
            yolo_count = float(len(person_boxes))
            final_count = csrnet_count + yolo_count
            
            final_density = output_A
            mode_str = "FILE FUSED" if not is_live else "LIVE FUSED"

            # Draw YOLO boxes
            for (x1, y1, x2, y2) in person_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw visual split line
            cv2.line(frame, (0, split_y), (w_orig, split_y), (255, 0, 255), 2)

            # CSV Logging
            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([datetime.now().strftime("%H:%M:%S"), mode_str, int(round(final_count))])

            # RENDER DISPLAY
            frame_out = render_overlay(frame, final_density, final_count, mode_str)
            cv2.imshow('Project Analysis Dashboard', frame_out)

            # --- THE "STAY OPEN" FIX ---
            if is_image:
                print(f"\n[COMPLETE] Counted {int(round(final_count))} people.")
                print(">> CLICK the Image Window and PRESS ANY KEY to close.")
                cv2.waitKey(0) # Waits forever
                break

            # Press Q to quit live stream or video
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[DONE] System closed. Data at: {log_file}")

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    run_inference(cfg)