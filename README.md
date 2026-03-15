# Hybrid Crowd Density Analysis & Prediction System

A high-performance crowd analytics pipeline combining **YOLOv8** (for sparse crowds) and **CSRNet** (for dense crowds) using an **Adaptive Weighted Average** fusion algorithm.

---

## 🌟 Features
- **Hybrid Fusion**: Decisive switching between YOLOv8 and CSRNet based on scene density.
- **Adaptive Weighting**: 100% YOLO for sparse scenes; 80% CSRNet + 20% YOLO for dense crowds.
- **Real-time Performance**: Optimized with frame skipping and sub-sampling for edge-level speeds.
- **Dual Interface**: Choice between a **Native Desktop Window** (fastest) or a **Web Dashboard** (beautiful UI).
- **Dataset Agnostic**: Easy training on ShanghaiTech, UCF-QNRF, or custom data.

---

## 🚀 How to Run

### Method 1: Desktop Application (Fastest)
Use this for real-time CCTV monitoring or processing large video files with maximum FPS.

```bash
# Activation of environment (if applicable)
# .\venv\Scripts\Activate.ps1

python run_app.py
```
- **Option 1**: Live Webcam
- **Option 2**: Process Video/Image File
- **Controls**: `Q` to Quit, `S` for Screenshot.

---

### Method 2: Web Dashboard (Beautiful UI)
Use this for a modern, browser-based analytics interface.

**Step 1: Start the Backend API**
```bash
python api.py
```

**Step 2: Start the Frontend Server (In a new terminal)**
```bash
cd frontend
python -m http.server 8000
```

**Step 3: Open in Browser**
Go to: [**http://localhost:8000**](http://localhost:8000)

---

## 📂 Project Structure

```
.
├── run_app.py               ← Recommended: Fast Native Desktop App
├── api.py                   ← Backend API for Web Interface
├── frontend/                ← Web Dashboard UI (HTML/JS/CSS)
├── config.yaml              ← ALL configuration (model paths, thresholds)
├── train.py                 ← Model Training Script
├── dataset.py               ← Dataset loader (MAT/JSON/CSV)
├── model.py                 ← CSRNet architecture
├── yolov8n.pt               ← YOLOv8 pre-trained weights
├── checkpoints/             ← Your trained model weights (e.g., best_model.pth)
└── logs/                    ← Training and inference report logs
```

---

## 🛠️ Training

### 1. Configure the dataset
Update `config.yaml` with your dataset paths:
```yaml
dataset:
  root: "./data/ShanghaiTech/part_A"
  annotation_format: "mat"
```

### 2. Run Training
```bash
python train.py --config config.yaml
```
Your best weights will be saved to `checkpoints/best_model.pth`.

---

## ⚙️ Configuration
The system is controlled via `config.yaml`. Key parameters:
- `device`: Set to `auto`, `cuda`, or `cpu`.
- `checkpoint_A`: Path to your trained CSRNet weights.
- `checkpoint_yolo`: Path to YOLOv8 weights (default: `yolov8n.pt`).
- `DENSITY_THRESHOLD`: (Inside `run_app.py`) The YOLO count where CSRNet starts contributing (Default: 15).

---

## 📊 Performance Benchmarks
| Mode | Target Hardware | Expected FPS |
|---------|-------------|-----|
| Desktop App | CPU (i5/i7) | 8 - 15 FPS |
| Desktop App | GPU (RTX 3060+) | 30+ FPS |
| Web Dashboard | Localhost | 5 - 10 FPS |

---

## 📜 Privacy & Ethics
- **Identity-Safe**: No facial recognition or biometric tracking.
- **Aggregate Only**: Estimates total density without identifying individuals.
- **Local First**: Data is processed locally; no images are uploaded to external clouds.

---
*Created for B.Tech Mini-Project - Hybrid Crowd Density Analysis.*
