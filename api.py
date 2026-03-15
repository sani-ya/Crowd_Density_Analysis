import os
import cv2
import yaml
import torch
import time
import torchvision.transforms as T
import numpy as np
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
import base64

app = Flask(__name__)
CORS(app)

# Load configuration and models
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

device_str = cfg['system'].get('device', 'cpu')
if device_str == 'auto':
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

device = torch.device(device_str)

from model import build_model
model_A = build_model(cfg).to(device).eval()
try:
    model_A.load_state_dict(torch.load(cfg['inference']['checkpoint_A'], map_location=device)['model_state'])
except FileNotFoundError:
    print("[WARNING] Could not load CSRNet checkpoint for API.")

yolo_model = YOLO(cfg['inference']['checkpoint_yolo']).to(device)

target_size = tuple(cfg['dataset']['image_size'])
transform = T.Compose([
    T.ToTensor(), 
    T.Normalize(mean=cfg['dataset']['normalize_mean'], std=cfg['dataset']['normalize_std'])
])

# Global state
live_feed_active = False
camera = None
frame_count = 0

latest_stats = {
    "count": 0,
    "density": "Normal",
    "anomaly": False,
    "conf": 100,
    "mode": "SPARSE",
    "timestamp": 0
}

DENSITY_THRESHOLD = 15

def fuse_counts(yolo_count, csrnet_count, density_available):
    if yolo_count == 0 and density_available:
        final = 0.20 * yolo_count + 0.80 * csrnet_count
        return final, 0.20, 0.80, "DENSE (YOLO=0)"
    elif yolo_count < DENSITY_THRESHOLD or not density_available:
        return yolo_count, 1.0, 0.0, "SPARSE"
    else:
        final = 0.20 * yolo_count + 0.80 * csrnet_count
        return final, 0.20, 0.80, "DENSE"

def process_frame(frame, is_live=False):
    global frame_count, latest_stats
    h_orig, w_orig = frame.shape[:2]
    
    scale = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    frame_count += 1
    # Only process AI every 3 frames if live, otherwise it lags
    if is_live and frame_count % 3 != 0:
        return frame, latest_stats["count"]

    # YOLO Foreground
    yolo_results = yolo_model(small_frame, verbose=False, conf=0.45)[0]
    person_boxes = []
    for box in yolo_results.boxes:
        if int(box.cls) == 0:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            person_boxes.append((int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)))

    # CSRNet Background
    img_input = cv2.resize(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB), target_size)
    input_tensor = transform(img_input).unsqueeze(0).to(device)
    with torch.no_grad():
        output_A = model_A(input_tensor)[0, 0].cpu().numpy()
    
    csrnet_count = float(output_A.sum())
    yolo_count = float(len(person_boxes))
    
    final_count, w_yolo, w_csr, density_mode = fuse_counts(yolo_count, csrnet_count, True)
    
    # Overlay logic
    for (x1, y1, x2, y2) in person_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Blend Heatmap
    if density_mode.startswith("DENSE") and output_A.max() > 0:
        d_norm = (output_A - output_A.min()) / (output_A.max() - output_A.min() + 1e-8)
        d_uint8 = (d_norm * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(d_uint8, cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap_color, (w_orig, h_orig))
        frame = cv2.addWeighted(frame, 0.6, heatmap_resized, 0.4, 0)
        
    # Stats for frontend
    density_str = 'Critical' if final_count > 150 else 'Crowded' if final_count > 60 else 'Normal'
    anomaly_bool = final_count > 180
    config_conf = int(w_csr * 100) if "DENSE" in density_mode else 100
    
    latest_stats = {
        "count": int(round(final_count)),
        "density": density_str,
        "anomaly": anomaly_bool,
        "conf": config_conf,
        "mode": density_mode,
        "yolo_c": int(yolo_count),
        "csr_c": int(round(csrnet_count)),
        "timestamp": time.time()
    }

    return frame, final_count


@app.route('/api/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    frame_out, final_count = process_frame(frame, is_live=False)

    _, buffer = cv2.imencode('.jpg', frame_out)
    b64_img = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        **latest_stats,
        'image': f'data:image/jpeg;base64,{b64_img}'
    })

def gen_frames():
    global camera
    camera = cv2.VideoCapture(0)
    # Calculate FPS
    prev_time = time.time()
    
    while live_feed_active:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_out, final_count = process_frame(frame, is_live=True)
            
            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            latest_stats["fps"] = round(fps, 1)
            
            # HUD Overlay so it appears in Stream Viewer too
            cv2.rectangle(frame_out, (0, 0), (frame_out.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(frame_out, f"COUNT: {latest_stats['count']} | MODE: {latest_stats['mode']}", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame_out)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/api/video_feed')
def video_feed():
    global live_feed_active
    live_feed_active = True
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stop_feed', methods=['POST'])
def stop_feed():
    global live_feed_active, camera
    live_feed_active = False
    if camera is not None:
        camera.release()
    return jsonify({'status': 'stopped'})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(latest_stats)

@app.route('/api/update_stats', methods=['POST'])
def update_stats():
    global latest_stats
    data = request.json
    if data:
        for k, v in data.items():
            latest_stats[k] = v
        latest_stats["timestamp"] = time.time()
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
