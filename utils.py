import csv
import os
import time
from datetime import datetime
import cv2

class Config:
    # BASE_PATH set to current directory to avoid Windows C:\ permissions issues
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(BASE_PATH, "csv_logs")
    
    # --- UPDATED FOR WIDERPERSON ---
    # Once training is done, change this to your 'best.pt' path
    # Example: r"C:\miniproject2\runs\detect\wider_v1\weights\best.pt"
    YOLO_MODEL = "yolov8n.pt" 
    
    # WiderPerson usually has these class IDs:
    # 0: Pedestrians, 1: Riders, 2: Partially-visible, 3: Ignore, 4: Crowd
    # We want to track IDs 0, 1, and 2 as "People"
    TARGET_CLASSES = [0, 1, 2]
    
    CONFIDENCE_THRESHOLD = 0.40  # Lowered slightly as WiderPerson has small targets
    
    # --- DENSITY THRESHOLDS ---
    LOW_MAX, MEDIUM_MAX = 10, 25
    COLOR_LOW = (0, 200, 0)      # Green
    COLOR_MEDIUM = (0, 220, 255) # Yellow
    COLOR_HIGH = (0, 0, 255)     # Red
    
    # --- UI SETTINGS ---
    WINDOW_NAME = "WiderPerson Crowd Monitor"
    BBOX_COLOR = (255, 180, 0)   # Cyan-ish
    ID_COLOR = (255, 255, 255)   # White

class CSVLogger:
    def __init__(self):
        if not os.path.exists(Config.LOG_DIR):
            try:
                os.makedirs(Config.LOG_DIR, exist_ok=True)
            except Exception as e:
                print(f"Folder Error: {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(Config.LOG_DIR, f"report_{timestamp}.csv")
        self._last_log_time = 0.0
        
        try:
            with open(self.filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "count", "density"])
            print(f"--- LOG FILE CREATED: {self.filepath} ---")
        except Exception as e:
            print(f"--- CRITICAL WRITE ERROR: {e} ---")

    def log(self, count, density_level):
        now = time.time()
        # Log once per second to prevent massive files
        if (now - self._last_log_time) < 1.0: return
        self._last_log_time = now
        
        ts = datetime.now().strftime("%H:%M:%S")
        try:
            with open(self.filepath, "a", newline="") as f:
                csv.writer(f).writerow([ts, count, density_level])
        except:
            pass

def draw_text(frame, text, pos, color=(255, 255, 255), scale=0.5, thickness=1, bg_color=None):
    if bg_color:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(frame, (pos[0], pos[1] - th - 5), (pos[0] + tw, pos[1] + 5), bg_color, -1)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

class FPSCounter:
    def __init__(self, smoothing=0.9):
        self._smoothing, self._prev_time, self._fps = smoothing, time.time(), 0.0
    def tick(self):
        now = time.time()
        dt = max(now - self._prev_time, 1e-6)
        self._prev_time = now
        self._fps = self._smoothing * self._fps + (1 - self._smoothing) * (1.0 / dt)
        return self._fps