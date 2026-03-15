import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import Config, draw_text

class PersonTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30, n_init=3)

    def update(self, frame, detections):
        ds_detections = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            ds_detections.append(([x1, y1, x2-x1, y2-y1], conf, "person"))
        
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)
        results = []
        for track in tracks:
            if not track.is_confirmed(): continue
            l, t, r, b = track.to_ltrb()
            results.append((track.track_id, int(l), int(t), int(r), int(b)))
        return results

    @staticmethod
    def draw_tracks(frame, tracks):
        for tid, x1, y1, x2, y2 in tracks:
            cv2.rectangle(frame, (x1, y1), (x2, y2), Config.BBOX_COLOR, 2)
            draw_text(frame, f"ID {tid}", (x1, y1 - 10), color=Config.ID_COLOR, 
                      scale=0.5, thickness=1, bg_color=(50, 50, 50))