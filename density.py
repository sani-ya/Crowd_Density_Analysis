import cv2
import numpy as np
from utils import Config, draw_text

class DensityEstimator:
    @staticmethod
    def classify(count):
        if count <= Config.LOW_MAX: return "Low", Config.COLOR_LOW
        elif count <= Config.MEDIUM_MAX: return "Medium", Config.COLOR_MEDIUM
        else: return "High", Config.COLOR_HIGH

    @staticmethod
    def draw_density_badge(frame, count, level, color):
        h, w = frame.shape[:2]
        badge_w, badge_h = 250, 80
        x1, y1 = w - badge_w - 10, 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + badge_w, y1 + badge_h), color, -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x1 + badge_w, y1 + badge_h), color, 2)
        draw_text(frame, f"Count: {count}", (x1+10, y1+30), scale=0.6, thickness=2)
        draw_text(frame, f"Density: {level}", (x1+10, y1+60), scale=0.6, thickness=2)