import cv2
from ultralytics import YOLO
from utils import Config, draw_text

class PersonDetector:
    def __init__(self):
        self.model = YOLO(Config.YOLO_MODEL)
        self.conf = Config.CONFIDENCE_THRESHOLD

    def detect(self, frame):
        results = self.model(frame, verbose=False, conf=self.conf)
        detections = []
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    if int(box.cls[0]) == Config.PERSON_CLASS_ID:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detections.append([x1, y1, x2, y2, float(box.conf[0])])
        return detections