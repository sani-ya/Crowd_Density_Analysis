import cv2
from ultralytics import YOLO
import torch
# Assuming your CSRNet model is in a file named csrnet_model.py
# from csrnet_model import CSRNet 

class CrowdSystem:
    def __init__(self, yolo_path, csrnet_path):
        # Load the WiderPerson YOLO model
        self.detector = YOLO(yolo_path)
        
        # Load the ShanghaiTech CSRNet model (PyTorch)
        # self.density_model = CSRNet().to('cuda')
        # self.density_model.load_state_dict(torch.load(csrnet_path))
        # self.density_model.eval()

    def process_frame(self, frame):
        # 1. Detection & Tracking (WiderPerson Progress)
        results = self.detector.track(frame, persist=True, conf=0.4, classes=[0,1,2])
        
        count = 0
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            count = len(ids)

            for box, id in zip(boxes, ids):
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 180, 0), 2)
                cv2.putText(frame, f"ID: {int(id)}", (int(box[0]), int(box[1]-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 2. UI Overlay
        cv2.putText(frame, f"People Count: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        return frame

def run_system():
    # Update these paths to your actual files
    YOLO_WEIGHTS = r"C:\miniproject2\runs\detect\wider_v1\weights\best.pt"
    CSR_WEIGHTS = r"C:\miniproject files\checkpoints\partB_best.pth"

    system = CrowdSystem(YOLO_WEIGHTS, CSR_WEIGHTS)
    cap = cv2.VideoCapture(0) # Use 0 for webcam or "video.mp4"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        output_frame = system.process_frame(frame)
        cv2.imshow("Hybrid Crowd Analysis", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_system()