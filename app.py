import cv2
import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from model import CSRNet

# --- SETTINGS ---
CHECKPOINT = 'checkpoints/best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    model = CSRNet().to(device)
    if os.path.exists(CHECKPOINT):
        # Load the whole dictionary
        checkpoint_data = torch.load(CHECKPOINT, map_location=device)
        
        # Extract ONLY the weights using the 'model_state' key
        model.load_state_dict(checkpoint_data['model_state'])
        
        model.eval()
        print(f"\n[SUCCESS] Weights extracted from {CHECKPOINT}")
        print(f"[INFO] Model was trained for {checkpoint_data['epoch']} epochs.")
        print(f"[INFO] Best MAE achieved during training: {checkpoint_data['best_mae']:.2f}")
        return model
    else:
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT}")

def process_frame(model, frame, transform):
    # Convert OpenCV BGR to RGB for the model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
    
    count = torch.sum(output).item()
    
    # Generate Heatmap
    heatmap = output.detach().cpu().numpy()[0][0]
    # Normalize 0-255
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-5)
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (frame.shape[1], frame.shape[0]))
    
    # Blend: 60% original image, 40% heatmap
    result = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
    return result, int(round(count))

def main():
    print("="*40)
    print(" CSRNet CROWD ANALYSIS SYSTEM ")
    print("="*40)
    
    try:
        model = load_model()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("\nSelect Mode:")
    print("1: Real-time Webcam")
    print("2: Specific Image File")
    choice = input("\nEnter choice (1 or 2): ")

    if choice == '1':
        cap = cv2.VideoCapture(0)
        print("\nStarting Webcam... Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            display_frame, count = process_frame(model, frame, transform)
            cv2.putText(display_frame, f"Live Count: {count}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Crowd Analysis - LIVE", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()

    elif choice == '2':
        img_path = input("Enter image path (e.g., data/ShanghaiTech/part_A/test_data/images/IMG_1.jpg): ")
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"[ERROR] Could not find image at {img_path}")
            return
        
        print("Analyzing...")
        display_frame, count = process_frame(model, frame, transform)
        
        # Display window
        cv2.putText(display_frame, f"Estimated Count: {count}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Crowd Analysis - STATIC", display_frame)
        print(f"\n[RESULT] Estimated People: {count}")
        print("Press any key on the image window to close.")
        cv2.waitKey(0)

    else:
        print("Invalid choice. Exiting.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()