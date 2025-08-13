import cv2
from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch.nn.functional as F
import mediapipe as mp

# Setup device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Load emotion detection model
try:
    model_name = "prithivMLmods/Facial-Emotion-Detection-SigLIP2"
    model = SiglipForImageClassification.from_pretrained(model_name).to(device)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

labels = {
    0: "Ahegao", 1: "Angry", 2: "Happy", 3: "Neutral",
    4: "Sad", 5: "Surprise"
}

def detect_faces_mediapipe(frame):
    """Detect faces using MediaPipe and return bounding boxes"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    faces = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            
            # Convert relative coordinates to absolute coordinates
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            faces.append((x, y, width, height))
    
    return faces

def crop_face(frame, face_coords, padding_ratio=0.2):
    """Crop face from frame with proportional padding"""
    x, y, w, h = face_coords
    
    # Calculate padding based on face size
    padding_x = int(w * padding_ratio)
    padding_y = int(h * padding_ratio)
    
    # Apply padding
    x_start = max(0, x - padding_x)
    y_start = max(0, y - padding_y)
    x_end = min(frame.shape[1], x + w + padding_x)
    y_end = min(frame.shape[0], y + h + padding_y)
    
    cropped_face = frame[y_start:y_end, x_start:x_end]
    return cropped_face

def emotion_classification(image: np.ndarray):
    """Classify emotion from cropped face image"""
    try:
        if image.size == 0:
            return "No Face", 0.0, {}
            
        pil_img = Image.fromarray(image).convert("RGB")
        inputs = processor(images=pil_img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze()
            
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            
            probs = probs.cpu().tolist()

        predictions = {labels[i]: round(probs[i], 3) for i in range(len(probs))}
        top_label = max(predictions, key=predictions.get)
        top_score = predictions[top_label]
        return top_label, top_score, predictions
    
    except Exception as e:
        print(f"Error in emotion classification: {e}")
        return "Error", 0.0, {}

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_count = 0
skip_frames = 3  # MediaPipe is more efficient, so we can process more frequently

face_emotions = {}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Detect faces
        faces = detect_faces_mediapipe(frame)
        
        # Process emotions every nth frame
        if frame_count % skip_frames == 0:
            face_emotions = {}
            
            for i, face in enumerate(faces):
                # Crop face from RGB frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cropped_face = crop_face(rgb_frame, face)
                
                # Classify emotion
                label, score, _ = emotion_classification(cropped_face)
                face_emotions[i] = (label, score)

        # Draw results
        for i, (x, y, w, h) in enumerate(faces):
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display emotion
            if i in face_emotions:
                label, score = face_emotions[i]
                display_text = f"{label}: {score:.2f}"
            else:
                display_text = "Detecting..."
            
            text_y = max(y - 10, 20)
            cv2.putText(frame, display_text, (x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Show frame info
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Facial Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
    if device.type == 'cuda':
        torch.cuda.empty_cache()