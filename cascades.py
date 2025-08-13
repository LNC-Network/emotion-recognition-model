import cv2
from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch.nn.functional as F

# Setup device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load face detection model (Haar Cascade) - optimized for speed
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

def detect_faces_fast(frame, scale_factor=1.2):
    """Fast face detection with optimized parameters"""
    # Convert to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize frame for faster detection, then scale back coordinates
    small_frame = cv2.resize(gray, (320, 240))
    
    faces = face_cascade.detectMultiScale(
        small_frame,
        scaleFactor=scale_factor,  # Larger scale factor = faster but less accurate
        minNeighbors=3,           # Fewer neighbors = faster but more false positives
        minSize=(20, 20),         # Smaller minimum size
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Scale coordinates back to original frame size
    scale_x = frame.shape[1] / 320
    scale_y = frame.shape[0] / 240
    
    scaled_faces = []
    for (x, y, w, h) in faces:
        scaled_faces.append((
            int(x * scale_x), int(y * scale_y),
            int(w * scale_x), int(h * scale_y)
        ))
    
    return scaled_faces

def crop_face_fast(frame, face_coords, target_size=112):
    """Fast face cropping with fixed target size"""
    x, y, w, h = face_coords
    
    # Add minimal padding
    padding = max(10, min(w, h) // 10)
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(frame.shape[1], x + w + padding)
    y_end = min(frame.shape[0], y + h + padding)
    
    cropped_face = frame[y_start:y_end, x_start:x_end]
    
    # Resize to fixed size for consistent processing
    if cropped_face.size > 0:
        cropped_face = cv2.resize(cropped_face, (target_size, target_size))
    
    return cropped_face

def emotion_classification(image: np.ndarray):
    """Fast emotion classification"""
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
        return "Error", 0.0, {}

# Initialize webcam with lower resolution for speed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Lower resolution = faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_count = 0
face_detection_skip = 3    # Detect faces every 3rd frame
emotion_detection_skip = 9  # Detect emotions every 9th frame

# Cache results
cached_faces = []
face_emotions = {}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Face detection (less frequent)
        if frame_count % face_detection_skip == 0:
            cached_faces = detect_faces_fast(frame, scale_factor=1.3)
        
        # Emotion detection (even less frequent)
        if frame_count % emotion_detection_skip == 0 and cached_faces:
            face_emotions = {}
            
            # Only process the largest face for speed (single face mode)
            if cached_faces:
                # Find largest face
                largest_face = max(cached_faces, key=lambda face: face[2] * face[3])
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cropped_face = crop_face_fast(rgb_frame, largest_face)
                
                if cropped_face.size > 0:
                    label, score, _ = emotion_classification(cropped_face)
                    face_emotions[0] = (label, score)

        # Draw results for all cached faces
        for i, (x, y, w, h) in enumerate(cached_faces):
            # Draw bounding box
            color = (0, 255, 0) if i == 0 else (100, 100, 100)  # Highlight main face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Display emotion (only for main face to keep it simple)
            if i == 0 and i in face_emotions:
                label, score = face_emotions[i]
                display_text = f"{label}: {score:.2f}"
                text_y = max(y - 10, 20)
                cv2.putText(frame, display_text, (x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Show FPS counter
        fps_text = f"Faces: {len(cached_faces)} | Frame: {frame_count}"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Fast Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
    if device.type == 'cuda':
        torch.cuda.empty_cache()