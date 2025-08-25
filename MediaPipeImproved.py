import cv2
from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch.nn.functional as F
import mediapipe as mp
from collections import defaultdict, deque
import time

class EmotionDetectionConfig:
    """Configuration class for emotion detection parameters"""
    def __init__(self):
        self.model_name = "prithivMLmods/Facial-Emotion-Detection-SigLIP2"
        self.min_detection_confidence = 0.5
        self.padding_ratio = 0.2
        self.min_face_size = 50
        self.base_skip_frames = 2
        self.max_skip_frames = 8
        self.smoothing_window = 5
        self.face_tracking_threshold = 0.3
        self.performance_check_interval = 30  # frames
        self.target_fps = 15
        
        # Filter out inappropriate emotions
        self.emotion_filter = {"Ahegao"}  # Emotions to exclude
        
        self.webcam_width = 640
        self.webcam_height = 480
        self.webcam_fps = 30

class FaceTracker:
    """Simple face tracker using IoU overlap"""
    def __init__(self, tracking_threshold=0.3):
        self.tracking_threshold = tracking_threshold
        self.tracked_faces = {}
        self.next_id = 0
        self.max_missing_frames = 10
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
            
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_tracks(self, detected_faces):
        """Update face tracks with new detections"""
        # Mark all existing tracks as not found
        for track_id in self.tracked_faces:
            self.tracked_faces[track_id]['found'] = False
            self.tracked_faces[track_id]['missing_frames'] += 1
        
        assigned_faces = []
        updated_tracks = {}
        
        # Try to match each detected face with existing tracks
        for face in detected_faces:
            best_match_id = None
            best_iou = 0
            
            for track_id, track_data in self.tracked_faces.items():
                if track_data['missing_frames'] > self.max_missing_frames:
                    continue
                    
                iou = self.calculate_iou(face, track_data['bbox'])
                if iou > best_iou and iou > self.tracking_threshold:
                    best_iou = iou
                    best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                self.tracked_faces[best_match_id]['bbox'] = face
                self.tracked_faces[best_match_id]['found'] = True
                self.tracked_faces[best_match_id]['missing_frames'] = 0
                updated_tracks[best_match_id] = face
                assigned_faces.append(face)
            else:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                self.tracked_faces[track_id] = {
                    'bbox': face,
                    'found': True,
                    'missing_frames': 0
                }
                updated_tracks[track_id] = face
        
        # Remove tracks that have been missing for too long
        to_remove = [track_id for track_id, track_data in self.tracked_faces.items() 
                    if track_data['missing_frames'] > self.max_missing_frames]
        for track_id in to_remove:
            del self.tracked_faces[track_id]
        
        return updated_tracks

class EmotionSmoother:
    """Smooth emotion predictions over time"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.emotion_history = defaultdict(lambda: deque(maxlen=window_size))
    
    def add_prediction(self, track_id, emotion, confidence):
        """Add new prediction for a tracked face"""
        self.emotion_history[track_id].append((emotion, confidence))
    
    def get_smoothed_emotion(self, track_id):
        """Get smoothed emotion for a tracked face"""
        if track_id not in self.emotion_history or len(self.emotion_history[track_id]) == 0:
            return "Unknown", 0.0
            
        history = list(self.emotion_history[track_id])
        
        # Weight recent predictions more heavily
        weights = np.linspace(0.5, 1.0, len(history))
        emotion_scores = defaultdict(float)
        total_weight = 0
        
        for i, (emotion, confidence) in enumerate(history):
            weight = weights[i] * confidence
            emotion_scores[emotion] += weight
            total_weight += weight
        
        if total_weight == 0:
            return "Unknown", 0.0
        
        # Normalize scores
        for emotion in emotion_scores:
            emotion_scores[emotion] /= total_weight
        
        best_emotion = max(emotion_scores, key=emotion_scores.get)
        return best_emotion, emotion_scores[best_emotion]
    
    def cleanup_old_tracks(self, active_track_ids):
        """Remove emotion history for tracks that no longer exist"""
        to_remove = [track_id for track_id in self.emotion_history.keys() 
                    if track_id not in active_track_ids]
        for track_id in to_remove:
            del self.emotion_history[track_id]

class PerformanceMonitor:
    """Monitor and adapt performance based on FPS"""
    def __init__(self, target_fps=15, check_interval=30):
        self.target_fps = target_fps
        self.check_interval = check_interval
        self.frame_times = deque(maxlen=check_interval)
        self.frame_count = 0
        
    def add_frame_time(self, frame_time):
        """Add processing time for a frame"""
        self.frame_times.append(frame_time)
        self.frame_count += 1
    
    def should_adjust_performance(self):
        """Check if performance adjustment is needed"""
        return self.frame_count % self.check_interval == 0 and len(self.frame_times) == self.check_interval
    
    def get_current_fps(self):
        """Calculate current FPS"""
        if len(self.frame_times) < 2:
            return self.target_fps
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / max(avg_frame_time, 0.001)
    
    def get_recommended_skip_frames(self, base_skip, max_skip):
        """Get recommended frame skip based on performance"""
        current_fps = self.get_current_fps()
        if current_fps < self.target_fps * 0.8:  # If FPS is too low
            return min(max_skip, base_skip + 2)
        elif current_fps > self.target_fps * 1.2:  # If FPS is too high
            return max(1, base_skip - 1)
        return base_skip

class ImprovedEmotionDetector:
    """Main emotion detection class with all improvements"""
    def __init__(self, config=None):
        self.config = config or EmotionDetectionConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.face_tracker = FaceTracker(self.config.face_tracking_threshold)
        self.emotion_smoother = EmotionSmoother(self.config.smoothing_window)
        self.performance_monitor = PerformanceMonitor(self.config.target_fps, 
                                                     self.config.performance_check_interval)
        
        # Initialize MediaPipe
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=self.config.min_detection_confidence
        )
        
        # Load model
        self._load_model()
        
        # Dynamic parameters
        self.current_skip_frames = self.config.base_skip_frames
        self.frame_count = 0
    
    def _load_model(self):
        """Load the emotion classification model"""
        try:
            self.model = SiglipForImageClassification.from_pretrained(
                self.config.model_name
            ).to(self.device)
            self.processor = AutoImageProcessor.from_pretrained(self.config.model_name)
            self.model.eval()
            
            # Define emotion labels (filtered)
            all_labels = {
                0: "Ahegao", 1: "Angry", 2: "Happy", 3: "Neutral",
                4: "Sad", 5: "Surprise"
            }
            
            # Filter out inappropriate emotions
            self.labels = {k: v for k, v in all_labels.items() 
                          if v not in self.config.emotion_filter}
            
            print("Model loaded successfully")
            print(f"Available emotions: {list(self.labels.values())}")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()
    
    def detect_faces_mediapipe(self, frame):
        """Detect faces using MediaPipe and return bounding boxes"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
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
                
                # Filter out faces that are too small
                if width >= self.config.min_face_size and height >= self.config.min_face_size:
                    faces.append((x, y, width, height))
        
        return faces
    
    def crop_face(self, frame, face_coords):
        """Crop face from frame with proportional padding"""
        x, y, w, h = face_coords
        
        # Calculate padding based on face size
        padding_x = int(w * self.config.padding_ratio)
        padding_y = int(h * self.config.padding_ratio)
        
        # Apply padding
        x_start = max(0, x - padding_x)
        y_start = max(0, y - padding_y)
        x_end = min(frame.shape[1], x + w + padding_x)
        y_end = min(frame.shape[0], y + h + padding_y)
        
        cropped_face = frame[y_start:y_end, x_start:x_end]
        return cropped_face
    
    def emotion_classification(self, image: np.ndarray):
        """Classify emotion from cropped face image"""
        try:
            if image.size == 0:
                return "No Face", 0.0, {}
                
            pil_img = Image.fromarray(image).convert("RGB")
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1).squeeze()
                
                if probs.dim() == 0:
                    probs = probs.unsqueeze(0)
                
                probs = probs.cpu().tolist()

            # Only include filtered emotions
            predictions = {}
            for i, prob in enumerate(probs):
                if i in self.labels:
                    predictions[self.labels[i]] = round(prob, 3)
            
            if not predictions:
                return "Unknown", 0.0, {}
                
            top_label = max(predictions, key=predictions.get)
            top_score = predictions[top_label]
            return top_label, top_score, predictions
        
        except Exception as e:
            print(f"Error in emotion classification: {e}")
            return "Error", 0.0, {}
    
    def run_detection(self):
        """Main detection loop"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.webcam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.webcam_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.webcam_fps)

        print("Starting emotion detection. Press 'q' to quit.")
        print("Press 's' to show detailed emotion probabilities.")
        
        show_detailed = False
        
        try:
            while True:
                frame_start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                
                # Detect faces
                faces = self.detect_faces_mediapipe(frame)
                
                # Update face tracking
                tracked_faces = self.face_tracker.update_tracks(faces)
                
                # Process emotions every nth frame
                if self.frame_count % self.current_skip_frames == 0:
                    for track_id, face_coords in tracked_faces.items():
                        # Crop face from RGB frame
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        cropped_face = self.crop_face(rgb_frame, face_coords)
                        
                        # Classify emotion
                        label, score, predictions = self.emotion_classification(cropped_face)
                        
                        if label not in ["No Face", "Error", "Unknown"]:
                            self.emotion_smoother.add_prediction(track_id, label, score)

                # Clean up old emotion history
                self.emotion_smoother.cleanup_old_tracks(set(tracked_faces.keys()))

                # Draw results
                for track_id, (x, y, w, h) in tracked_faces.items():
                    # Get smoothed emotion
                    emotion, confidence = self.emotion_smoother.get_smoothed_emotion(track_id)
                    
                    # Choose color based on emotion
                    color_map = {
                        "Happy": (0, 255, 0),      # Green
                        "Sad": (255, 0, 0),        # Blue
                        "Angry": (0, 0, 255),      # Red
                        "Surprise": (0, 255, 255), # Yellow
                        "Neutral": (128, 128, 128) # Gray
                    }
                    color = color_map.get(emotion, (255, 255, 255))  # Default white
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw track ID
                    cv2.putText(frame, f"ID: {track_id}", (x, y - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                    
                    # Display emotion
                    display_text = f"{emotion}: {confidence:.2f}"
                    text_y = max(y - 10, 20)
                    cv2.putText(frame, display_text, (x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                # Show frame info
                current_fps = self.performance_monitor.get_current_fps()
                info_text = f"Faces: {len(tracked_faces)} | FPS: {current_fps:.1f} | Skip: {self.current_skip_frames}"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                if show_detailed and tracked_faces:
                    y_offset = 60
                    for track_id in tracked_faces.keys():
                        emotion, confidence = self.emotion_smoother.get_smoothed_emotion(track_id)
                        detail_text = f"ID {track_id}: {emotion} ({confidence:.2f})"
                        cv2.putText(frame, detail_text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        y_offset += 25

                cv2.imshow("Improved Facial Emotion Detection", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    show_detailed = not show_detailed
                    print(f"Detailed view: {'ON' if show_detailed else 'OFF'}")
                
                # Performance monitoring and adjustment
                frame_time = time.time() - frame_start_time
                self.performance_monitor.add_frame_time(frame_time)
                
                if self.performance_monitor.should_adjust_performance():
                    new_skip_frames = self.performance_monitor.get_recommended_skip_frames(
                        self.config.base_skip_frames, self.config.max_skip_frames
                    )
                    if new_skip_frames != self.current_skip_frames:
                        self.current_skip_frames = new_skip_frames
                        print(f"Adjusted skip frames to: {self.current_skip_frames}")

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            print("Cleanup completed")

# Usage
if __name__ == "__main__":
    # Create custom configuration if needed
    config = EmotionDetectionConfig()
    # config.min_face_size = 60  # Increase minimum face size
    # config.smoothing_window = 7  # Increase smoothing window
    
    # Create and run detector
    detector = ImprovedEmotionDetector(config)
    detector.run_detection()