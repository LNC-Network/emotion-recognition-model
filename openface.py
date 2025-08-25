import cv2 # type: ignore
import numpy as np # type: ignore
import subprocess
import os
import tempfile
from collections import defaultdict, deque
import time

class EmotionDetectionConfig:
    """Configuration class for emotion detection parameters"""
    def __init__(self):
        self.openface_executable = "FeatureExtraction"  # OpenFace executable name
        self.min_face_size = 50
        self.base_skip_frames = 2
        self.max_skip_frames = 8
        self.smoothing_window = 5
        self.face_tracking_threshold = 0.3
        self.performance_check_interval = 30  # frames
        self.target_fps = 15
        
        self.webcam_width = 640
        self.webcam_height = 480
        self.webcam_fps = 60
        
        # OpenFace2 confidence threshold
        self.openface_confidence_threshold = 0.7
        
        # AU to emotion mapping thresholds
        self.emotion_thresholds = {
            'happy': {'AU06': 1.0, 'AU12': 1.0},      # Cheek raiser + Lip corner puller
            'sad': {'AU01': 1.0, 'AU04': 1.0, 'AU15': 1.0},  # Inner brow raiser + Brow lowerer + Lip corner depressor
            'angry': {'AU04': 1.5, 'AU05': 1.0, 'AU07': 1.0, 'AU23': 1.0},  # Brow lowerer + Upper lid raiser + Lid tightener + Lip tightener
            'surprise': {'AU01': 1.5, 'AU02': 1.5, 'AU05': 2.0, 'AU26': 1.5},  # Inner brow raiser + Outer brow raiser + Upper lid raiser + Jaw drop
        }

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
        
        best_emotion = max(emotion_scores, key=emotion_scores.get) # type: ignore
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
    """Main emotion detection class with OpenFace2 integration"""
    def __init__(self, config=None):
        self.config = config or EmotionDetectionConfig()
        
        # Initialize components
        self.face_tracker = FaceTracker(self.config.face_tracking_threshold)
        self.emotion_smoother = EmotionSmoother(self.config.smoothing_window)
        self.performance_monitor = PerformanceMonitor(self.config.target_fps, 
                                                     self.config.performance_check_interval)
        
        # Initialize OpenFace2
        self._check_openface()
        
        # Dynamic parameters
        self.current_skip_frames = self.config.base_skip_frames
        self.frame_count = 0
        
        # Temporary directories for OpenFace processing
        self.temp_dir = tempfile.mkdtemp()
        
    def _check_openface(self):
        """Check if OpenFace2 is available"""
        try:
            # Try to run OpenFace to check if it's available
            result = subprocess.run([self.config.openface_executable, "-help"], 
                                  capture_output=True, text=True, timeout=10)
            print("OpenFace2 detected and ready")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            print("Error: OpenFace2 not found. Please ensure:")
            print("1. OpenFace2 is installed")
            print("2. FeatureExtraction executable is in your PATH")
            print("3. Or modify config.openface_executable to point to the correct path")
            exit(1)
    
    def detect_faces_and_emotions_openface(self, frame):
        """Use OpenFace2 to detect faces and extract AUs for emotion classification"""
        # Save frame temporarily
        temp_image_path = os.path.join(self.temp_dir, f"temp_frame_{self.frame_count}.jpg")
        temp_output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(temp_output_dir, exist_ok=True)
        
        cv2.imwrite(temp_image_path, frame)
        
        try:
            # Run OpenFace2 feature extraction
            cmd = [
                self.config.openface_executable,
                "-f", temp_image_path,
                "-out_dir", temp_output_dir,
                "-2Dfp", "-3Dfp", "-pdmparams", "-pose", "-aus", "-gaze"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return [], {}
            
            # Parse OpenFace2 output
            csv_filename = os.path.splitext(os.path.basename(temp_image_path))[0] + ".csv"
            csv_path = os.path.join(temp_output_dir, csv_filename)
            
            faces = []
            face_emotions = {}
            
            if os.path.exists(csv_path):
                faces, face_emotions = self._parse_openface_output(csv_path, frame.shape)
            
            return faces, face_emotions
            
        except subprocess.TimeoutExpired:
            print("OpenFace2 processing timeout")
            return [], {}
        except Exception as e:
            print(f"Error in OpenFace2 processing: {e}")
            return [], {}
        finally:
            # Cleanup temporary files
            try:
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                # Clean output directory
                if os.path.exists(temp_output_dir):
                    for file in os.listdir(temp_output_dir):
                        file_path = os.path.join(temp_output_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
            except:
                pass
    
    def _parse_openface_output(self, csv_path, frame_shape):
        """Parse OpenFace2 CSV output to extract face locations and AUs"""
        faces = []
        face_emotions = {}
        
        try:
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                
            if len(lines) < 2:  # No data
                return faces, face_emotions
            
            # Parse header
            header = lines[0].strip().split(', ')
            
            # Find relevant columns
            confidence_col = header.index('confidence') if 'confidence' in header else -1
            
            # Find bounding box columns
            x_col = header.index('x_0') if 'x_0' in header else -1
            y_col = header.index('y_0') if 'y_0' in header else -1
            
            # Find AU columns
            au_cols = {}
            for i, col in enumerate(header):
                if col.startswith(' AU') and col.endswith('_r'):
                    au_name = col.strip().replace('_r', '')
                    au_cols[au_name] = i
            
            # Process each row (face detection)
            for line in lines[1:]:
                values = line.strip().split(', ')
                
                # Check confidence
                if confidence_col >= 0 and len(values) > confidence_col:
                    try:
                        confidence = float(values[confidence_col])
                        if confidence < self.config.openface_confidence_threshold:
                            continue
                    except:
                        continue
                
                # Extract bounding box from landmarks if available
                if x_col >= 0 and y_col >= 0 and len(values) > max(x_col, y_col):
                    try:
                        # Get all x and y coordinates for landmarks
                        x_coords = []
                        y_coords = []
                        
                        for i in range(68):  # 68 facial landmarks
                            x_idx = header.index(f'x_{i}') if f'x_{i}' in header else -1
                            y_idx = header.index(f'y_{i}') if f'y_{i}' in header else -1
                            
                            if x_idx >= 0 and y_idx >= 0 and len(values) > max(x_idx, y_idx):
                                x_coords.append(float(values[x_idx]))
                                y_coords.append(float(values[y_idx]))
                        
                        if x_coords and y_coords:
                            # Calculate bounding box from landmarks
                            min_x = max(0, int(min(x_coords)) - 20)
                            max_x = min(frame_shape[1], int(max(x_coords)) + 20)
                            min_y = max(0, int(min(y_coords)) - 20)
                            max_y = min(frame_shape[0], int(max(y_coords)) + 20)
                            
                            width = max_x - min_x
                            height = max_y - min_y
                            
                            if width >= self.config.min_face_size and height >= self.config.min_face_size:
                                face_id = len(faces)
                                faces.append((min_x, min_y, width, height))
                                
                                # Extract AUs for this face
                                aus = {}
                                for au_name, au_col in au_cols.items():
                                    if au_col < len(values):
                                        try:
                                            aus[au_name] = float(values[au_col])
                                        except:
                                            aus[au_name] = 0.0
                                
                                # Map AUs to emotion
                                emotion, confidence = self._map_aus_to_emotion(aus)
                                face_emotions[face_id] = (emotion, confidence)
                    except Exception as e:
                        continue
                        
        except Exception as e:
            print(f"Error parsing OpenFace2 output: {e}")
        
        return faces, face_emotions
    
    def _map_aus_to_emotion(self, aus):
        """Map Action Units to emotions"""
        emotion_scores = {}
        
        # Calculate scores for each emotion based on AU activations
        for emotion, au_requirements in self.config.emotion_thresholds.items():
            score = 0.0
            active_aus = 0
            
            for au, threshold in au_requirements.items():
                if au in aus and aus[au] >= threshold:
                    score += aus[au]
                    active_aus += 1
            
            # Emotion score is average activation of required AUs
            if active_aus > 0:
                emotion_scores[emotion] = score / len(au_requirements)
        
        # Default to neutral if no strong emotion detected
        if not emotion_scores or max(emotion_scores.values()) < 1.0:
            return "Neutral", 0.5
        
        best_emotion = max(emotion_scores, key=emotion_scores.get) # type: ignore
        confidence = min(1.0, emotion_scores[best_emotion] / 3.0)  # Normalize to 0-1
        
        return best_emotion.capitalize(), confidence
    
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

        print("Starting OpenFace2 emotion detection. Press 'q' to quit.")
        print("Press 's' to show detailed emotion probabilities.")
        
        show_detailed = False
        
        try:
            while True:
                frame_start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                
                # Process with OpenFace2 every nth frame
                faces = []
                face_emotions = {}
                if self.frame_count % self.current_skip_frames == 0:
                    faces, face_emotions = self.detect_faces_and_emotions_openface(frame)
                
                # Update face tracking
                tracked_faces = self.face_tracker.update_tracks(faces)
                
                # Update emotion predictions
                for i, (track_id, face_coords) in enumerate(tracked_faces.items()):
                    if i in face_emotions:
                        emotion, confidence = face_emotions[i]
                        self.emotion_smoother.add_prediction(track_id, emotion, confidence)

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

                cv2.imshow("OpenFace2 Facial Emotion Detection", frame)
                
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
            
            # Cleanup temporary directory
            try:
                import shutil
                if os.path.exists(self.temp_dir):
                    shutil.rmtree(self.temp_dir)
            except:
                pass
                
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