import cv2
from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch.nn.functional as F
import mediapipe as mp
from collections import defaultdict, deque
import time
import json
import threading
import socket
import base64
from datetime import datetime
import asyncio
import websockets
from flask import Flask, jsonify
from flask_cors import CORS
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

class EmotionDetectionConfig:
    """Configuration class for headless emotion detection"""
    def __init__(self):
        self.model_name = "prithivMLmods/Facial-Emotion-Detection-SigLIP2"
        self.min_detection_confidence = 0.5
        self.padding_ratio = 0.2
        self.min_face_size = 50
        self.base_skip_frames = 2
        self.smoothing_window = 5
        self.face_tracking_threshold = 0.3
        
        # Broadcasting settings
        self.websocket_port = 8765
        self.http_port = 5000
        self.udp_port = 9999
        self.broadcast_fps = 10  # How often to broadcast results
        self.max_broadcast_history = 100  # Keep last N results
        
        # Filter out inappropriate emotions
        self.emotion_filter = {"Ahegao"}
        
        self.webcam_width = 640
        self.webcam_height = 480
        self.webcam_fps = 30
        
        # Enable/disable broadcast methods
        self.enable_websocket = True
        self.enable_http_api = True
        self.enable_udp = True
        self.enable_frame_streaming = True  # Stream processed frames

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
        for track_id in self.tracked_faces:
            self.tracked_faces[track_id]['found'] = False
            self.tracked_faces[track_id]['missing_frames'] += 1
        
        updated_tracks = {}
        
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
                self.tracked_faces[best_match_id]['bbox'] = face
                self.tracked_faces[best_match_id]['found'] = True
                self.tracked_faces[best_match_id]['missing_frames'] = 0
                updated_tracks[best_match_id] = face
            else:
                track_id = self.next_id
                self.next_id += 1
                self.tracked_faces[track_id] = {
                    'bbox': face,
                    'found': True,
                    'missing_frames': 0
                }
                updated_tracks[track_id] = face
        
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
        self.emotion_history[track_id].append((emotion, confidence))
    
    def get_smoothed_emotion(self, track_id):
        if track_id not in self.emotion_history or len(self.emotion_history[track_id]) == 0:
            return "Unknown", 0.0
            
        history = list(self.emotion_history[track_id])
        weights = np.linspace(0.5, 1.0, len(history))
        emotion_scores = defaultdict(float)
        total_weight = 0
        
        for i, (emotion, confidence) in enumerate(history):
            weight = weights[i] * confidence
            emotion_scores[emotion] += weight
            total_weight += weight
        
        if total_weight == 0:
            return "Unknown", 0.0
        
        for emotion in emotion_scores:
            emotion_scores[emotion] /= total_weight
        
        best_emotion = max(emotion_scores, key=emotion_scores.get)
        return best_emotion, emotion_scores[best_emotion]
    
    def cleanup_old_tracks(self, active_track_ids):
        to_remove = [track_id for track_id in self.emotion_history.keys() 
                    if track_id not in active_track_ids]
        for track_id in to_remove:
            del self.emotion_history[track_id]

class EmotionBroadcaster:
    """Handles broadcasting emotion detection results"""
    def __init__(self, config):
        self.config = config
        self.latest_results = {}
        self.results_history = deque(maxlen=config.max_broadcast_history)
        self.websocket_clients = set()
        self.latest_frame = None
        
        # Threading locks
        self.results_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        
        # Initialize Flask app
        if config.enable_http_api:
            self.app = Flask(__name__)
            CORS(self.app)
            self._setup_http_routes()
        
        # Start broadcasting services
        self._start_services()
    
    def _setup_http_routes(self):
        """Setup HTTP API routes"""
        @self.app.route('/api/emotions', methods=['GET'])
        def get_emotions():
            with self.results_lock:
                return jsonify({
                    'timestamp': datetime.now().isoformat(),
                    'emotions': self.latest_results,
                    'face_count': len(self.latest_results)
                })
        
        @self.app.route('/api/emotions/history', methods=['GET'])
        def get_emotions_history():
            with self.results_lock:
                return jsonify({
                    'history': list(self.results_history),
                    'count': len(self.results_history)
                })
        
        @self.app.route('/api/frame', methods=['GET'])
        def get_frame():
            with self.frame_lock:
                if self.latest_frame is not None:
                    _, buffer = cv2.imencode('.jpg', self.latest_frame)
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    return jsonify({
                        'frame': frame_b64,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({'error': 'No frame available'}), 404
        
        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            return jsonify({
                'status': 'running',
                'timestamp': datetime.now().isoformat(),
                'services': {
                    'websocket': self.config.enable_websocket,
                    'http': self.config.enable_http_api,
                    'udp': self.config.enable_udp
                },
                'ports': {
                    'websocket': self.config.websocket_port,
                    'http': self.config.http_port,
                    'udp': self.config.udp_port
                }
            })
        
        @self.app.route('/', methods=['GET'])
        def index():
            return jsonify({
                'message': 'Emotion Detection API',
                'endpoints': {
                    '/api/emotions': 'Current emotion detections',
                    '/api/emotions/history': 'Historical detections',
                    '/api/frame': 'Latest processed frame (base64)',
                    '/api/status': 'System status'
                }
            })
    
    def _start_services(self):
        """Start all broadcasting services"""
        if self.config.enable_http_api:
            threading.Thread(target=self._start_http_server, daemon=True).start()
        
        if self.config.enable_websocket:
            threading.Thread(target=self._start_websocket_server, daemon=True).start()
        
        if self.config.enable_udp:
            threading.Thread(target=self._start_udp_broadcaster, daemon=True).start()
    
    def _start_http_server(self):
        """Start HTTP API server"""
        try:
            self.app.run(host='0.0.0.0', port=self.config.http_port, debug=False, threaded=True)
        except Exception as e:
            print(f"HTTP server error: {e}")
    
    def _start_websocket_server(self):
        """Start WebSocket server"""
        async def handle_client(websocket, path):
            self.websocket_clients.add(websocket)
            try:
                await websocket.wait_closed()
            except Exception as e:
                print(f"WebSocket client error: {e}")
            finally:
                self.websocket_clients.discard(websocket)
        
        async def start_server():
            try:
                await websockets.serve(handle_client, "0.0.0.0", self.config.websocket_port)
                print(f"WebSocket server started on port {self.config.websocket_port}")
            except Exception as e:
                print(f"WebSocket server error: {e}")
        
        def run_websocket():
            asyncio.new_event_loop().run_until_complete(start_server())
            asyncio.get_event_loop().run_forever()
        
        threading.Thread(target=run_websocket, daemon=True).start()
    
    def _start_udp_broadcaster(self):
        """Start UDP broadcaster"""
        def udp_broadcast():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            while True:
                try:
                    with self.results_lock:
                        if self.latest_results:
                            message = json.dumps({
                                'timestamp': datetime.now().isoformat(),
                                'emotions': self.latest_results,
                                'face_count': len(self.latest_results)
                            })
                            sock.sendto(message.encode(), ('255.255.255.255', self.config.udp_port))
                    
                    time.sleep(1.0 / self.config.broadcast_fps)
                except Exception as e:
                    print(f"UDP broadcast error: {e}")
                    time.sleep(1)
        
        threading.Thread(target=udp_broadcast, daemon=True).start()
        print(f"UDP broadcaster started on port {self.config.udp_port}")
    
    def update_results(self, results, frame=None):
        """Update emotion results and broadcast"""
        timestamp = datetime.now().isoformat()
        
        with self.results_lock:
            self.latest_results = results
            self.results_history.append({
                'timestamp': timestamp,
                'emotions': results,
                'face_count': len(results)
            })
        
        if frame is not None and self.config.enable_frame_streaming:
            with self.frame_lock:
                self.latest_frame = frame.copy()
        
        # Broadcast via WebSocket
        if self.config.enable_websocket and self.websocket_clients:
            self._broadcast_websocket({
                'timestamp': timestamp,
                'emotions': results,
                'face_count': len(results)
            })
    
    def _broadcast_websocket(self, data):
        """Broadcast data to WebSocket clients"""
        if not self.websocket_clients:
            return
        
        message = json.dumps(data)
        disconnected_clients = set()
        
        for client in self.websocket_clients.copy():
            try:
                asyncio.run_coroutine_threadsafe(
                    client.send(message),
                    asyncio.get_event_loop()
                )
            except Exception:
                disconnected_clients.add(client)
        
        self.websocket_clients -= disconnected_clients

class HeadlessEmotionDetector:
    """Headless emotion detection system with broadcasting"""
    def __init__(self, config=None):
        self.config = config or EmotionDetectionConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.face_tracker = FaceTracker(self.config.face_tracking_threshold)
        self.emotion_smoother = EmotionSmoother(self.config.smoothing_window)
        self.broadcaster = EmotionBroadcaster(self.config)
        
        # Initialize MediaPipe
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=self.config.min_detection_confidence
        )
        
        # Load model
        self._load_model()
        
        self.frame_count = 0
        self.running = False
    
    def _load_model(self):
        """Load the emotion classification model"""
        try:
            self.model = SiglipForImageClassification.from_pretrained(
                self.config.model_name
            ).to(self.device)
            self.processor = AutoImageProcessor.from_pretrained(
                self.config.model_name, use_fast=True
            )
            self.model.eval()
            
            all_labels = {
                0: "Ahegao", 1: "Angry", 2: "Happy", 3: "Neutral",
                4: "Sad", 5: "Surprise"
            }
            
            self.labels = {k: v for k, v in all_labels.items() 
                          if v not in self.config.emotion_filter}
            
            print("Model loaded successfully")
            print(f"Available emotions: {list(self.labels.values())}")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()
    
    def detect_faces_mediapipe(self, frame):
        """Detect faces using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                if width >= self.config.min_face_size and height >= self.config.min_face_size:
                    faces.append((x, y, width, height))
        
        return faces
    
    def crop_face(self, frame, face_coords):
        """Crop face from frame with proportional padding"""
        x, y, w, h = face_coords
        
        padding_x = int(w * self.config.padding_ratio)
        padding_y = int(h * self.config.padding_ratio)
        
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
    
    def print_status(self, tracked_faces):
        """Print current detection status"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Detection Status:")
        print(f"  Detected Faces: {len(tracked_faces)}")
        
        for track_id, face_coords in tracked_faces.items():
            emotion, confidence = self.emotion_smoother.get_smoothed_emotion(track_id)
            x, y, w, h = face_coords
            print(f"  Face ID {track_id}: {emotion} ({confidence:.2f}) at ({x},{y}) {w}x{h}")
        
        print(f"  Broadcasting on:")
        if self.config.enable_http_api:
            print(f"    HTTP API: http://localhost:{self.config.http_port}/api/emotions")
        if self.config.enable_websocket:
            print(f"    WebSocket: ws://localhost:{self.config.websocket_port}")
        if self.config.enable_udp:
            print(f"    UDP Broadcast: port {self.config.udp_port}")
    
    def run_detection(self):
        """Main detection loop"""
        # Use headless OpenCV
        import os
        os.environ['DISPLAY'] = ''  # Force headless mode
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.webcam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.webcam_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.webcam_fps)

        print("\n🚀 Headless Emotion Detection Started!")
        print("📡 Broadcasting emotion data...")
        print("⏹️  Press Ctrl+C to stop")
        print("=" * 50)
        
        self.running = True
        last_broadcast = time.time()
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue

                self.frame_count += 1
                
                # Detect faces
                faces = self.detect_faces_mediapipe(frame)
                
                # Update face tracking
                tracked_faces = self.face_tracker.update_tracks(faces)
                
                # Process emotions every nth frame
                if self.frame_count % self.config.base_skip_frames == 0:
                    for track_id, face_coords in tracked_faces.items():
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        cropped_face = self.crop_face(rgb_frame, face_coords)
                        
                        label, score, predictions = self.emotion_classification(cropped_face)
                        
                        if label not in ["No Face", "Error", "Unknown"]:
                            self.emotion_smoother.add_prediction(track_id, label, score)

                # Clean up old emotion history
                self.emotion_smoother.cleanup_old_tracks(set(tracked_faces.keys()))
                
                # Broadcast results
                current_time = time.time()
                if current_time - last_broadcast >= (1.0 / self.config.broadcast_fps):
                    # Prepare results for broadcasting
                    results = {}
                    annotated_frame = frame.copy() if self.config.enable_frame_streaming else None
                    
                    for track_id, face_coords in tracked_faces.items():
                        emotion, confidence = self.emotion_smoother.get_smoothed_emotion(track_id)
                        x, y, w, h = face_coords
                        
                        results[str(track_id)] = {
                            'emotion': emotion,
                            'confidence': round(confidence, 3),
                            'bbox': {'x': x, 'y': y, 'width': w, 'height': h}
                        }
                        
                        # Annotate frame for streaming
                        if annotated_frame is not None:
                            color_map = {
                                "Happy": (0, 255, 0), "Sad": (255, 0, 0), 
                                "Angry": (0, 0, 255), "Surprise": (0, 255, 255), 
                                "Neutral": (128, 128, 128)
                            }
                            color = color_map.get(emotion, (255, 255, 255))
                            
                            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(annotated_frame, f"ID:{track_id}", (x, y - 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(annotated_frame, f"{emotion}: {confidence:.2f}", 
                                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Broadcast the results
                    self.broadcaster.update_results(results, annotated_frame)
                    
                    # Print status periodically
                    if self.frame_count % 150 == 0:  # Every ~5 seconds at 30fps
                        self.print_status(tracked_faces)
                    
                    last_broadcast = current_time

        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            self.running = False
            cap.release()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            print("✅ Cleanup completed")

# Usage
if __name__ == "__main__":
    # Create configuration
    config = EmotionDetectionConfig()
    
    # Customize settings if needed
    # config.broadcast_fps = 5  # Lower broadcast rate
    # config.enable_websocket = False  # Disable WebSocket
    
    print("🎯 Initializing Headless Emotion Detection System...")
    
    # Install required packages check
    try:
        import websockets
        import flask
        import flask_cors
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("Install with: pip install websockets flask flask-cors")
        exit(1)
    
    # Create and run detector
    detector = HeadlessEmotionDetector(config)
    detector.run_detection()