"""
Enhanced Face Analyzer Module
Combines MediaPipe landmarks with FER for advanced facial analysis
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import mediapipe as mp

try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    logging.warning("FER not available. Using basic emotion detection.")

logger = logging.getLogger(__name__)

@dataclass
class EnhancedFacialMetrics:
    """Enhanced facial metrics including emotion recognition"""
    # Basic expressions
    smile_intensity: float
    frown_intensity: float
    eyebrow_raise: float
    mouth_open: float
    
    # Eye contact metrics
    eye_contact_percentage: float
    gaze_direction: Tuple[float, float]  # (x, y) offset from center
    blink_detected: bool
    eye_openness: float
    
    # Head pose
    head_tilt: float  # degrees
    head_nod: float   # degrees
    head_yaw: float   # degrees
    
    # Emotion recognition
    primary_emotion: str
    emotion_confidence: float
    emotion_scores: Dict[str, float]
    
    # Quality metrics
    face_detection_confidence: float
    landmark_quality: float
    timestamp: datetime

class EnhancedFaceAnalyzer:
    """Enhanced facial analysis with FER and advanced metrics"""
    
    def __init__(self, 
                 use_fer: bool = True,
                 emotion_threshold: float = 0.3,
                 eye_contact_threshold: float = 0.7):
        """
        Initialize Enhanced Face Analyzer
        
        Args:
            use_fer: Whether to use FER for emotion recognition
            emotion_threshold: Minimum confidence for emotion detection
            eye_contact_threshold: Threshold for eye contact detection
        """
        self.use_fer = use_fer and FER_AVAILABLE
        self.emotion_threshold = emotion_threshold
        self.eye_contact_threshold = eye_contact_threshold
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Face mesh with enhanced settings
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize FER if available
        if self.use_fer:
            try:
                self.fer_detector = FER(mtcnn=True)
                logger.info("FER emotion detector initialized")
            except Exception as e:
                logger.warning(f"FER initialization failed: {e}")
                self.use_fer = False
        
        # Eye contact tracking
        self.eye_contact_history = []
        self.blink_history = []
        self.head_pose_history = []
        
        # Constants for calculations
        self.FRAME_CENTER = np.array([320, 240])  # Center of 640x480 frame
        self.EYE_ASPECT_RATIO_THRESHOLD = 0.2
        self.HEAD_POSE_SMOOTHING = 0.8
        
        logger.info("Enhanced Face Analyzer initialized")
    
    def analyze_frame(self, frame: np.ndarray) -> Optional[EnhancedFacialMetrics]:
        """
        Analyze facial features in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            EnhancedFacialMetrics or None if no face detected
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None
            
            # Get the first face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract landmarks
            landmarks = self._extract_landmarks(face_landmarks, frame.shape)
            
            # Analyze expressions
            expressions = self._analyze_expressions(landmarks)
            
            # Analyze eye contact
            eye_metrics = self._analyze_eye_contact(landmarks)
            
            # Analyze head pose
            head_pose = self._analyze_head_pose(landmarks)
            
            # Analyze emotions with FER
            emotions = self._analyze_emotions(frame, landmarks)
            
            # Create enhanced metrics
            metrics = EnhancedFacialMetrics(
                # Basic expressions
                smile_intensity=expressions['smile'],
                frown_intensity=expressions['frown'],
                eyebrow_raise=expressions['eyebrow_raise'],
                mouth_open=expressions['mouth_open'],
                
                # Eye contact
                eye_contact_percentage=eye_metrics['contact_percentage'],
                gaze_direction=eye_metrics['gaze_direction'],
                blink_detected=eye_metrics['blink_detected'],
                eye_openness=eye_metrics['eye_openness'],
                
                # Head pose
                head_tilt=head_pose['tilt'],
                head_nod=head_pose['nod'],
                head_yaw=head_pose['yaw'],
                
                # Emotions
                primary_emotion=emotions['primary'],
                emotion_confidence=emotions['confidence'],
                emotion_scores=emotions['scores'],
                
                # Quality
                face_detection_confidence=0.8,  # MediaPipe confidence
                landmark_quality=self._calculate_landmark_quality(landmarks),
                timestamp=datetime.now()
            )
            
            # Update history
            self._update_history(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            return None
    
    def _extract_landmarks(self, face_landmarks, frame_shape) -> Dict:
        """Extract landmark coordinates"""
        height, width = frame_shape[:2]
        landmarks = {}
        
        # Extract all landmarks
        all_landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z
            all_landmarks.append((x, y, z))
        
        # Define landmark groups
        landmark_groups = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'right_eyebrow': [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'lips': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
            'nose': [1, 2, 98, 327, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }
        
        # Extract landmark groups
        for group_name, indices in landmark_groups.items():
            landmarks[group_name] = [all_landmarks[i] for i in indices if i < len(all_landmarks)]
        
        landmarks['all_landmarks'] = all_landmarks
        return landmarks
    
    def _analyze_expressions(self, landmarks: Dict) -> Dict:
        """Analyze basic facial expressions"""
        expressions = {}
        
        try:
            # Smile detection (mouth corner movement)
            if 'mouth' in landmarks and len(landmarks['mouth']) >= 8:
                mouth_corners = [landmarks['mouth'][0], landmarks['mouth'][6]]  # Left and right corners
                mouth_width = np.linalg.norm(np.array(mouth_corners[1]) - np.array(mouth_corners[0]))
                expressions['smile'] = min(1.0, mouth_width / 100.0)  # Normalize
            else:
                expressions['smile'] = 0.0
            
            # Frown detection (eyebrow movement)
            if 'left_eyebrow' in landmarks and 'right_eyebrow' in landmarks:
                left_eyebrow_center = np.mean(landmarks['left_eyebrow'], axis=0)
                right_eyebrow_center = np.mean(landmarks['right_eyebrow'], axis=0)
                
                # Calculate eyebrow height relative to eyes
                if 'left_eye' in landmarks and 'right_eye' in landmarks:
                    left_eye_center = np.mean(landmarks['left_eye'], axis=0)
                    right_eye_center = np.mean(landmarks['right_eye'], axis=0)
                    
                    left_height = left_eye_center[1] - left_eyebrow_center[1]
                    right_height = right_eye_center[1] - right_eyebrow_center[1]
                    
                    avg_height = (left_height + right_height) / 2
                    expressions['eyebrow_raise'] = max(0, min(1.0, (avg_height - 20) / 30))
                else:
                    expressions['eyebrow_raise'] = 0.0
            else:
                expressions['eyebrow_raise'] = 0.0
            
            # Mouth openness
            if 'mouth' in landmarks and len(landmarks['mouth']) >= 8:
                mouth_center = landmarks['mouth'][2]  # Upper lip center
                mouth_bottom = landmarks['mouth'][6]  # Lower lip center
                mouth_height = abs(mouth_center[1] - mouth_bottom[1])
                expressions['mouth_open'] = min(1.0, mouth_height / 50.0)
            else:
                expressions['mouth_open'] = 0.0
            
            # Frown (simplified)
            expressions['frown'] = max(0, 1 - expressions['eyebrow_raise'])
            
        except Exception as e:
            logger.error(f"Expression analysis failed: {e}")
            expressions = {'smile': 0.0, 'frown': 0.0, 'eyebrow_raise': 0.0, 'mouth_open': 0.0}
        
        return expressions
    
    def _analyze_eye_contact(self, landmarks: Dict) -> Dict:
        """Analyze eye contact and gaze direction"""
        eye_metrics = {}
        
        try:
            if 'left_eye' in landmarks and 'right_eye' in landmarks:
                # Calculate eye centers
                left_eye_center = np.mean(landmarks['left_eye'], axis=0)
                right_eye_center = np.mean(landmarks['right_eye'], axis=0)
                eye_center = (left_eye_center + right_eye_center) / 2
                
                # Calculate gaze direction (offset from frame center)
                gaze_offset = eye_center[:2] - self.FRAME_CENTER
                gaze_direction = (float(gaze_offset[0]), float(gaze_offset[1]))
                
                # Calculate eye contact percentage
                distance_from_center = np.linalg.norm(gaze_offset)
                max_distance = np.linalg.norm(np.array([0, 0]) - self.FRAME_CENTER)
                eye_contact_percentage = max(0, 1 - (distance_from_center / max_distance))
                
                # Detect blink using Eye Aspect Ratio (EAR)
                left_ear = self._calculate_ear(landmarks['left_eye'])
                right_ear = self._calculate_ear(landmarks['right_eye'])
                avg_ear = (left_ear + right_ear) / 2
                
                blink_detected = avg_ear < self.EYE_ASPECT_RATIO_THRESHOLD
                eye_openness = min(1.0, avg_ear / 0.3)  # Normalize to 0-1
                
                eye_metrics.update({
                    'contact_percentage': eye_contact_percentage,
                    'gaze_direction': gaze_direction,
                    'blink_detected': blink_detected,
                    'eye_openness': eye_openness
                })
            else:
                eye_metrics.update({
                    'contact_percentage': 0.5,
                    'gaze_direction': (0.0, 0.0),
                    'blink_detected': False,
                    'eye_openness': 0.5
                })
                
        except Exception as e:
            logger.error(f"Eye contact analysis failed: {e}")
            eye_metrics.update({
                'contact_percentage': 0.5,
                'gaze_direction': (0.0, 0.0),
                'blink_detected': False,
                'eye_openness': 0.5
            })
        
        return eye_metrics
    
    def _calculate_ear(self, eye_landmarks: List) -> float:
        """Calculate Eye Aspect Ratio (EAR)"""
        try:
            if len(eye_landmarks) < 6:
                return 0.0
            
            # Vertical distances
            A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
            B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
            
            # Horizontal distance
            C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
            
            # Avoid division by zero
            if C == 0:
                return 0.0
            
            ear = (A + B) / (2.0 * C)
            return ear
            
        except Exception:
            return 0.0
    
    def _analyze_head_pose(self, landmarks: Dict) -> Dict:
        """Analyze head pose (tilt, nod, yaw)"""
        head_pose = {}
        
        try:
            if 'face_oval' in landmarks and len(landmarks['face_oval']) >= 10:
                # Calculate face orientation using key points
                face_points = np.array(landmarks['face_oval'])
                
                # Head tilt (rotation around Z-axis)
                left_ear = face_points[0]  # Left ear
                right_ear = face_points[16]  # Right ear
                tilt_angle = np.arctan2(right_ear[1] - left_ear[1], right_ear[0] - left_ear[0])
                head_pose['tilt'] = np.degrees(tilt_angle)
                
                # Head nod (rotation around X-axis) - simplified
                nose_tip = landmarks['nose'][4] if len(landmarks['nose']) > 4 else face_points[8]
                chin = face_points[8]
                forehead = face_points[9]
                
                nod_angle = np.arctan2(nose_tip[1] - chin[1], nose_tip[0] - chin[0])
                head_pose['nod'] = np.degrees(nod_angle)
                
                # Head yaw (rotation around Y-axis) - simplified
                left_cheek = face_points[2]
                right_cheek = face_points[14]
                yaw_angle = np.arctan2(right_cheek[0] - left_cheek[0], right_cheek[1] - left_cheek[1])
                head_pose['yaw'] = np.degrees(yaw_angle)
                
            else:
                head_pose.update({'tilt': 0.0, 'nod': 0.0, 'yaw': 0.0})
                
        except Exception as e:
            logger.error(f"Head pose analysis failed: {e}")
            head_pose.update({'tilt': 0.0, 'nod': 0.0, 'yaw': 0.0})
        
        return head_pose
    
    def _analyze_emotions(self, frame: np.ndarray, landmarks: Dict) -> Dict:
        """Analyze emotions using FER or fallback methods"""
        emotions = {}
        
        try:
            if self.use_fer and self.fer_detector:
                # Use FER for emotion detection
                fer_result = self.fer_detector.detect_emotions(frame)
                
                if fer_result:
                    # Get the first detected face
                    face_emotions = fer_result[0]['emotions']
                    
                    # Find primary emotion
                    primary_emotion = max(face_emotions.items(), key=lambda x: x[1])
                    
                    emotions.update({
                        'primary': primary_emotion[0],
                        'confidence': primary_emotion[1],
                        'scores': face_emotions
                    })
                else:
                    emotions.update({
                        'primary': 'neutral',
                        'confidence': 0.5,
                        'scores': {'neutral': 0.5, 'happy': 0.2, 'sad': 0.1, 'angry': 0.1, 'fear': 0.05, 'surprise': 0.05}
                    })
            else:
                # Fallback emotion detection based on expressions
                if 'smile' in landmarks:
                    smile_score = landmarks.get('smile', 0.0)
                    if smile_score > 0.6:
                        primary_emotion = 'happy'
                        confidence = smile_score
                    elif smile_score < 0.2:
                        primary_emotion = 'sad'
                        confidence = 1 - smile_score
                    else:
                        primary_emotion = 'neutral'
                        confidence = 0.5
                    
                    emotions.update({
                        'primary': primary_emotion,
                        'confidence': confidence,
                        'scores': {
                            'happy': smile_score,
                            'sad': 1 - smile_score,
                            'neutral': 1 - abs(smile_score - 0.5) * 2
                        }
                    })
                else:
                    emotions.update({
                        'primary': 'neutral',
                        'confidence': 0.5,
                        'scores': {'neutral': 0.5, 'happy': 0.25, 'sad': 0.25}
                    })
                    
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            emotions.update({
                'primary': 'neutral',
                'confidence': 0.5,
                'scores': {'neutral': 0.5, 'happy': 0.25, 'sad': 0.25}
            })
        
        return emotions
    
    def _calculate_landmark_quality(self, landmarks: Dict) -> float:
        """Calculate quality score for landmark detection"""
        try:
            if 'all_landmarks' not in landmarks:
                return 0.0
            
            # Check if landmarks are within frame bounds
            all_landmarks = landmarks['all_landmarks']
            valid_landmarks = 0
            
            for landmark in all_landmarks:
                x, y = landmark[0], landmark[1]
                if 0 <= x <= 640 and 0 <= y <= 480:  # Frame bounds
                    valid_landmarks += 1
            
            quality = valid_landmarks / len(all_landmarks) if all_landmarks else 0.0
            return quality
            
        except Exception:
            return 0.5
    
    def _update_history(self, metrics: EnhancedFacialMetrics):
        """Update analysis history for smoothing"""
        # Eye contact history
        self.eye_contact_history.append(metrics.eye_contact_percentage)
        if len(self.eye_contact_history) > 30:  # Keep last 30 frames
            self.eye_contact_history.pop(0)
        
        # Blink history
        self.blink_history.append(metrics.blink_detected)
        if len(self.blink_history) > 30:
            self.blink_history.pop(0)
        
        # Head pose history
        self.head_pose_history.append((metrics.head_tilt, metrics.head_nod, metrics.head_yaw))
        if len(self.head_pose_history) > 30:
            self.head_pose_history.pop(0)
    
    def get_smoothed_metrics(self) -> Optional[EnhancedFacialMetrics]:
        """Get smoothed metrics using history"""
        if not self.eye_contact_history:
            return None
        
        # Calculate smoothed values
        smoothed_eye_contact = np.mean(self.eye_contact_history[-10:])  # Last 10 frames
        smoothed_head_tilt = np.mean([pose[0] for pose in self.head_pose_history[-10:]])
        smoothed_head_nod = np.mean([pose[1] for pose in self.head_pose_history[-10:]])
        smoothed_head_yaw = np.mean([pose[2] for pose in self.head_pose_history[-10:]])
        
        # Create smoothed metrics (copy last metrics with smoothed values)
        if hasattr(self, '_last_metrics') and self._last_metrics:
            smoothed_metrics = EnhancedFacialMetrics(
                smile_intensity=self._last_metrics.smile_intensity,
                frown_intensity=self._last_metrics.frown_intensity,
                eyebrow_raise=self._last_metrics.eyebrow_raise,
                mouth_open=self._last_metrics.mouth_open,
                eye_contact_percentage=smoothed_eye_contact,
                gaze_direction=self._last_metrics.gaze_direction,
                blink_detected=self._last_metrics.blink_detected,
                eye_openness=self._last_metrics.eye_openness,
                head_tilt=smoothed_head_tilt,
                head_nod=smoothed_head_nod,
                head_yaw=smoothed_head_yaw,
                primary_emotion=self._last_metrics.primary_emotion,
                emotion_confidence=self._last_metrics.emotion_confidence,
                emotion_scores=self._last_metrics.emotion_scores,
                face_detection_confidence=self._last_metrics.face_detection_confidence,
                landmark_quality=self._last_metrics.landmark_quality,
                timestamp=datetime.now()
            )
            return smoothed_metrics
        
        return None
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        logger.info("Enhanced Face Analyzer cleanup completed")
