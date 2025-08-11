"""
Facial Landmark Detection Module
Uses MediaPipe to detect facial landmarks for expression analysis
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FacialLandmarkDetector:
    """Detects facial landmarks using MediaPipe"""
    
    def __init__(self, 
                 static_mode: bool = False,
                 max_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe face mesh
        
        Args:
            static_mode: If True, processes every frame. If False, tracks landmarks
            max_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # MediaPipe face mesh has 468 landmarks
        self.num_landmarks = 468
        
        # Define important landmark indices for expressions
        self.landmark_indices = {
            # Eyes
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            
            # Eyebrows
            'left_eyebrow': [70, 63, 105, 66, 107],
            'right_eyebrow': [336, 296, 334, 293, 300],
            
            # Mouth
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'lips': [0, 267, 37, 39, 40, 185, 191, 80, 81, 82, 13, 14, 15, 16, 17, 18, 200, 199, 175],
            
            # Nose
            'nose': [1, 2, 98, 327],
            
            # Head pose landmarks (ears, chin)
            'head_pose': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }
        
        logger.info("Facial Landmark Detector initialized")
    
    def detect_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect facial landmarks in the given frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dict containing landmarks and face info, or None if no face detected
        """
        if frame is None:
            return None
        
        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get the first face (assuming single face detection)
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract landmark coordinates
        landmarks = []
        height, width = frame.shape[:2]
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z
            landmarks.append((x, y, z))
        
        # Extract specific landmark groups
        landmark_data = {
            'all_landmarks': landmarks,
            'left_eye': [landmarks[i] for i in self.landmark_indices['left_eye']],
            'right_eye': [landmarks[i] for i in self.landmark_indices['right_eye']],
            'left_eyebrow': [landmarks[i] for i in self.landmark_indices['left_eyebrow']],
            'right_eyebrow': [landmarks[i] for i in self.landmark_indices['right_eyebrow']],
            'mouth': [landmarks[i] for i in self.landmark_indices['mouth']],
            'lips': [landmarks[i] for i in self.landmark_indices['lips']],
            'nose': [landmarks[i] for i in self.landmark_indices['nose']],
            'head_pose': [landmarks[i] for i in self.landmark_indices['head_pose']]
        }
        
        return landmark_data
    
    def calculate_eye_aspect_ratio(self, eye_landmarks: List[Tuple[int, int, float]]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) to detect eye openness
        
        Args:
            eye_landmarks: List of eye landmark coordinates
            
        Returns:
            float: Eye aspect ratio
        """
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
    
    def calculate_mouth_aspect_ratio(self, mouth_landmarks: List[Tuple[int, int, float]]) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR) to detect mouth openness
        
        Args:
            mouth_landmarks: List of mouth landmark coordinates
            
        Returns:
            float: Mouth aspect ratio
        """
        if len(mouth_landmarks) < 6:
            return 0.0
        
        # Vertical distances
        A = np.linalg.norm(np.array(mouth_landmarks[1]) - np.array(mouth_landmarks[7]))
        B = np.linalg.norm(np.array(mouth_landmarks[2]) - np.array(mouth_landmarks[6]))
        C = np.linalg.norm(np.array(mouth_landmarks[3]) - np.array(mouth_landmarks[5]))
        
        # Horizontal distance
        D = np.linalg.norm(np.array(mouth_landmarks[0]) - np.array(mouth_landmarks[4]))
        
        # Avoid division by zero
        if D == 0:
            return 0.0
        
        mar = (A + B + C) / (2.0 * D)
        return mar
    
    def calculate_eyebrow_position(self, eyebrow_landmarks: List[Tuple[int, int, float]]) -> float:
        """
        Calculate eyebrow position (raised/lowered)
        
        Args:
            eyebrow_landmarks: List of eyebrow landmark coordinates
            
        Returns:
            float: Average Y position (lower = raised eyebrow)
        """
        if not eyebrow_landmarks:
            return 0.0
        
        # Calculate average Y position
        avg_y = sum(landmark[1] for landmark in eyebrow_landmarks) / len(eyebrow_landmarks)
        return avg_y
    
    def estimate_head_pose(self, head_landmarks: List[Tuple[int, int, float]]) -> Dict[str, float]:
        """
        Estimate head pose (pitch, yaw, roll)
        
        Args:
            head_landmarks: List of head pose landmark coordinates
            
        Returns:
            Dict containing pitch, yaw, roll angles
        """
        if len(head_landmarks) < 6:
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
        # Simple head pose estimation using key landmarks
        # This is a simplified version - more sophisticated methods exist
        
        # Get key points for pose estimation
        left_ear = head_landmarks[0]  # Left ear
        right_ear = head_landmarks[1]  # Right ear
        nose = head_landmarks[2]  # Nose tip
        
        # Calculate roll (head tilt)
        roll = np.arctan2(right_ear[1] - left_ear[1], right_ear[0] - left_ear[0])
        
        # Calculate yaw (head turn)
        ear_center_x = (left_ear[0] + right_ear[0]) / 2
        yaw = np.arctan2(nose[0] - ear_center_x, nose[2] - (left_ear[2] + right_ear[2]) / 2)
        
        # Calculate pitch (head nod)
        pitch = np.arctan2(nose[1] - (left_ear[1] + right_ear[1]) / 2, nose[2] - (left_ear[2] + right_ear[2]) / 2)
        
        return {
            'pitch': np.degrees(pitch),
            'yaw': np.degrees(yaw),
            'roll': np.degrees(roll)
        }
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: Dict) -> np.ndarray:
        """
        Draw facial landmarks on the frame
        
        Args:
            frame: Input frame
            landmarks: Landmark data
            
        Returns:
            Frame with landmarks drawn
        """
        if landmarks is None:
            return frame
        
        # Draw eyes
        for eye_landmarks in [landmarks['left_eye'], landmarks['right_eye']]:
            for point in eye_landmarks:
                cv2.circle(frame, (point[0], point[1]), 2, (0, 255, 0), -1)
        
        # Draw eyebrows
        for eyebrow_landmarks in [landmarks['left_eyebrow'], landmarks['right_eyebrow']]:
            for point in eyebrow_landmarks:
                cv2.circle(frame, (point[0], point[1]), 2, (255, 0, 0), -1)
        
        # Draw mouth
        for point in landmarks['mouth']:
            cv2.circle(frame, (point[0], point[1]), 2, (0, 0, 255), -1)
        
        # Draw nose
        for point in landmarks['nose']:
            cv2.circle(frame, (point[0], point[1]), 2, (255, 255, 0), -1)
        
        return frame
    
    def release(self):
        """Release MediaPipe resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
            logger.info("MediaPipe resources released")

def test_landmark_detector():
    """Test function for landmark detection"""
    print("Testing facial landmark detection...")
    
    # Initialize detector
    detector = FacialLandmarkDetector()
    
    # Test with video capture
    from video_capture import VideoCapture
    
    with VideoCapture() as cap:
        if not cap.is_initialized:
            print("Camera not available for testing")
            return
        
        print("Press 'q' to quit, 's' to save frame")
        
        while True:
            frame = cap.read_frame()
            if frame is None:
                continue
            
            # Detect landmarks
            landmarks = detector.detect_landmarks(frame)
            
            if landmarks:
                # Calculate metrics
                left_ear = detector.calculate_eye_aspect_ratio(landmarks['left_eye'])
                right_ear = detector.calculate_eye_aspect_ratio(landmarks['right_eye'])
                mar = detector.calculate_mouth_aspect_ratio(landmarks['mouth'])
                head_pose = detector.estimate_head_pose(landmarks['head_pose'])
                
                # Display metrics on frame
                cv2.putText(frame, f"Left EAR: {left_ear:.3f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right EAR: {right_ear:.3f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.3f}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Pitch: {head_pose['pitch']:.1f}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw landmarks
                frame = detector.draw_landmarks(frame, landmarks)
            
            # Show frame
            cv2.imshow('Facial Landmark Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('test_frame.jpg', frame)
                print("Frame saved as test_frame.jpg")
        
        cv2.destroyAllWindows()
    
    detector.release()

if __name__ == "__main__":
    test_landmark_detector()
