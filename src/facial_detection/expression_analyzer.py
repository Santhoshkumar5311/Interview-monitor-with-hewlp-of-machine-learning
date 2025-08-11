"""
Expression Analyzer Module
Analyzes facial landmarks to detect expressions and emotions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from .landmark_detector import FacialLandmarkDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpressionAnalyzer:
    """Analyzes facial expressions using landmark data"""
    
    def __init__(self):
        """Initialize expression analyzer with thresholds"""
        
        # Thresholds for expression detection (these can be tuned)
        self.thresholds = {
            'smile': {
                'mouth_corner_ratio': 0.6,  # Ratio of mouth corner distances
                'mouth_openness': 0.3,      # Maximum mouth openness for smile
            },
            'frown': {
                'mouth_corner_ratio': 1.4,  # Ratio of mouth corner distances
                'mouth_openness': 0.3,      # Maximum mouth openness for frown
            },
            'raised_eyebrow': {
                'eyebrow_ratio': 0.9,      # Ratio of eyebrow positions
            },
            'eye_blink': {
                'ear_threshold': 0.21,     # Eye aspect ratio threshold
            }
        }
        
        # Baseline measurements (will be updated during calibration)
        self.baseline_measurements = {}
        self.is_calibrated = False
        
        logger.info("Expression Analyzer initialized")
    
    def calibrate(self, landmarks: Dict, num_frames: int = 30) -> bool:
        """
        Calibrate the analyzer with baseline measurements
        
        Args:
            landmarks: Landmark data from detector
            num_frames: Number of frames to use for calibration
            
        Returns:
            bool: True if calibration successful
        """
        if not landmarks:
            return False
        
        # This would typically collect measurements over multiple frames
        # For now, we'll use the current frame as baseline
        try:
            # Calculate baseline measurements
            left_ear = self._calculate_eye_aspect_ratio(landmarks['left_eye'])
            right_ear = self._calculate_eye_aspect_ratio(landmarks['right_eye'])
            mar = self._calculate_mouth_aspect_ratio(landmarks['mouth'])
            
            self.baseline_measurements = {
                'left_ear': left_ear,
                'right_ear': right_ear,
                'mar': mar,
                'left_eyebrow_y': self._get_eyebrow_position(landmarks['left_eyebrow']),
                'right_eyebrow_y': self._get_eyebrow_position(landmarks['right_eyebrow'])
            }
            
            self.is_calibrated = True
            logger.info("Expression analyzer calibrated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            return False
    
    def analyze_expressions(self, landmarks: Dict) -> Dict:
        """
        Analyze facial expressions from landmarks
        
        Args:
            landmarks: Landmark data from detector
            
        Returns:
            Dict containing detected expressions and confidence scores
        """
        if not landmarks:
            return {}
        
        try:
            # Calculate current measurements
            left_ear = self._calculate_eye_aspect_ratio(landmarks['left_eye'])
            right_ear = self._calculate_eye_aspect_ratio(landmarks['right_eye'])
            mar = self._calculate_mouth_aspect_ratio(landmarks['mouth'])
            left_eyebrow_y = self._get_eyebrow_position(landmarks['left_eyebrow'])
            right_eyebrow_y = self._get_eyebrow_position(landmarks['right_eyebrow'])
            
            # Analyze expressions
            expressions = {
                'smile': self._detect_smile(landmarks, mar),
                'frown': self._detect_frown(landmarks, mar),
                'raised_eyebrows': self._detect_raised_eyebrows(left_eyebrow_y, right_eyebrow_y),
                'eye_blink': self._detect_eye_blink(left_ear, right_ear),
                'head_tilt': self._detect_head_tilt(landmarks),
                'mouth_open': self._detect_mouth_open(mar)
            }
            
            return expressions
            
        except Exception as e:
            logger.error(f"Expression analysis failed: {str(e)}")
            return {}
    
    def _calculate_eye_aspect_ratio(self, eye_landmarks: List[Tuple[int, int, float]]) -> float:
        """Calculate Eye Aspect Ratio (EAR)"""
        if len(eye_landmarks) < 6:
            return 0.0
        
        # Vertical distances
        A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        
        # Horizontal distance
        C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        if C == 0:
            return 0.0
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def _calculate_mouth_aspect_ratio(self, mouth_landmarks: List[Tuple[int, int, float]]) -> float:
        """Calculate Mouth Aspect Ratio (MAR)"""
        if len(mouth_landmarks) < 8:
            return 0.0
        
        # Vertical distances
        A = np.linalg.norm(np.array(mouth_landmarks[1]) - np.array(mouth_landmarks[7]))
        B = np.linalg.norm(np.array(mouth_landmarks[2]) - np.array(mouth_landmarks[6]))
        C = np.linalg.norm(np.array(mouth_landmarks[3]) - np.array(mouth_landmarks[5]))
        
        # Horizontal distance
        D = np.linalg.norm(np.array(mouth_landmarks[0]) - np.array(mouth_landmarks[4]))
        
        if D == 0:
            return 0.0
        
        mar = (A + B + C) / (2.0 * D)
        return mar
    
    def _get_eyebrow_position(self, eyebrow_landmarks: List[Tuple[int, int, float]]) -> float:
        """Get average Y position of eyebrow landmarks"""
        if not eyebrow_landmarks:
            return 0.0
        
        avg_y = sum(landmark[1] for landmark in eyebrow_landmarks) / len(eyebrow_landmarks)
        return avg_y
    
    def _detect_smile(self, landmarks: Dict, mar: float) -> Dict:
        """Detect smile expression"""
        if not landmarks.get('mouth'):
            return {'detected': False, 'confidence': 0.0}
        
        mouth_landmarks = landmarks['mouth']
        
        # Calculate mouth corner distances
        left_corner = mouth_landmarks[0]  # Left corner
        right_corner = mouth_landmarks[6]  # Right corner
        
        # Calculate vertical distance from corners to center
        center_y = (left_corner[1] + right_corner[1]) / 2
        left_vertical = abs(left_corner[1] - center_y)
        right_vertical = abs(right_corner[1] - center_y)
        
        # Smile detection: corners should be higher than center
        smile_score = 0.0
        
        if left_corner[1] < center_y and right_corner[1] < center_y:
            smile_score += 0.5
        
        # Check mouth openness (shouldn't be too open for smile)
        if mar < self.thresholds['smile']['mouth_openness']:
            smile_score += 0.3
        
        # Additional smile features
        if len(mouth_landmarks) >= 8:
            # Check if mouth corners are pulled outward
            horizontal_distance = np.linalg.norm(
                np.array(left_corner) - np.array(right_corner)
            )
            if horizontal_distance > 50:  # Threshold for wide smile
                smile_score += 0.2
        
        detected = smile_score > 0.6
        confidence = min(smile_score, 1.0)
        
        return {
            'detected': detected,
            'confidence': confidence,
            'score': smile_score
        }
    
    def _detect_frown(self, landmarks: Dict, mar: float) -> Dict:
        """Detect frown expression"""
        if not landmarks.get('mouth'):
            return {'detected': False, 'confidence': 0.0}
        
        mouth_landmarks = landmarks['mouth']
        
        # Calculate mouth corner distances
        left_corner = mouth_landmarks[0]  # Left corner
        right_corner = mouth_landmarks[6]  # Right corner
        
        # Calculate vertical distance from corners to center
        center_y = (left_corner[1] + right_corner[1]) / 2
        left_vertical = abs(left_corner[1] - center_y)
        right_vertical = abs(right_corner[1] - center_y)
        
        # Frown detection: corners should be lower than center
        frown_score = 0.0
        
        if left_corner[1] > center_y and right_corner[1] > center_y:
            frown_score += 0.5
        
        # Check mouth openness
        if mar < self.thresholds['frown']['mouth_openness']:
            frown_score += 0.3
        
        # Check if mouth corners are pulled down
        if left_vertical > 10 and right_vertical > 10:
            frown_score += 0.2
        
        detected = frown_score > 0.6
        confidence = min(frown_score, 1.0)
        
        return {
            'detected': detected,
            'confidence': confidence,
            'score': frown_score
        }
    
    def _detect_raised_eyebrows(self, left_eyebrow_y: float, right_eyebrow_y: float) -> Dict:
        """Detect raised eyebrows"""
        if not self.is_calibrated:
            return {'detected': False, 'confidence': 0.0}
        
        # Compare current eyebrow position with baseline
        baseline_left = self.baseline_measurements.get('left_eyebrow_y', 0)
        baseline_right = self.baseline_measurements.get('right_eyebrow_y', 0)
        
        # Calculate how much eyebrows are raised
        left_raised = baseline_left - left_eyebrow_y  # Positive = raised
        right_raised = baseline_right - right_eyebrow_y
        
        # Threshold for raised eyebrows
        raise_threshold = 15  # pixels
        
        left_score = max(0, min(1, left_raised / raise_threshold))
        right_score = max(0, min(1, right_raised / raise_threshold))
        
        # Average score for both eyebrows
        raise_score = (left_score + right_score) / 2
        
        detected = raise_score > 0.3
        confidence = min(raise_score, 1.0)
        
        return {
            'detected': detected,
            'confidence': confidence,
            'score': raise_score,
            'left_raised': left_raised,
            'right_raised': right_raised
        }
    
    def _detect_eye_blink(self, left_ear: float, right_ear: float) -> Dict:
        """Detect eye blink"""
        # Use the lower EAR value (more closed eye)
        min_ear = min(left_ear, right_ear)
        
        # Check if eyes are closed
        blink_threshold = self.thresholds['eye_blink']['ear_threshold']
        blink_score = max(0, (blink_threshold - min_ear) / blink_threshold)
        
        detected = blink_score > 0.5
        confidence = min(blink_score, 1.0)
        
        return {
            'detected': detected,
            'confidence': confidence,
            'score': blink_score,
            'left_ear': left_ear,
            'right_ear': right_ear
        }
    
    def _detect_head_tilt(self, landmarks: Dict) -> Dict:
        """Detect head tilt (roll)"""
        if not landmarks.get('head_pose'):
            return {'detected': False, 'confidence': 0.0, 'angle': 0.0}
        
        # Simple head tilt detection using ear positions
        head_landmarks = landmarks['head_pose']
        
        if len(head_landmarks) < 2:
            return {'detected': False, 'confidence': 0.0, 'angle': 0.0}
        
        # Use ear landmarks for tilt calculation
        left_ear = head_landmarks[0]  # Left ear
        right_ear = head_landmarks[1]  # Right ear
        
        # Calculate tilt angle
        dx = right_ear[0] - left_ear[0]
        dy = right_ear[1] - left_ear[1]
        
        if dx == 0:
            angle = 0.0
        else:
            angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Normalize angle to -90 to 90 degrees
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        
        # Detect significant tilt
        tilt_threshold = 15  # degrees
        tilt_score = min(1.0, abs(angle) / tilt_threshold)
        
        detected = tilt_score > 0.3
        confidence = min(tilt_score, 1.0)
        
        return {
            'detected': detected,
            'confidence': confidence,
            'score': tilt_score,
            'angle': angle
        }
    
    def _detect_mouth_open(self, mar: float) -> Dict:
        """Detect if mouth is open"""
        # Threshold for mouth openness
        open_threshold = 0.4
        
        open_score = max(0, (mar - open_threshold) / open_threshold)
        
        detected = open_score > 0.3
        confidence = min(open_score, 1.0)
        
        return {
            'detected': detected,
            'confidence': confidence,
            'score': open_score,
            'mar': mar
        }
    
    def get_expression_summary(self, expressions: Dict) -> str:
        """Get a human-readable summary of detected expressions"""
        if not expressions:
            return "No expressions detected"
        
        summary_parts = []
        
        if expressions.get('smile', {}).get('detected'):
            confidence = expressions['smile']['confidence']
            summary_parts.append(f"Smile ({confidence:.1%})")
        
        if expressions.get('frown', {}).get('detected'):
            confidence = expressions['frown']['confidence']
            summary_parts.append(f"Frown ({confidence:.1%})")
        
        if expressions.get('raised_eyebrows', {}).get('detected'):
            confidence = expressions['raised_eyebrows']['confidence']
            summary_parts.append(f"Raised eyebrows ({confidence:.1%})")
        
        if expressions.get('eye_blink', {}).get('detected'):
            confidence = expressions['eye_blink']['confidence']
            summary_parts.append(f"Eye blink ({confidence:.1%})")
        
        if expressions.get('head_tilt', {}).get('detected'):
            angle = expressions['head_tilt']['angle']
            summary_parts.append(f"Head tilt ({angle:.1f}°)")
        
        if expressions.get('mouth_open', {}).get('detected'):
            confidence = expressions['mouth_open']['confidence']
            summary_parts.append(f"Mouth open ({confidence:.1%})")
        
        if not summary_parts:
            return "Neutral expression"
        
        return ", ".join(summary_parts)

def test_expression_analyzer():
    """Test function for expression analyzer"""
    print("Testing expression analyzer...")
    
    # This would typically test with real video data
    # For now, we'll just test the class initialization
    analyzer = ExpressionAnalyzer()
    print("✓ Expression analyzer initialized successfully")
    
    # Test thresholds
    print(f"✓ Smile threshold: {analyzer.thresholds['smile']}")
    print(f"✓ Frown threshold: {analyzer.thresholds['frown']}")
    print(f"✓ Eye blink threshold: {analyzer.thresholds['eye_blink']}")
    
    print("\nExpression analyzer is ready for use!")

if __name__ == "__main__":
    test_expression_analyzer()
