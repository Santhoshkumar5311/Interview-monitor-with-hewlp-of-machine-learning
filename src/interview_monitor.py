"""
Interview Monitor - Main Application
Integrates facial detection, confidence analysis, and live transcription
"""

import cv2
import numpy as np
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from facial_detection.video_capture import VideoCapture
from facial_detection.landmark_detector import FacialLandmarkDetector
from facial_detection.expression_analyzer import ExpressionAnalyzer
from confidence_analysis.confidence_classifier import ConfidenceClassifier
from confidence_analysis.sentiment_analyzer import SentimentAnalyzer
from confidence_analysis.speech_analyzer import SpeechAnalyzer
from transcription.speech_transcriber import SpeechTranscriber
from transcription.subtitle_renderer import SubtitleRenderer, SubtitleStyle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterviewMonitor:
    """Main interview monitoring system"""
    
    def __init__(self, 
                 enable_transcription: bool = True,
                 enable_confidence_analysis: bool = True,
                 subtitle_style: Optional[SubtitleStyle] = None):
        """
        Initialize interview monitor
        
        Args:
            enable_transcription: Enable live speech transcription
            enable_confidence_analysis: Enable confidence analysis
            subtitle_style: Custom subtitle styling
        """
        self.enable_transcription = enable_transcription
        self.enable_confidence_analysis = enable_confidence_analysis
        
        # Initialize components
        self.video_capture = None
        self.landmark_detector = None
        self.expression_analyzer = None
        self.confidence_classifier = None
        self.sentiment_analyzer = None
        self.speech_analyzer = None
        self.speech_transcriber = None
        self.subtitle_renderer = None
        
        # State variables
        self.is_running = False
        self.current_frame = None
        self.current_landmarks = None
        self.current_expressions = None
        self.current_confidence = None
        self.current_transcription = ""
        
        # Timing
        self.start_time = 0.0
        self.frame_count = 0
        self.fps = 0.0
        
        # Initialize all components
        self._initialize_components(subtitle_style)
        
        logger.info("Interview Monitor initialized")
    
    def _initialize_components(self, subtitle_style: Optional[SubtitleStyle]):
        """Initialize all system components"""
        try:
            # 1. Video capture
            self.video_capture = VideoCapture()
            logger.info("✓ Video capture initialized")
            
            # 2. Facial landmark detection
            self.landmark_detector = FacialLandmarkDetector()
            logger.info("✓ Landmark detector initialized")
            
            # 3. Expression analysis
            self.expression_analyzer = ExpressionAnalyzer()
            logger.info("✓ Expression analyzer initialized")
            
            # 4. Confidence analysis (if enabled)
            if self.enable_confidence_analysis:
                self.confidence_classifier = ConfidenceClassifier()
                self.sentiment_analyzer = SentimentAnalyzer()
                self.speech_analyzer = SpeechAnalyzer()
                logger.info("✓ Confidence analysis components initialized")
            
            # 5. Speech transcription (if enabled)
            if self.enable_transcription:
                try:
                    self.speech_transcriber = SpeechTranscriber()
                    logger.info("✓ Speech transcriber initialized")
                except Exception as e:
                    logger.warning(f"Speech transcription disabled: {e}")
                    self.enable_transcription = False
            
            # 6. Subtitle rendering
            self.subtitle_renderer = SubtitleRenderer(subtitle_style)
            logger.info("✓ Subtitle renderer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def start_monitoring(self):
        """Start the interview monitoring system"""
        if self.is_running:
            logger.warning("Monitoring already running")
            return False
        
        try:
            self.is_running = True
            self.start_time = time.time()
            self.frame_count = 0
            
            # Start transcription if enabled
            if self.enable_transcription and self.speech_transcriber:
                self.speech_transcriber.on_transcription_update = self._on_transcription_update
                self.speech_transcriber.on_transcription_complete = self._on_transcription_complete
                self.speech_transcriber.start_transcription()
                logger.info("✓ Transcription started")
            
            logger.info("Interview monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            self.is_running = False
            return False
    
    def stop_monitoring(self):
        """Stop the interview monitoring system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop transcription
        if self.speech_transcriber:
            self.speech_transcriber.stop_transcription()
        
        logger.info("Interview monitoring stopped")
    
    def run_monitoring_loop(self):
        """Main monitoring loop"""
        if not self.is_running:
            logger.error("Monitoring not started")
            return
        
        logger.info("Starting monitoring loop...")
        logger.info("Controls:")
        logger.info("- Press 'q' to quit")
        logger.info("- Press 'c' to calibrate expression analyzer")
        logger.info("- Press 's' to save current frame")
        logger.info("- Press 't' to toggle transcription")
        logger.info("- Press 'h' to show/hide confidence display")
        
        show_confidence = True
        transcription_enabled = self.enable_transcription
        
        try:
            while self.is_running:
                # Capture frame
                ret, frame = self.video_capture.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    continue
                
                self.current_frame = frame
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                if elapsed_time > 0:
                    self.fps = self.frame_count / elapsed_time
                
                # Process frame
                processed_frame = self._process_frame(frame, current_time)
                
                # Display frame
                cv2.imshow("Interview Monitor", processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self._calibrate_expression_analyzer()
                elif key == ord('s'):
                    self._save_current_frame()
                elif key == ord('t'):
                    transcription_enabled = self._toggle_transcription()
                elif key == ord('h'):
                    show_confidence = not show_confidence
                
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        finally:
            self.stop_monitoring()
            cv2.destroyAllWindows()
    
    def _process_frame(self, frame: np.ndarray, current_time: float) -> np.ndarray:
        """Process a single video frame"""
        try:
            # 1. Detect facial landmarks
            landmarks = self.landmark_detector.detect_landmarks(frame)
            self.current_landmarks = landmarks
            
            if landmarks:
                # 2. Analyze expressions
                expressions = self.expression_analyzer.analyze_expressions(landmarks)
                self.current_expressions = expressions
                
                # 3. Analyze confidence (if enabled)
                if self.enable_confidence_analysis:
                    confidence_metrics = self._analyze_confidence(expressions, landmarks)
                    self.current_confidence = confidence_metrics
                
                # 4. Draw landmarks and expressions
                frame = self.landmark_detector.draw_landmarks(frame, landmarks)
                frame = self._draw_expression_info(frame, expressions)
                
                # 5. Draw confidence info (if enabled and shown)
                if self.enable_confidence_analysis and hasattr(self, 'show_confidence'):
                    frame = self._draw_confidence_info(frame, confidence_metrics)
            
            # 6. Render subtitles
            if self.enable_transcription:
                frame = self.subtitle_renderer.render_subtitle(frame, current_time)
                frame = self.subtitle_renderer.render_subtitle_history(frame, current_time)
            
            # 7. Draw system info
            frame = self._draw_system_info(frame, current_time)
            
            return frame
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return frame
    
    def _analyze_confidence(self, expressions: Dict, landmarks: Dict) -> Optional[Dict]:
        """Analyze overall confidence from expressions and other factors"""
        try:
            if not self.enable_confidence_analysis:
                return None
            
            # Extract head pose from landmarks
            head_pose = self._extract_head_pose(landmarks)
            
            # Get speech characteristics (placeholder for now)
            speech_data = {
                'clear_articulation': 0.7,
                'steady_pace': 0.8,
                'appropriate_volume': 0.8
            }
            
            # Get sentiment from transcription
            sentiment_score = 0.0
            if self.current_transcription and self.sentiment_analyzer:
                sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(self.current_transcription)
                sentiment_score = sentiment_analysis['overall_sentiment']
            
            # Classify confidence
            confidence_metrics = self.confidence_classifier.classify_confidence(
                expressions, head_pose, speech_data, sentiment_score
            )
            
            return confidence_metrics
            
        except Exception as e:
            logger.error(f"Confidence analysis failed: {e}")
            return None
    
    def _extract_head_pose(self, landmarks: Dict) -> Dict:
        """Extract head pose information from landmarks"""
        try:
            # This is a simplified head pose extraction
            # In a real implementation, you'd use more sophisticated methods
            
            if not landmarks.get('head_pose'):
                return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
            
            # Placeholder head pose calculation
            head_pose = {
                'pitch': 0.0,   # Forward/backward tilt
                'yaw': 0.0,     # Left/right turn
                'roll': 0.0     # Left/right tilt
            }
            
            return head_pose
            
        except Exception as e:
            logger.error(f"Head pose extraction failed: {e}")
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
    
    def _draw_expression_info(self, frame: np.ndarray, expressions: Dict) -> np.ndarray:
        """Draw expression information on frame"""
        if not expressions:
            return frame
        
        # Draw expression summary
        summary = self.expression_analyzer.get_expression_summary(expressions)
        
        # Position in top-left corner
        y_offset = 30
        cv2.putText(frame, f"Expressions: {summary}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def _draw_confidence_info(self, frame: np.ndarray, confidence_metrics: Dict) -> np.ndarray:
        """Draw confidence information on frame"""
        if not confidence_metrics:
            return frame
        
        # Draw confidence summary
        summary = self.confidence_classifier.get_confidence_summary(confidence_metrics)
        lines = summary.split('\n')
        
        # Position in top-right corner
        x_offset = frame.shape[1] - 300
        y_offset = 30
        
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x_offset, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return frame
    
    def _draw_system_info(self, frame: np.ndarray, current_time: float) -> np.ndarray:
        """Draw system information on frame"""
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Frame count
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Transcription status
        status = "ON" if self.enable_transcription else "OFF"
        cv2.putText(frame, f"Transcription: {status}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _calibrate_expression_analyzer(self):
        """Calibrate the expression analyzer"""
        if self.current_landmarks:
            success = self.expression_analyzer.calibrate(self.current_landmarks)
            if success:
                logger.info("Expression analyzer calibrated successfully")
            else:
                logger.warning("Expression analyzer calibration failed")
        else:
            logger.warning("No landmarks available for calibration")
    
    def _save_current_frame(self):
        """Save current frame to file"""
        if self.current_frame is not None:
            timestamp = int(time.time())
            filename = f"interview_frame_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            logger.info(f"Frame saved as {filename}")
        else:
            logger.warning("No frame available to save")
    
    def _toggle_transcription(self) -> bool:
        """Toggle transcription on/off"""
        if not self.enable_transcription:
            logger.info("Transcription not available")
            return False
        
        if self.speech_transcriber.is_transcribing:
            self.speech_transcriber.stop_transcription()
            logger.info("Transcription stopped")
            return False
        else:
            self.speech_transcriber.start_transcription()
            logger.info("Transcription started")
            return True
    
    def _on_transcription_update(self, text: str):
        """Callback for transcription updates"""
        self.current_transcription = text
        self.subtitle_renderer.update_subtitle(text, is_final=False)
    
    def _on_transcription_complete(self, transcription_entry: Dict):
        """Callback for completed transcriptions"""
        text = transcription_entry['text']
        self.current_transcription = text
        self.subtitle_renderer.update_subtitle(text, is_final=True)
        
        logger.info(f"Transcription complete: {text}")
    
    def cleanup(self):
        """Clean up all resources"""
        self.stop_monitoring()
        
        if self.video_capture:
            self.video_capture.release()
        
        if self.speech_transcriber:
            self.speech_transcriber.cleanup()
        
        logger.info("Interview Monitor cleaned up")

def test_interview_monitor():
    """Test function for interview monitor"""
    print("Testing interview monitor...")
    
    try:
        # Create custom subtitle style
        subtitle_style = SubtitleStyle(
            font_scale=1.0,
            font_color=(255, 255, 0),  # Yellow
            background_color=(0, 0, 128),  # Dark blue
            background_opacity=0.8
        )
        
        # Initialize monitor
        monitor = InterviewMonitor(
            enable_transcription=True,
            enable_confidence_analysis=True,
            subtitle_style=subtitle_style
        )
        print("✓ Interview monitor initialized")
        
        # Test component access
        print(f"✓ Video capture: {monitor.video_capture is not None}")
        print(f"✓ Landmark detector: {monitor.landmark_detector is not None}")
        print(f"✓ Expression analyzer: {monitor.expression_analyzer is not None}")
        print(f"✓ Confidence classifier: {monitor.confidence_classifier is not None}")
        print(f"✓ Speech transcriber: {monitor.speech_transcriber is not None}")
        print(f"✓ Subtitle renderer: {monitor.subtitle_renderer is not None}")
        
        print("\nInterview monitor is ready for use!")
        print("Note: Google Cloud credentials required for transcription")
        
        monitor.cleanup()
        
    except Exception as e:
        print(f"❌ Interview monitor test failed: {e}")
        print("This is expected if some components are not fully configured")

if __name__ == "__main__":
    test_interview_monitor()
