"""
Enhanced Interview Monitor with Advanced Sentiment Analysis
Integrates facial detection, speech transcription, and advanced confidence analysis
"""

import cv2
import numpy as np
import time
import logging
import os
import sys
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import our modules
from src.facial_detection.video_capture import VideoCapture
from src.facial_detection.landmark_detector import FacialLandmarkDetector
from src.facial_detection.expression_analyzer import ExpressionAnalyzer
from src.confidence_analysis.confidence_classifier import (
    ConfidenceClassifier, FusionMetrics, FacialMetrics, SpeechMetrics
)
from src.transcription.speech_transcriber import SpeechTranscriber
from src.transcription.subtitle_renderer import SubtitleRenderer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedInterviewMonitor:
    """Enhanced interview monitor with advanced sentiment analysis and fusion layer"""
    
    def __init__(self):
        """Initialize the enhanced interview monitor"""
        # Initialize components
        self.video_capture = VideoCapture()
        self.landmark_detector = FacialLandmarkDetector()
        self.expression_analyzer = ExpressionAnalyzer()
        self.confidence_classifier = ConfidenceClassifier(use_advanced_sentiment=True)
        self.speech_transcriber = SpeechTranscriber()
        self.subtitle_renderer = SubtitleRenderer()
        
        # State variables
        self.is_running = False
        self.frame_count = 0
        self.start_time = time.time()
        self.current_transcript = ""
        self.transcript_history = []
        self.current_fusion_metrics: Optional[FusionMetrics] = None
        
        # Performance tracking
        self.fps_history = []
        self.last_fps_update = time.time()
        
        # Initialize video capture
        if not self.video_capture.initialize():
            raise Exception("Failed to initialize video capture")
        
        logger.info("Enhanced Interview Monitor initialized successfully")
    
    def run(self):
        """Main monitoring loop"""
        self.is_running = True
        logger.info("Starting enhanced interview monitoring...")
        
        try:
            while self.is_running:
                # Process frame
                self._process_frame()
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_session_data()
                elif key == ord('h'):
                    self._show_help()
        
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        finally:
            self._cleanup()
    
    def _process_frame(self):
        """Process a single frame"""
        # Read frame
        ret, frame = self.video_capture.read()
        if not ret:
            logger.warning("Failed to read frame")
            return
        
        # Update frame count and FPS
        self.frame_count += 1
        self._update_fps()
        
        # Process facial landmarks and expressions
        landmarks = self.landmark_detector.detect_landmarks(frame)
        if landmarks is not None:
            # Analyze expressions
            expressions = self.expression_analyzer.analyze_expressions(landmarks)
            
            # Create facial metrics for fusion
            facial_metrics = self._create_facial_metrics(expressions, landmarks)
            
            # Process speech transcription
            transcript_chunk = self.speech_transcriber.get_latest_transcription()
            if transcript_chunk and transcript_chunk != self.current_transcript:
                self.current_transcript = transcript_chunk
                self.transcript_history.append({
                    'text': transcript_chunk,
                    'timestamp': datetime.now(),
                    'duration': 5.0  # Assuming 5-second chunks
                })
                
                # Create speech metrics (mock for now - will be enhanced with pyAudioAnalysis)
                speech_metrics = self._create_speech_metrics(transcript_chunk)
                
                # Analyze chunk with fusion layer
                self.current_fusion_metrics = self.confidence_classifier.analyze_chunk(
                    transcript=transcript_chunk,
                    facial_metrics=facial_metrics,
                    speech_metrics=speech_metrics,
                    chunk_duration=5.0
                )
                
                logger.info(f"New transcript: {transcript_chunk[:50]}...")
                logger.info(f"Fusion metrics: Confidence={self.current_fusion_metrics.confidence_level.value}, "
                           f"Relevance={self.current_fusion_metrics.relevance_level.value}")
            
            # Draw facial landmarks (optional - can be disabled)
            # self.landmark_detector.draw_landmarks(frame, landmarks)
            
            # Draw expression analysis (optional - can be disabled)
            # self._draw_expression_info(frame, expressions)
        
        # Render subtitles
        if self.current_transcript:
            self.subtitle_renderer.render_subtitle(frame, self.current_transcript)
            self.subtitle_renderer.render_subtitle_history(frame, self.transcript_history[-3:])
        
        # Draw enhanced system info with fusion metrics
        self._draw_enhanced_system_info(frame)
        
        # Display frame
        cv2.imshow('Enhanced Interview Monitor', frame)
    
    def _create_facial_metrics(self, expressions: Dict, landmarks: Dict) -> FacialMetrics:
        """Create FacialMetrics from expression analysis"""
        # Extract metrics from expressions
        smile_intensity = expressions.get('smile', {}).get('intensity', 0.0)
        frown_intensity = expressions.get('frown', {}).get('intensity', 0.0)
        eyebrow_raise = expressions.get('raised_eyebrows', {}).get('intensity', 0.0)
        
        # Calculate eye contact using the landmark groups
        try:
            if 'left_eye' in landmarks and 'right_eye' in landmarks:
                # Calculate center of left and right eyes
                left_eye_center = np.mean(landmarks['left_eye'], axis=0)
                right_eye_center = np.mean(landmarks['right_eye'], axis=0)
                eye_center = (left_eye_center + right_eye_center) / 2
                
                # Calculate distance from frame center (assuming 640x480 frame)
                frame_center = np.array([320, 240])  # Center of 640x480 frame
                distance_from_center = np.linalg.norm(eye_center[:2] - frame_center)  # Use only x,y coordinates
                max_distance = np.linalg.norm(np.array([0, 0]) - frame_center)
                eye_contact = max(0, 1 - (distance_from_center / max_distance))
            else:
                eye_contact = 0.5
        except (IndexError, ValueError, KeyError):
            # Fallback if landmark indexing fails
            eye_contact = 0.5
        
        # Head pose estimation (simplified)
        head_tilt = 0.0  # Will be enhanced with proper head pose estimation
        head_nod = 0.0
        
        # Blink rate (simplified)
        blink_rate = 0.3  # Will be enhanced with temporal analysis
        
        # Mouth open detection
        mouth_open = expressions.get('mouth_open', {}).get('intensity', 0.0)
        
        return FacialMetrics(
            smile_intensity=smile_intensity,
            frown_intensity=frown_intensity,
            eyebrow_raise=eyebrow_raise,
            eye_contact=eye_contact,
            head_tilt=head_tilt,
            head_nod=head_nod,
            blink_rate=blink_rate,
            mouth_open=mouth_open
        )
    
    def _create_speech_metrics(self, transcript: str) -> SpeechMetrics:
        """Create SpeechMetrics from transcript (will be enhanced with audio analysis)"""
        # Mock metrics for now - will be replaced with pyAudioAnalysis
        word_count = len(transcript.split())
        
        # Volume (mock - will come from audio analysis)
        volume = 0.8
        
        # Pace (words per second - simplified)
        pace = min(1.0, word_count / 15.0)  # Normalize to 15 words per 5 seconds
        
        # Clarity (based on word length and complexity)
        words = transcript.split()
        avg_word_length = np.mean([len(word) for word in words]) if words else 5.0
        clarity = max(0.3, 1 - (avg_word_length - 5) / 10)  # 5 letters = optimal
        
        # Tone (mock - will come from audio analysis)
        tone = 0.7
        
        # Pause frequency (mock - will come from audio analysis)
        pause_frequency = 0.2
        
        # Pitch variation (mock - will come from audio analysis)
        pitch_variation = 0.6
        
        return SpeechMetrics(
            volume=volume,
            pace=pace,
            clarity=clarity,
            tone=tone,
            pause_frequency=pause_frequency,
            pitch_variation=pitch_variation
        )
    
    def _draw_enhanced_system_info(self, frame: np.ndarray):
        """Draw enhanced system information with fusion metrics"""
        height, width = frame.shape[:2]
        
        # Create info box at bottom
        info_height = 120
        info_y = height - info_height - 10
        
        # Draw background box
        cv2.rectangle(frame, (10, info_y), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, info_y), (width - 10, height - 10), (255, 255, 255), 2)
        
        # Calculate current FPS
        current_fps = self._get_current_fps()
        
        # Basic system info
        y_offset = info_y + 25
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Frame: {self.frame_count}", (120, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Transcription status
        trans_status = "Active" if self.current_transcript else "Waiting..."
        cv2.putText(frame, f"Transcription: {trans_status}", (250, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Fusion metrics (if available)
        if self.current_fusion_metrics:
            y_offset += 30
            
            # Confidence and Relevance
            conf_color = self._get_confidence_color(self.current_fusion_metrics.confidence_score)
            rel_color = self._get_confidence_color(self.current_fusion_metrics.relevance_score)
            
            cv2.putText(frame, f"Confidence: {self.current_fusion_metrics.confidence_level.value.upper()}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
            
            cv2.putText(frame, f"Relevance: {self.current_fusion_metrics.relevance_level.value.upper()}", 
                       (250, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rel_color, 2)
            
            # Sentiment and Emotion
            y_offset += 25
            sentiment_color = self._get_sentiment_color(self.current_fusion_metrics.sentiment_score)
            
            cv2.putText(frame, f"Sentiment: {self.current_fusion_metrics.sentiment_score:.2f}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sentiment_color, 2)
            
            cv2.putText(frame, f"Emotion: {self.current_fusion_metrics.emotion_label.upper()}", 
                       (250, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Toxicity warning (if high)
            if self.current_fusion_metrics.toxicity_score > 0.5:
                cv2.putText(frame, "⚠ TOXICITY WARNING", (450, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Help text
        y_offset += 30
        cv2.putText(frame, "Press 'q' to quit, 's' to save session, 'h' for help", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    def _get_confidence_color(self, score: float) -> Tuple[int, int, int]:
        """Get color for confidence/relevance score"""
        if score < 0.4:
            return (0, 0, 255)  # Red
        elif score < 0.6:
            return (0, 165, 255)  # Orange
        elif score < 0.8:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 255, 0)  # Green
    
    def _get_sentiment_color(self, score: float) -> Tuple[int, int, int]:
        """Get color for sentiment score"""
        if score < -0.3:
            return (0, 0, 255)  # Red (negative)
        elif score < 0.3:
            return (255, 255, 255)  # White (neutral)
        else:
            return (0, 255, 0)  # Green (positive)
    
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:  # Update every second
            fps = self.frame_count / (current_time - self.start_time)
            self.fps_history.append(fps)
            if len(self.fps_history) > 10:
                self.fps_history.pop(0)
            self.last_fps_update = current_time
    
    def _get_current_fps(self) -> float:
        """Get current FPS"""
        if self.fps_history:
            return np.mean(self.fps_history)
        return 0.0
    
    def _save_session_data(self):
        """Save session data and performance log"""
        try:
            # Export performance log
            log_filename = self.confidence_classifier.export_performance_log()
            
            # Get session summary
            summary = self.confidence_classifier.get_session_summary()
            
            # Save transcript history
            transcript_filename = f"transcript_{self.confidence_classifier.session_id}.json"
            import json
            with open(transcript_filename, 'w') as f:
                json.dump(self.transcript_history, f, indent=2, default=str)
            
            logger.info(f"Session data saved:")
            logger.info(f"  - Performance log: {log_filename}")
            logger.info(f"  - Transcript: {transcript_filename}")
            logger.info(f"  - Session summary: {summary}")
            
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
    
    def _show_help(self):
        """Show help information"""
        help_text = """
Enhanced Interview Monitor Help
==============================

Controls:
- 'q': Quit monitoring
- 's': Save session data
- 'h': Show this help

Features:
- Real-time facial expression analysis
- Live speech transcription
- Advanced sentiment analysis (emotion, toxicity, politeness)
- Confidence and relevance scoring
- Fusion layer with EMA smoothing
- Performance logging and export

Metrics Displayed:
- Confidence Level: Based on facial expressions, speech, and sentiment
- Relevance Level: Content relevance and quality
- Sentiment Score: Emotional tone (-1 to 1)
- Emotion Label: Primary detected emotion
- Toxicity Warning: Alerts for harmful content
- Real-time FPS and frame count
        """
        print(help_text)
    
    def _cleanup(self):
        """Clean up resources"""
        self.is_running = False
        
        # Save final session data
        self._save_session_data()
        
        # Release resources
        self.video_capture.release()
        cv2.destroyAllWindows()
        
        logger.info("Enhanced Interview Monitor cleanup completed")

def main():
    """Main function to run the enhanced interview monitor"""
    try:
        monitor = EnhancedInterviewMonitor()
        monitor.run()
    except Exception as e:
        logger.error(f"Failed to run enhanced interview monitor: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
