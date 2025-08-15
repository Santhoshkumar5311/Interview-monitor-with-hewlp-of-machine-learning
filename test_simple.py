#!/usr/bin/env python3
"""
Simple Test for Interview Monitor Core Components
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_core_components():
    """Test core components"""
    print("Testing core components...")
    
    try:
        # Test facial detection
        from facial_detection.video_capture import VideoCapture
        from facial_detection.landmark_detector import FacialLandmarkDetector
        print("✅ Facial detection modules imported")
        
        # Test confidence analysis
        from confidence_analysis.confidence_classifier import ConfidenceClassifier
        print("✅ Confidence analysis modules imported")
        
        # Test transcription
        from transcription.speech_transcriber import SpeechTranscriber
        print("✅ Transcription modules imported")
        
        # Test UI (with fallback)
        from ui.interview_hud import InterviewHUD
        print("✅ UI modules imported")
        
        # Test logging
        from src.logging.bigquery_logger import BigQueryLogger
        print("✅ Logging modules imported")
        
        # Test enhanced face analyzer
        from facial_detection.enhanced_face_analyzer import EnhancedFaceAnalyzer
        print("✅ Enhanced face analyzer imported")
        
        # Test main monitor
        from interview_monitor import EnhancedInterviewMonitor
        print("✅ Main monitor imported")
        
        print("\n🎉 All core components are working!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_core_components()
    sys.exit(0 if success else 1)
