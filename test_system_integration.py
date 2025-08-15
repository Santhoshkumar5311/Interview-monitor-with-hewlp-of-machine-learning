#!/usr/bin/env python3
"""
System Integration Test for Interview Monitor
Tests all major components working together
"""

import sys
import os
import logging
import time
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_facial_detection():
    """Test facial detection components"""
    logger.info("Testing facial detection components...")
    
    try:
        from facial_detection.video_capture import VideoCapture
        from facial_detection.landmark_detector import FacialLandmarkDetector
        from facial_detection.expression_analyzer import ExpressionAnalyzer
        
        # Test video capture
        video_cap = VideoCapture()
        if video_cap.initialize():
            logger.info("✅ Video capture initialized successfully")
        else:
            logger.warning("⚠️ Video capture initialization failed (expected if no camera)")
        
        # Test landmark detector
        landmark_detector = FacialLandmarkDetector()
        logger.info("✅ Landmark detector created successfully")
        
        # Test expression analyzer
        expression_analyzer = ExpressionAnalyzer()
        logger.info("✅ Expression analyzer created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Facial detection test failed: {e}")
        return False

def test_confidence_analysis():
    """Test confidence analysis components"""
    logger.info("Testing confidence analysis components...")
    
    try:
        from confidence_analysis.confidence_classifier import ConfidenceClassifier
        from confidence_analysis.advanced_sentiment_analyzer import AdvancedSentimentAnalyzer
        from confidence_analysis.audio_prosody_analyzer import AudioProsodyAnalyzer
        
        # Test confidence classifier
        confidence_classifier = ConfidenceClassifier(use_advanced_sentiment=True)
        logger.info("✅ Confidence classifier created successfully")
        
        # Test advanced sentiment analyzer
        sentiment_analyzer = AdvancedSentimentAnalyzer()
        logger.info("✅ Advanced sentiment analyzer created successfully")
        
        # Test audio prosody analyzer
        prosody_analyzer = AudioProsodyAnalyzer()
        logger.info("✅ Audio prosody analyzer created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Confidence analysis test failed: {e}")
        return False

def test_transcription():
    """Test transcription components"""
    logger.info("Testing transcription components...")
    
    try:
        from transcription.speech_transcriber import SpeechTranscriber
        from transcription.whisper_asr import WhisperASR
        from transcription.subtitle_renderer import SubtitleRenderer
        
        # Test speech transcriber
        speech_transcriber = SpeechTranscriber()
        logger.info("✅ Speech transcriber created successfully")
        
        # Test Whisper ASR
        whisper_asr = WhisperASR()
        logger.info("✅ Whisper ASR created successfully")
        
        # Test subtitle renderer
        subtitle_renderer = SubtitleRenderer()
        logger.info("✅ Subtitle renderer created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Transcription test failed: {e}")
        return False

def test_ui_components():
    """Test UI components"""
    logger.info("Testing UI components...")
    
    try:
        from ui.interview_hud import InterviewHUD
        
        # Test HUD creation
        hud = InterviewHUD()
        logger.info("✅ Interview HUD created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ UI test failed: {e}")
        return False

def test_logging():
    """Test logging components"""
    logger.info("Testing logging components...")
    
    try:
        from logging.bigquery_logger import BigQueryLogger
        
        # Test BigQuery logger
        logger_instance = BigQueryLogger()
        logger.info("✅ BigQuery logger created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Logging test failed: {e}")
        return False

def test_enhanced_face_analyzer():
    """Test enhanced face analyzer"""
    logger.info("Testing enhanced face analyzer...")
    
    try:
        from facial_detection.enhanced_face_analyzer import EnhancedFaceAnalyzer
        
        # Test enhanced face analyzer
        enhanced_analyzer = EnhancedFaceAnalyzer()
        logger.info("✅ Enhanced face analyzer created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced face analyzer test failed: {e}")
        return False

def test_system_integration():
    """Test system integration"""
    logger.info("Testing system integration...")
    
    try:
        from interview_monitor import EnhancedInterviewMonitor
        
        # Test monitor creation (without running)
        monitor = EnhancedInterviewMonitor()
        logger.info("✅ Enhanced interview monitor created successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Interview Monitor System Integration Test")
    logger.info("=" * 60)
    
    tests = [
        ("Facial Detection", test_facial_detection),
        ("Enhanced Face Analyzer", test_enhanced_face_analyzer),
        ("Confidence Analysis", test_confidence_analysis),
        ("Transcription", test_transcription),
        ("UI Components", test_ui_components),
        ("Logging", test_logging),
        ("System Integration", test_system_integration),
    ]
    
    results = {}
    total_tests = len(tests)
    passed_tests = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            if test_func():
                results[test_name] = "PASSED"
                passed_tests += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                results[test_name] = "FAILED"
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            results[test_name] = f"ERROR: {str(e)}"
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status_icon = "✅" if result == "PASSED" else "❌"
        logger.info(f"{status_icon} {test_name}: {result}")
    
    logger.info(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! System is ready for use.")
        return True
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
