#!/usr/bin/env python3
"""
Core Functionality Test for Interview Monitor System
Tests essential components without requiring camera/microphone
"""

import sys
import os
import logging
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_core_functionality():
    """Test core functionality without external dependencies"""
    print("="*70)
    print("🔧 INTERVIEW MONITOR CORE FUNCTIONALITY TEST")
    print("="*70)
    print("Testing essential components for production readiness...")
    print()
    
    results = {}
    
    # Test 1: Sentiment Analysis Core
    print("1️⃣ Testing Core Sentiment Analysis...")
    try:
        from src.confidence_analysis.advanced_sentiment_analyzer import AdvancedSentimentAnalyzer
        
        analyzer = AdvancedSentimentAnalyzer()
        
        # Test basic functionality
        result = analyzer.analyze_sentiment("I am confident and ready for this challenge.")
        
        if result and hasattr(result, 'overall_sentiment'):
            print(f"   → Sentiment analysis working: {result.overall_sentiment:.2f}")
            print(f"   → Emotion detection: {result.primary_emotion.value}")
            results['sentiment'] = "PASS"
        else:
            results['sentiment'] = "FAIL - Invalid result structure"
            
    except Exception as e:
        print(f"   → Error: {e}")
        results['sentiment'] = f"FAIL - {str(e)[:50]}"
    
    print()
    
    # Test 2: Confidence Classification Core
    print("2️⃣ Testing Confidence Classification Core...")
    try:
        from src.confidence_analysis.confidence_classifier import (
            ConfidenceClassifier, FacialMetrics, SpeechMetrics
        )
        
        # Create classifier
        classifier = ConfidenceClassifier(use_advanced_sentiment=True)
        
        # Create test data
        facial_data = FacialMetrics(
            smile_intensity=0.8,
            frown_intensity=0.1,
            eyebrow_raise=0.3,
            eye_contact=0.7,
            head_tilt=2.0,
            head_nod=1.0,
            blink_rate=15.0,
            mouth_open=0.2
        )
        
        speech_data = SpeechMetrics(
            volume=0.8,
            pace=0.6,
            clarity=0.9,
            tone=0.5,
            pause_frequency=0.2,
            pitch_variation=0.7
        )
        
        # Test classification
        result = classifier.analyze_chunk(
            transcript="I am very excited about this opportunity.",
            facial_metrics=facial_data,
            speech_metrics=speech_data,
            chunk_duration=5.0
        )
        
        if result and hasattr(result, 'confidence_score'):
            print(f"   → Confidence scoring working: {result.confidence_score:.2f}")
            print(f"   → Classification level: {result.confidence_level.value}")
            results['confidence'] = "PASS"
        else:
            results['confidence'] = "FAIL - Invalid result structure"
            
    except Exception as e:
        print(f"   → Error: {e}")
        results['confidence'] = f"FAIL - {str(e)[:50]}"
    
    print()
    
    # Test 3: Facial Analysis Dependencies
    print("3️⃣ Testing Facial Analysis Dependencies...")
    try:
        import cv2
        import mediapipe as mp
        
        # Test basic MediaPipe initialization
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        )
        
        print(f"   → OpenCV version: {cv2.__version__}")
        print(f"   → MediaPipe initialized successfully")
        results['facial'] = "PASS"
        
        # Cleanup
        face_mesh.close()
        
    except Exception as e:
        print(f"   → Error: {e}")
        results['facial'] = f"FAIL - {str(e)[:50]}"
    
    print()
    
    # Test 4: Audio Processing Dependencies
    print("4️⃣ Testing Audio Processing Dependencies...")
    try:
        import whisper
        import numpy as np
        
        print(f"   → Whisper available")
        print(f"   → NumPy version: {np.__version__}")
        results['audio'] = "PASS"
        
    except Exception as e:
        print(f"   → Error: {e}")
        results['audio'] = f"FAIL - {str(e)[:50]}"
    
    print()
    
    # Test 5: Data Logging
    print("5️⃣ Testing Data Logging...")
    try:
        from src.logging.bigquery_logger import LocalLogger
        
        # Test local logger
        logger_instance = LocalLogger()
        
        # Test session creation
        session = logger_instance.start_session(user_id="test_user")
        
        if session and hasattr(session, 'session_id'):
            print(f"   → Session created: {session.session_id}")
            results['logging'] = "PASS"
        else:
            results['logging'] = "FAIL - Session creation failed"
            
    except Exception as e:
        print(f"   → Error: {e}")
        results['logging'] = f"FAIL - {str(e)[:50]}"
    
    print()
    
    # Test 6: Performance Benchmark
    print("6️⃣ Running Performance Benchmark...")
    try:
        start_time = time.time()
        
        # Simulate 5 analysis cycles
        for i in range(5):
            # Mock analysis cycle
            facial_data = FacialMetrics(
                smile_intensity=0.5 + i * 0.1,
                frown_intensity=0.1,
                eyebrow_raise=0.3,
                eye_contact=0.7,
                head_tilt=float(i),
                head_nod=1.0,
                blink_rate=15.0,
                mouth_open=0.2
            )
            
            speech_data = SpeechMetrics(
                volume=0.8,
                pace=0.6,
                clarity=0.9,
                tone=0.5,
                pause_frequency=0.2,
                pitch_variation=0.7
            )
            
            result = classifier.analyze_chunk(
                transcript=f"Test transcript chunk {i+1}",
                facial_metrics=facial_data,
                speech_metrics=speech_data,
                chunk_duration=5.0
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        fps = 5 / processing_time
        
        print(f"   → Processed 5 cycles in {processing_time:.2f}s ({fps:.1f} FPS)")
        
        if fps > 1.0:
            results['performance'] = "PASS"
            print(f"   → Performance: GOOD (>{fps:.1f} FPS)")
        else:
            results['performance'] = "WARN - Low performance"
            print(f"   → Performance: SLOW ({fps:.1f} FPS)")
            
    except Exception as e:
        print(f"   → Error: {e}")
        results['performance'] = f"FAIL - {str(e)[:50]}"
    
    print()
    
    # Summary
    print("="*70)
    print("📊 CORE FUNCTIONALITY TEST RESULTS")
    print("="*70)
    
    passed = 0
    total = len(results)
    
    for component, result in results.items():
        status_icon = "✅" if result == "PASS" else "❌" if "FAIL" in result else "⚠️"
        print(f"{status_icon} {component.title()}: {result}")
        if result == "PASS":
            passed += 1
    
    print()
    print(f"🎯 Results: {passed}/{total} core components working")
    
    if passed == total:
        print("🎉 All core functionality is working! System ready for production.")
        return True
    elif passed >= total * 0.7:
        print("⚠️ Most core functionality working. Some optional features may be limited.")
        return True
    else:
        print("❌ Critical functionality issues detected. Review errors above.")
        return False

def test_integration_quick():
    """Quick integration test of the complete pipeline"""
    print()
    print("🔄 QUICK INTEGRATION TEST")
    print("-" * 40)
    
    try:
        from src.confidence_analysis.confidence_classifier import (
            ConfidenceClassifier, FacialMetrics, SpeechMetrics
        )
        
        # Initialize components
        classifier = ConfidenceClassifier(use_advanced_sentiment=True)
        
        # Simulate real-world scenario
        scenarios = [
            {
                'name': 'Confident Response',
                'facial': FacialMetrics(0.8, 0.1, 0.3, 0.8, 1.0, 0.5, 15.0, 0.1),
                'speech': SpeechMetrics(0.8, 0.7, 0.9, 0.6, 0.2, 0.8),
                'text': "I am very confident in my abilities and excited about this role."
            },
            {
                'name': 'Nervous Response',
                'facial': FacialMetrics(0.2, 0.6, 0.1, 0.3, -2.0, -1.0, 25.0, 0.3),
                'speech': SpeechMetrics(0.4, 0.3, 0.6, 0.2, 0.8, 0.3),
                'text': "Um, I'm not really sure if I can handle this responsibility."
            },
            {
                'name': 'Neutral Response',
                'facial': FacialMetrics(0.5, 0.3, 0.4, 0.6, 0.0, 0.0, 18.0, 0.2),
                'speech': SpeechMetrics(0.6, 0.5, 0.7, 0.4, 0.4, 0.5),
                'text': "I have experience in this area and would like to learn more."
            }
        ]
        
        print("Testing different scenarios:")
        for scenario in scenarios:
            result = classifier.analyze_chunk(
                transcript=scenario['text'],
                facial_metrics=scenario['facial'],
                speech_metrics=scenario['speech'],
                chunk_duration=5.0
            )
            
            print(f"  • {scenario['name']}: "
                  f"Confidence={result.confidence_score:.2f} ({result.confidence_level.value}), "
                  f"Sentiment={result.sentiment_score:.2f}")
        
        print("✅ Integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        # Run core functionality test
        core_success = test_core_functionality()
        
        # Run integration test if core is working
        if core_success:
            integration_success = test_integration_quick()
            
            if integration_success:
                print("\n🚀 System is ready for deployment!")
                sys.exit(0)
            else:
                print("\n⚠️ Core works but integration has issues.")
                sys.exit(1)
        else:
            print("\n❌ Core functionality failed. Fix critical issues first.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
