#!/usr/bin/env python3
"""
Test script for Confidence Analysis System
Tests facial expressions, head pose, speech analysis, and sentiment analysis
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_confidence_classifier():
    """Test confidence classifier with sample data"""
    print("=" * 50)
    print("Testing Confidence Classifier")
    print("=" * 50)
    
    try:
        from confidence_analysis.confidence_classifier import ConfidenceClassifier, ConfidenceLevel
        
        classifier = ConfidenceClassifier()
        print("✓ Confidence classifier initialized")
        
        # Test with sample data
        facial_data = {
            'smile': {'detected': True, 'confidence': 0.8},
            'raised_eyebrows': {'detected': True, 'confidence': 0.6},
            'eye_blink': {'detected': False, 'confidence': 0.0}
        }
        
        head_pose = {
            'pitch': 5.0,  # Small forward tilt
            'yaw': 2.0,    # Slight turn
            'roll': 1.0    # Minimal tilt
        }
        
        speech_data = {
            'clear_articulation': 0.8,
            'steady_pace': 0.7,
            'appropriate_volume': 0.9,
            'minimal_fillers': 0.6,
            'consistent_tone': 0.8
        }
        
        sentiment_score = 0.3  # Slightly positive
        
        # Classify confidence
        metrics = classifier.classify_confidence(facial_data, head_pose, speech_data, sentiment_score)
        
        print(f"✓ Overall confidence: {metrics.overall_confidence:.1%}")
        print(f"✓ Confidence level: {metrics.confidence_level.value}")
        print(f"✓ Facial confidence: {metrics.facial_confidence:.1%}")
        print(f"✓ Posture confidence: {metrics.posture_confidence:.1%}")
        print(f"✓ Speech confidence: {metrics.speech_confidence:.1%}")
        
        # Get recommendations
        recommendations = classifier.get_recommendations(metrics)
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
        
        return True
        
    except Exception as e:
        print(f"❌ Confidence classifier test failed: {e}")
        return False

def test_sentiment_analyzer():
    """Test sentiment analyzer with sample texts"""
    print("\n" + "=" * 50)
    print("Testing Sentiment Analyzer")
    print("=" * 50)
    
    try:
        from confidence_analysis.sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        print("✓ Sentiment analyzer initialized")
        
        # Test with sample texts
        test_texts = [
            "I am very confident in my abilities and I believe I can excel in this role.",
            "I'm not sure about this, maybe I could try but it might be difficult.",
            "The project was challenging but I successfully completed it with excellent results."
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text[:50]}...")
            analysis = analyzer.analyze_sentiment(text)
            summary = analyzer.get_sentiment_summary(analysis)
            print(summary)
        
        return True
        
    except Exception as e:
        print(f"❌ Sentiment analyzer test failed: {e}")
        return False

def test_speech_analyzer():
    """Test speech analyzer with sample audio data"""
    print("\n" + "=" * 50)
    print("Testing Speech Analyzer")
    print("=" * 50)
    
    try:
        from confidence_analysis.speech_analyzer import SpeechAnalyzer
        
        analyzer = SpeechAnalyzer()
        print("✓ Speech analyzer initialized")
        
        # Create dummy audio data for testing
        sample_rate = 16000
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate test audio (sine wave + noise)
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        
        # Test speech analysis
        analysis = analyzer.analyze_speech_characteristics(test_audio, "This is a test transcription for speech analysis")
        
        print(f"✓ Overall speech confidence: {analysis['overall_confidence']:.2f}")
        print(f"✓ Volume: {analysis['volume']['level']} (score: {analysis['volume']['score']:.2f})")
        print(f"✓ Pace: {analysis['pace']['level']} (score: {analysis['pace']['score']:.2f})")
        print(f"✓ Clarity: {analysis['clarity']['level']} (score: {analysis['clarity']['score']:.2f})")
        print(f"✓ Tone score: {analysis['tone']['score']:.2f}")
        
        print("\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"  • {rec}")
        
        return True
        
    except Exception as e:
        print(f"❌ Speech analyzer test failed: {e}")
        return False

def test_integrated_system():
    """Test the integrated confidence analysis system"""
    print("\n" + "=" * 50)
    print("Testing Integrated Confidence Analysis System")
    print("=" * 50)
    
    try:
        from confidence_analysis.confidence_classifier import ConfidenceClassifier
        from confidence_analysis.sentiment_analyzer import SentimentAnalyzer
        from confidence_analysis.speech_analyzer import SpeechAnalyzer
        
        print("✓ All modules imported successfully")
        
        # Initialize all components
        confidence_classifier = ConfidenceClassifier()
        sentiment_analyzer = SentimentAnalyzer()
        speech_analyzer = SpeechAnalyzer()
        
        print("✓ All components initialized")
        
        # Simulate interview data
        print("\nSimulating interview analysis...")
        
        # 1. Facial expressions (from previous step)
        facial_data = {
            'smile': {'detected': True, 'confidence': 0.7},
            'raised_eyebrows': {'detected': False, 'confidence': 0.0},
            'eye_blink': {'detected': True, 'confidence': 0.3}
        }
        
        # 2. Head pose
        head_pose = {
            'pitch': 3.0,  # Good posture
            'yaw': 1.0,    # Minimal turn
            'roll': 0.5    # Very slight tilt
        }
        
        # 3. Speech characteristics
        speech_data = {
            'clear_articulation': 0.8,
            'steady_pace': 0.9,
            'appropriate_volume': 0.8,
            'minimal_fillers': 0.7,
            'consistent_tone': 0.8
        }
        
        # 4. Sentiment analysis
        interview_text = "I am confident in my technical skills and I believe I can contribute significantly to your team. I have successfully completed several challenging projects and I'm excited about this opportunity."
        sentiment_analysis = sentiment_analyzer.analyze_sentiment(interview_text)
        sentiment_score = sentiment_analysis['overall_sentiment']
        
        print(f"✓ Sentiment score: {sentiment_score:.2f}")
        
        # 5. Overall confidence classification
        confidence_metrics = confidence_classifier.classify_confidence(
            facial_data, head_pose, speech_data, sentiment_score
        )
        
        print("\n" + "=" * 60)
        print("INTERVIEW CONFIDENCE ASSESSMENT")
        print("=" * 60)
        
        summary = confidence_classifier.get_confidence_summary(confidence_metrics)
        print(summary)
        
        print("\nDetailed Breakdown:")
        for factor, data in confidence_metrics.detailed_breakdown.items():
            print(f"  {factor.replace('_', ' ').title()}: {data['score']:.1%} (weight: {data['weight']:.1%})")
        
        print("\nRecommendations:")
        recommendations = confidence_classifier.get_recommendations(confidence_metrics)
        for rec in recommendations:
            print(f"  • {rec}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 70)
    print("Confidence Analysis System - Comprehensive Test")
    print("=" * 70)
    
    print("This will test the complete confidence analysis system including:")
    print("• Confidence classification")
    print("• Sentiment analysis")
    print("• Speech analysis")
    print("• Integrated system")
    
    # Run individual tests
    tests = [
        ("Confidence Classifier", test_confidence_classifier),
        ("Sentiment Analyzer", test_sentiment_analyzer),
        ("Speech Analyzer", test_speech_analyzer),
        ("Integrated System", test_integrated_system)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} test PASSED")
        else:
            print(f"❌ {test_name} test FAILED")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Confidence analysis system is ready!")
        print("\nNext steps:")
        print("1. Integrate with facial detection system")
        print("2. Add Google Speech-to-Text for live transcription")
        print("3. Create web interface")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
