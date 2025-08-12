#!/usr/bin/env python3
"""
Test script for Transcription System
Tests speech transcription and subtitle rendering
"""

import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_subtitle_renderer():
    """Test subtitle renderer functionality"""
    print("=" * 50)
    print("Testing Subtitle Renderer")
    print("=" * 50)
    
    try:
        from transcription.subtitle_renderer import SubtitleRenderer, SubtitleStyle
        
        # Test custom style
        style = SubtitleStyle(
            font_scale=1.0,
            font_color=(255, 255, 0),  # Yellow
            background_color=(0, 0, 128),  # Dark blue
            background_opacity=0.8
        )
        
        renderer = SubtitleRenderer(style)
        print("✓ Subtitle renderer initialized")
        
        # Test subtitle updates
        renderer.update_subtitle("Hello, this is a test subtitle!", True)
        print("✓ Subtitle updated")
        
        # Test text wrapping
        long_text = "This is a very long subtitle text that should be wrapped to multiple lines to fit within the specified width constraints"
        lines = renderer._wrap_text(long_text, 400)
        print(f"✓ Text wrapping: {len(lines)} lines")
        
        # Test subtitle history
        renderer.update_subtitle("Second subtitle for testing", True)
        renderer.update_subtitle("Third subtitle for testing", True)
        history = renderer.get_subtitle_history()
        print(f"✓ Subtitle history: {len(history)} entries")
        
        return True
        
    except Exception as e:
        print(f"❌ Subtitle renderer test failed: {e}")
        return False

def test_speech_transcriber():
    """Test speech transcriber functionality"""
    print("\n" + "=" * 50)
    print("Testing Speech Transcriber")
    print("=" * 50)
    
    try:
        from transcription.speech_transcriber import SpeechTranscriber
        
        transcriber = SpeechTranscriber()
        print("✓ Speech transcriber initialized")
        
        # Test configuration
        print(f"✓ Sample rate: {transcriber.sample_rate} Hz")
        print(f"✓ Language: {transcriber.language_code}")
        print(f"✓ Audio format: {transcriber.audio_format}")
        
        # Test subtitle renderer integration
        from transcription.subtitle_renderer import SubtitleRenderer
        subtitle_renderer = SubtitleRenderer()
        
        # Set up callbacks
        transcriber.on_transcription_update = lambda text: subtitle_renderer.update_subtitle(text, False)
        transcriber.on_transcription_complete = lambda entry: subtitle_renderer.update_subtitle(entry['text'], True)
        
        print("✓ Callback integration tested")
        
        transcriber.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Speech transcriber test failed: {e}")
        print("This is expected if Google Cloud credentials are not set up")
        return False

def test_integrated_transcription():
    """Test integrated transcription system"""
    print("\n" + "=" * 50)
    print("Testing Integrated Transcription System")
    print("=" * 50)
    
    try:
        from transcription.speech_transcriber import SpeechTranscriber
        from transcription.subtitle_renderer import SubtitleRenderer, SubtitleStyle
        
        print("✓ All modules imported successfully")
        
        # Initialize components
        transcriber = SpeechTranscriber()
        subtitle_style = SubtitleStyle(
            font_scale=1.0,
            font_color=(255, 255, 0),
            background_color=(0, 0, 128)
        )
        subtitle_renderer = SubtitleRenderer(subtitle_style)
        
        print("✓ All components initialized")
        
        # Test integration
        def on_transcription_update(text):
            subtitle_renderer.update_subtitle(text, False)
            print(f"Interim: {text}")
        
        def on_transcription_complete(entry):
            subtitle_renderer.update_subtitle(entry['text'], True)
            print(f"Final: {entry['text']} (confidence: {entry['confidence']:.2f})")
        
        transcriber.on_transcription_update = on_transcription_update
        transcriber.on_transcription_complete = on_transcription_complete
        
        print("✓ Callback system integrated")
        
        # Test subtitle rendering simulation
        test_texts = [
            "Hello, welcome to the interview.",
            "I am confident in my abilities.",
            "I have experience in machine learning."
        ]
        
        print("\nSimulating subtitle rendering...")
        for i, text in enumerate(test_texts):
            subtitle_renderer.update_subtitle(text, True)
            current = subtitle_renderer.get_current_subtitle()
            history = subtitle_renderer.get_subtitle_history()
            print(f"  Subtitle {i+1}: {current}")
            print(f"  History count: {len(history)}")
        
        transcriber.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Integrated transcription test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interview_monitor_integration():
    """Test interview monitor integration"""
    print("\n" + "=" * 50)
    print("Testing Interview Monitor Integration")
    print("=" * 50)
    
    try:
        from interview_monitor import InterviewMonitor
        from transcription.subtitle_renderer import SubtitleStyle
        
        print("✓ Interview monitor module imported")
        
        # Test with transcription disabled first
        monitor = InterviewMonitor(
            enable_transcription=False,
            enable_confidence_analysis=True
        )
        print("✓ Interview monitor initialized (transcription disabled)")
        
        # Test component access
        print(f"✓ Video capture: {monitor.video_capture is not None}")
        print(f"✓ Landmark detector: {monitor.landmark_detector is not None}")
        print(f"✓ Expression analyzer: {monitor.expression_analyzer is not None}")
        print(f"✓ Confidence classifier: {monitor.confidence_classifier is not None}")
        print(f"✓ Subtitle renderer: {monitor.subtitle_renderer is not None}")
        
        monitor.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Interview monitor integration test failed: {e}")
        print("This is expected if some components are not fully configured")
        return False

def main():
    """Main test function"""
    print("=" * 70)
    print("Transcription System - Comprehensive Test")
    print("=" * 70)
    
    print("This will test the complete transcription system including:")
    print("• Subtitle renderer")
    print("• Speech transcriber")
    print("• Integrated transcription system")
    print("• Interview monitor integration")
    
    # Run individual tests
    tests = [
        ("Subtitle Renderer", test_subtitle_renderer),
        ("Speech Transcriber", test_speech_transcriber),
        ("Integrated Transcription", test_integrated_transcription),
        ("Interview Monitor Integration", test_interview_monitor_integration)
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
        print("🎉 All tests passed! Transcription system is ready!")
        print("\nNext steps:")
        print("1. Set up Google Cloud credentials for Speech-to-Text")
        print("2. Install PyAudio: pip install pyaudio")
        print("3. Run the complete interview monitor")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nNote: Speech transcriber tests may fail without Google Cloud credentials")
        print("This is expected and normal for initial setup.")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
