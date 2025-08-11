#!/usr/bin/env python3
"""
Test script for facial landmark detection and expression analysis
"""

import sys
import os
import cv2
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_facial_detection():
    """Test facial landmark detection and expression analysis"""
    print("=" * 60)
    print("Facial Detection Test - Interview Monitor")
    print("=" * 60)
    
    try:
        from facial_detection.video_capture import VideoCapture
        from facial_detection.landmark_detector import FacialLandmarkDetector
        from facial_detection.expression_analyzer import ExpressionAnalyzer
        
        print("✓ All modules imported successfully")
        
        # Initialize components
        print("\nInitializing components...")
        detector = FacialLandmarkDetector()
        analyzer = ExpressionAnalyzer()
        
        print("✓ Facial landmark detector initialized")
        print("✓ Expression analyzer initialized")
        
        # Test with video capture
        print("\nStarting video capture test...")
        print("Controls:")
        print("- Press 'c' to calibrate expression analyzer")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        
        with VideoCapture() as cap:
            if not cap.is_initialized:
                print("❌ Camera not available for testing")
                return
            
            print("✓ Camera initialized successfully")
            
            # Calibration flag
            calibrated = False
            
            while True:
                frame = cap.read_frame()
                if frame is None:
                    continue
                
                # Create a copy for display
                display_frame = frame.copy()
                
                # Detect landmarks
                landmarks = detector.detect_landmarks(frame)
                
                if landmarks:
                    # Draw landmarks
                    display_frame = detector.draw_landmarks(display_frame, landmarks)
                    
                    # Analyze expressions if calibrated
                    if calibrated:
                        expressions = analyzer.analyze_expressions(landmarks)
                        expression_summary = analyzer.get_expression_summary(expressions)
                        
                        # Display expression summary
                        cv2.putText(display_frame, f"Expressions: {expression_summary}", 
                                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, (255, 255, 255), 2)
                        
                        # Display individual expression scores
                        y_offset = 30
                        for expr_name, expr_data in expressions.items():
                            if expr_data.get('detected'):
                                confidence = expr_data.get('confidence', 0)
                                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                                cv2.putText(display_frame, f"{expr_name}: {confidence:.2f}", 
                                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                           0.5, color, 2)
                                y_offset += 25
                    else:
                        # Show calibration message
                        cv2.putText(display_frame, "Press 'c' to calibrate", 
                                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 255, 255), 2)
                    
                    # Display landmark count
                    cv2.putText(display_frame, f"Landmarks: {len(landmarks['all_landmarks'])}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                else:
                    # No face detected
                    cv2.putText(display_frame, "No face detected", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show frame
                cv2.imshow('Facial Detection Test', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('c') and landmarks:
                    print("\nCalibrating expression analyzer...")
                    if analyzer.calibrate(landmarks):
                        calibrated = True
                        print("✓ Calibration successful!")
                    else:
                        print("❌ Calibration failed")
                elif key == ord('s'):
                    filename = f"test_frame_{len(os.listdir('.'))}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"✓ Frame saved as {filename}")
        
        cv2.destroyAllWindows()
        
        # Cleanup
        detector.release()
        print("\n✅ Test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies:")
        print("pip install opencv-python mediapipe numpy")
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

def test_offline_components():
    """Test components without video capture"""
    print("\n" + "=" * 40)
    print("Offline Component Tests")
    print("=" * 40)
    
    try:
        from facial_detection.expression_analyzer import ExpressionAnalyzer
        
        # Test expression analyzer
        analyzer = ExpressionAnalyzer()
        print("✓ Expression analyzer created")
        
        # Test thresholds
        print(f"✓ Smile thresholds: {analyzer.thresholds['smile']}")
        print(f"✓ Frown thresholds: {analyzer.thresholds['frown']}")
        print(f"✓ Eye blink threshold: {analyzer.thresholds['eye_blink']}")
        
        # Test expression summary
        test_expressions = {
            'smile': {'detected': True, 'confidence': 0.8},
            'raised_eyebrows': {'detected': True, 'confidence': 0.6}
        }
        
        summary = analyzer.get_expression_summary(test_expressions)
        print(f"✓ Expression summary: {summary}")
        
        print("\n✅ Offline tests passed!")
        
    except Exception as e:
        print(f"❌ Offline test error: {e}")

def main():
    """Main test function"""
    print("Interview Monitor - Facial Detection Test")
    print("This will test the facial landmark detection and expression analysis")
    
    # Test offline components first
    test_offline_components()
    
    # Ask user if they want to test with camera
    print("\n" + "=" * 50)
    response = input("Do you want to test with camera? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        test_facial_detection()
    else:
        print("Skipping camera test. You can run it later with:")
        print("python test_facial_detection.py")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
