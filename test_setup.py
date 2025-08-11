#!/usr/bin/env python3
"""
Test script to verify basic setup and video capture functionality
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from facial_detection.video_capture import VideoCapture
        print("✓ VideoCapture class imported successfully")
    except ImportError as e:
        print(f"✗ VideoCapture import failed: {e}")
        return False
    
    return True

def test_video_capture():
    """Test video capture functionality"""
    print("\nTesting video capture...")
    
    try:
        from facial_detection.video_capture import VideoCapture
        
        with VideoCapture() as cap:
            if cap.is_initialized:
                print("✓ Camera initialized successfully")
                
                # Try to read one frame
                frame = cap.read_frame()
                if frame is not None:
                    print(f"✓ Frame captured successfully. Shape: {frame.shape}")
                    return True
                else:
                    print("✗ Failed to capture frame")
                    return False
            else:
                print("✗ Camera initialization failed")
                return False
                
    except Exception as e:
        print(f"✗ Video capture test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("Interview Monitor - Setup Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your installation.")
        return
    
    # Test video capture
    if test_video_capture():
        print("\n✅ All tests passed! Basic setup is working.")
        print("\nNext steps:")
        print("1. Install additional dependencies: pip install -r requirements.txt")
        print("2. Set up Google Cloud credentials for Speech-to-Text")
        print("3. Move to Step 2: Facial landmark detection")
    else:
        print("\n❌ Video capture test failed.")
        print("This might be due to:")
        print("- No webcam available")
        print("- Webcam permissions")
        print("- OpenCV installation issues")

if __name__ == "__main__":
    main()
