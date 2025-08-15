#!/usr/bin/env python3
"""
Simple Facial Analysis Test
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_enhanced_face_analyzer():
    """Test the enhanced face analyzer with a simple image"""
    print("Testing Enhanced Face Analyzer...")
    
    try:
        from facial_detection.enhanced_face_analyzer import EnhancedFaceAnalyzer
        
        # Create analyzer
        analyzer = EnhancedFaceAnalyzer()
        print("✅ Enhanced face analyzer created")
        
        # Create a simple test image (smaller size)
        test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        print("✅ Test image created (240x320)")
        
        # Try to analyze
        print("Attempting to analyze image...")
        result = analyzer.analyze_frame(test_image)
        
        if result:
            print("✅ Analysis successful!")
            print(f"Result type: {type(result)}")
            print(f"Result: {result}")
        else:
            print("⚠️ Analysis returned None (expected for random image)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_face_analyzer()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
