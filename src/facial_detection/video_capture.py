"""
Video Capture Module for Interview Monitoring
Handles webcam access and basic video processing using OpenCV
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoCapture:
    """Handles video capture from webcam using OpenCV"""
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        Initialize video capture
        
        Args:
            camera_index: Index of the camera (usually 0 for default webcam)
            width: Frame width
            height: Frame height
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize the video capture
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera at index {self.camera_index}")
                return False
            
            # Set frame dimensions
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Set frame rate
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_initialized = True
            logger.info(f"Video capture initialized successfully. Resolution: {self.width}x{self.height}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing video capture: {str(e)}")
            return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a frame from the camera
        
        Returns:
            np.ndarray: Frame as numpy array, or None if failed
        """
        if not self.is_initialized or self.cap is None:
            logger.warning("Video capture not initialized")
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                logger.warning("Failed to read frame from camera")
                return None
                
        except Exception as e:
            logger.error(f"Error reading frame: {str(e)}")
            return None
    
    def get_frame_info(self) -> Tuple[int, int, float]:
        """
        Get current frame information
        
        Returns:
            Tuple[int, int, float]: Width, height, and FPS
        """
        if not self.is_initialized or self.cap is None:
            return 0, 0, 0
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        return width, height, fps
    
    def release(self):
        """Release the video capture resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.is_initialized = False
            logger.info("Video capture released")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()

def test_video_capture():
    """Test function for video capture"""
    print("Testing video capture...")
    
    with VideoCapture() as cap:
        if cap.is_initialized:
            print("Camera initialized successfully!")
            
            # Read a few frames
            for i in range(5):
                frame = cap.read_frame()
                if frame is not None:
                    print(f"Frame {i+1} shape: {frame.shape}")
                else:
                    print(f"Failed to read frame {i+1}")
            
            # Get frame info
            width, height, fps = cap.get_frame_info()
            print(f"Camera info: {width}x{height} @ {fps} FPS")
        else:
            print("Failed to initialize camera")

if __name__ == "__main__":
    test_video_capture()
