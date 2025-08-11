"""
Subtitle Renderer Module
Renders live transcription as subtitles on video feed
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SubtitleStyle:
    """Subtitle styling configuration"""
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.8
    font_thickness: int = 2
    font_color: Tuple[int, int, int] = (255, 255, 255)  # White
    background_color: Tuple[int, int, int] = (0, 0, 0)   # Black
    background_opacity: float = 0.7
    padding: int = 10
    max_width: int = 600
    line_height: int = 35
    fade_duration: float = 3.0  # seconds

class SubtitleRenderer:
    """Renders live transcription as subtitles on video frames"""
    
    def __init__(self, style: Optional[SubtitleStyle] = None):
        """
        Initialize subtitle renderer
        
        Args:
            style: Subtitle styling configuration
        """
        self.style = style or SubtitleStyle()
        
        # Subtitle state
        self.current_subtitle = ""
        self.subtitle_history = []
        self.max_history = 5
        
        # Timing for fade effects
        self.subtitle_start_time = 0.0
        self.fade_timer = 0.0
        
        logger.info("Subtitle Renderer initialized")
    
    def update_subtitle(self, text: str, is_final: bool = False):
        """
        Update current subtitle text
        
        Args:
            text: New subtitle text
            is_final: Whether this is a final transcription
        """
        if not text.strip():
            return
        
        clean_text = text.strip()
        
        # Add to history if final
        if is_final:
            subtitle_entry = {
                'text': clean_text,
                'timestamp': self.fade_timer,
                'is_final': True
            }
            self.subtitle_history.append(subtitle_entry)
            
            # Keep history size manageable
            if len(self.subtitle_history) > self.max_history:
                self.subtitle_history.pop(0)
        
        # Update current subtitle
        self.current_subtitle = clean_text
        self.subtitle_start_time = self.fade_timer
        
        logger.debug(f"Subtitle updated: {clean_text}")
    
    def render_subtitle(self, frame: np.ndarray, current_time: float) -> np.ndarray:
        """
        Render subtitle on video frame
        
        Args:
            frame: Input video frame
            current_time: Current time for fade effects
            
        Returns:
            Frame with subtitle rendered
        """
        self.fade_timer = current_time
        
        if not self.current_subtitle:
            return frame
        
        # Check if subtitle should fade out
        if current_time - self.subtitle_start_time > self.style.fade_duration:
            self.current_subtitle = ""
            return frame
        
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Render subtitle
        self._draw_subtitle(output_frame, self.current_subtitle)
        
        return output_frame
    
    def render_subtitle_history(self, frame: np.ndarray, current_time: float) -> np.ndarray:
        """
        Render subtitle history on video frame
        
        Args:
            frame: Input video frame
            current_time: Current time for fade effects
            
        Returns:
            Frame with subtitle history rendered
        """
        if not self.subtitle_history:
            return frame
        
        # Create a copy of the frame
        output_frame = frame.copy()
        
        # Render recent subtitles
        recent_subtitles = self.subtitle_history[-3:]  # Show last 3
        
        for i, subtitle in enumerate(recent_subtitles):
            # Calculate fade based on age
            age = current_time - subtitle['timestamp']
            if age > self.style.fade_duration:
                continue
            
            # Calculate opacity
            opacity = 1.0 - (age / self.style.fade_duration)
            opacity = max(0.0, min(1.0, opacity))
            
            # Render with opacity
            self._draw_subtitle_with_opacity(output_frame, subtitle['text'], opacity, i)
        
        return output_frame
    
    def _draw_subtitle(self, frame: np.ndarray, text: str):
        """Draw subtitle text on frame"""
        if not text:
            return
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Split text into lines
        lines = self._wrap_text(text, width)
        
        # Calculate subtitle position (bottom center)
        total_height = len(lines) * self.style.line_height
        start_y = height - total_height - self.style.padding - 50  # 50px from bottom
        
        # Draw background rectangle
        self._draw_subtitle_background(frame, lines, start_y, width)
        
        # Draw text
        self._draw_subtitle_text(frame, lines, start_y)
    
    def _draw_subtitle_with_opacity(self, frame: np.ndarray, text: str, opacity: float, line_index: int):
        """Draw subtitle with specific opacity"""
        if not text:
            return
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Split text into lines
        lines = self._wrap_text(text, width)
        
        # Calculate position (above current subtitle)
        total_height = len(lines) * self.style.line_height
        start_y = height - total_height - self.style.padding - 50 - (line_index + 1) * (total_height + 10)
        
        # Apply opacity to colors
        font_color = tuple(int(c * opacity) for c in self.style.font_color)
        bg_color = tuple(int(c * opacity) for c in self.style.background_color)
        
        # Draw background rectangle with opacity
        self._draw_subtitle_background(frame, lines, start_y, width, bg_color)
        
        # Draw text with opacity
        self._draw_subtitle_text(frame, lines, start_y, font_color)
    
    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        """Wrap text to fit within max width"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            
            # Estimate text width (approximate)
            text_width = len(test_line) * 15  # Rough estimate
            
            if text_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines if lines else [text]
    
    def _draw_subtitle_background(self, frame: np.ndarray, lines: List[str], start_y: int, width: int, 
                                 bg_color: Optional[Tuple[int, int, int]] = None):
        """Draw subtitle background rectangle"""
        if not lines:
            return
        
        bg_color = bg_color or self.style.background_color
        
        # Calculate background dimensions
        total_height = len(lines) * self.style.line_height
        bg_width = min(width - 2 * self.style.padding, self.style.max_width)
        bg_x = (width - bg_width) // 2
        
        # Draw background rectangle
        cv2.rectangle(
            frame,
            (bg_x, start_y - self.style.padding),
            (bg_x + bg_width, start_y + total_height + self.style.padding),
            bg_color,
            -1  # Filled rectangle
        )
    
    def _draw_subtitle_text(self, frame: np.ndarray, lines: List[str], start_y: int, 
                           font_color: Optional[Tuple[int, int, int]] = None):
        """Draw subtitle text"""
        if not lines:
            return
        
        font_color = font_color or self.style.font_color
        
        for i, line in enumerate(lines):
            y = start_y + i * self.style.line_height
            
            # Calculate text position (center horizontally)
            text_size = cv2.getTextSize(line, self.style.font_face, self.style.font_scale, self.style.font_thickness)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            
            # Draw text
            cv2.putText(
                frame,
                line,
                (text_x, y),
                self.style.font_face,
                self.style.font_scale,
                font_color,
                self.style.font_thickness
            )
    
    def clear_subtitles(self):
        """Clear all subtitles"""
        self.current_subtitle = ""
        self.subtitle_history.clear()
        logger.info("Subtitles cleared")
    
    def set_style(self, style: SubtitleStyle):
        """Update subtitle style"""
        self.style = style
        logger.info("Subtitle style updated")
    
    def get_current_subtitle(self) -> str:
        """Get current subtitle text"""
        return self.current_subtitle
    
    def get_subtitle_history(self) -> List[Dict]:
        """Get subtitle history"""
        return self.subtitle_history.copy()

def test_subtitle_renderer():
    """Test function for subtitle renderer"""
    print("Testing subtitle renderer...")
    
    try:
        # Create custom style
        style = SubtitleStyle(
            font_scale=1.0,
            font_color=(255, 255, 0),  # Yellow
            background_color=(0, 0, 128),  # Dark blue
            background_opacity=0.8
        )
        
        renderer = SubtitleRenderer(style)
        print("✓ Subtitle renderer initialized")
        
        # Test subtitle update
        renderer.update_subtitle("Hello, this is a test subtitle!", True)
        print("✓ Subtitle updated")
        
        # Test text wrapping
        long_text = "This is a very long subtitle text that should be wrapped to multiple lines to fit within the specified width constraints"
        lines = renderer._wrap_text(long_text, 400)
        print(f"✓ Text wrapping: {len(lines)} lines")
        
        print("\nSubtitle renderer is ready for use!")
        
    except Exception as e:
        print(f"❌ Subtitle renderer test failed: {e}")

if __name__ == "__main__":
    test_subtitle_renderer()
