#!/usr/bin/env python3
"""
Interview Monitor System - Main Entry Point
A comprehensive system for monitoring interview candidates using AI and computer vision
"""

import sys
import os
import logging
import argparse
import signal
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('interview_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment and check dependencies"""
    logger.info("🔧 Setting up Interview Monitor environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ is required")
        return False
    
    # Check if src directory exists
    if not os.path.exists('src'):
        logger.error("'src' directory not found. Please run from project root.")
        return False
    
    # Check required directories
    required_dirs = ['src/facial_detection', 'src/confidence_analysis', 'src/transcription']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logger.error(f"Required directory '{dir_path}' not found")
            return False
    
    logger.info("✅ Environment setup completed")
    return True

def check_dependencies():
    """Check if all required dependencies are available"""
    logger.info("🔍 Checking system dependencies...")
    
    missing_deps = []
    
    # Core dependencies
    try:
        import cv2
        logger.info("✅ OpenCV available")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import numpy as np
        logger.info("✅ NumPy available")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import mediapipe as mp
        logger.info("✅ MediaPipe available")
    except ImportError:
        missing_deps.append("mediapipe")
    
    try:
        import torch
        logger.info(f"✅ PyTorch available (version: {torch.__version__})")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
        logger.info("✅ Transformers available")
    except ImportError:
        missing_deps.append("transformers")
    
    # Optional dependencies
    try:
        import PyQt5
        logger.info("✅ PyQt5 available (GUI will work)")
    except ImportError:
        logger.warning("⚠️ PyQt5 not available (GUI will use fallback)")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Install with: pip install " + " ".join(missing_deps))
        return False
    
    logger.info("✅ All core dependencies available")
    return True

def run_demo_mode():
    """Run the system in demo mode without camera/microphone"""
    logger.info("🎭 Starting Demo Mode...")
    
    try:
        from demo_interview_monitor import main as run_demo
        success = run_demo()
        return success
    except Exception as e:
        logger.error(f"Demo mode failed: {e}")
        return False

def run_interview_monitor():
    """Run the main interview monitor with camera and microphone"""
    logger.info("🎥 Starting Interview Monitor...")
    
    try:
        from interview_monitor import EnhancedInterviewMonitor
        monitor = EnhancedInterviewMonitor()
        monitor.run()
        return True
    except KeyboardInterrupt:
        logger.info("🛑 Interview Monitor stopped by user")
        return True
    except Exception as e:
        logger.error(f"Interview Monitor failed: {e}")
        return False

def run_hud_only():
    """Run only the HUD interface (for testing UI components)"""
    logger.info("🖥️ Starting HUD Interface...")
    
    try:
        from ui.interview_hud import InterviewHUD, QApplication
        import sys
        
        app = QApplication(sys.argv)
        hud = InterviewHUD()
        hud.show()
        
        logger.info("✅ HUD interface started. Close window to exit.")
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"HUD interface failed: {e}")
        return False

def run_tests():
    """Run system tests"""
    logger.info("🧪 Running System Tests...")
    
    try:
        from test_simple import test_core_components
        success = test_core_components()
        return success
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        return False

def show_help():
    """Show help information"""
    help_text = """
🎯 Interview Monitor System - Help
===================================

Usage: python main.py [OPTION]

Options:
  -d, --demo          Run in demo mode (no camera/microphone required)
  -m, --monitor       Run full interview monitor (requires camera/microphone)
  -h, --hud           Run only HUD interface
  -t, --test          Run system tests
  --help              Show this help message

Examples:
  python main.py --demo          # Run demo mode
  python main.py --monitor       # Run full system
  python main.py --hud           # Run HUD only
  python main.py --test          # Run tests

Features:
  • Real-time facial expression analysis
  • Live speech transcription with Whisper ASR
  • Advanced sentiment analysis (emotion, toxicity, politeness)
  • Confidence and relevance scoring
  • Eye contact and head pose tracking
  • Live metrics display with PyQt HUD
  • Data logging to BigQuery/local files
  • Docker deployment support

Requirements:
  • Camera for facial analysis
  • Microphone for speech transcription
  • Python 3.8+
  • See requirements.txt for Python packages
    """
    print(help_text)

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    logger.info(f"🛑 Received signal {signum}. Shutting down gracefully...")
    sys.exit(0)

def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Interview Monitor System",
        add_help=False
    )
    parser.add_argument('-d', '--demo', action='store_true', help='Run in demo mode')
    parser.add_argument('-m', '--monitor', action='store_true', help='Run full interview monitor')
    parser.add_argument('--hud', action='store_true', help='Run only HUD interface')
    parser.add_argument('-t', '--test', action='store_true', help='Run system tests')
    parser.add_argument('--help', action='store_true', help='Show help message')
    
    args = parser.parse_args()
    
    # Show help if requested
    if args.help:
        show_help()
        return
    
    # Print banner
    print("=" * 70)
    print("🎯 INTERVIEW MONITOR SYSTEM")
    print("=" * 70)
    print("AI-powered interview candidate analysis with computer vision")
    print("=" * 70)
    
    # Setup environment
    if not setup_environment():
        logger.error("Environment setup failed")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed")
        sys.exit(1)
    
    # Determine mode based on arguments
    if args.demo:
        logger.info("🎭 Starting in DEMO MODE")
        success = run_demo_mode()
    elif args.monitor:
        logger.info("🎥 Starting in MONITOR MODE")
        success = run_interview_monitor()
    elif args.hud:
        logger.info("🖥️ Starting in HUD MODE")
        success = run_hud_only()
    elif args.test:
        logger.info("🧪 Starting in TEST MODE")
        success = run_tests()
    else:
        # Interactive mode - ask user what to do
        print("\n🎯 What would you like to do?")
        print("1. Run Demo Mode (no camera/microphone)")
        print("2. Run Full Interview Monitor (requires camera/microphone)")
        print("3. Run HUD Interface Only")
        print("4. Run System Tests")
        print("5. Show Help")
        print("6. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-6): ").strip()
                
                if choice == '1':
                    success = run_demo_mode()
                    break
                elif choice == '2':
                    success = run_interview_monitor()
                    break
                elif choice == '3':
                    success = run_hud_only()
                    break
                elif choice == '4':
                    success = run_tests()
                    break
                elif choice == '5':
                    show_help()
                    return
                elif choice == '6':
                    logger.info("👋 Goodbye!")
                    return
                else:
                    print("Invalid choice. Please enter 1-6.")
            except KeyboardInterrupt:
                logger.info("👋 Goodbye!")
                return
    
    # Report results
    if success:
        logger.info("✅ Interview Monitor completed successfully")
    else:
        logger.error("Interview Monitor failed")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
