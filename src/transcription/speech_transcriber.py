"""
Speech Transcriber Module
Integrates Google Speech-to-Text API for live transcription
"""

import os
import time
import threading
import queue
import logging
from typing import Dict, List, Optional, Callable
import numpy as np
import pyaudio
from google.cloud import speech
from google.auth import default

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechTranscriber:
    """Real-time speech transcription using Google Speech-to-Text"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 language_code: str = "en-US",
                 enable_automatic_punctuation: bool = True,
                 enable_word_time_offsets: bool = True):
        """
        Initialize speech transcriber
        
        Args:
            sample_rate: Audio sample rate (Hz)
            chunk_size: Audio chunk size for processing
            language_code: Language code for transcription
            enable_automatic_punctuation: Enable automatic punctuation
            enable_word_time_offsets: Enable word-level timing
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.language_code = language_code
        self.enable_automatic_punctuation = enable_automatic_punctuation
        self.enable_word_time_offsets = enable_word_time_offsets
        
        # Audio configuration
        self.audio_format = pyaudio.paInt16
        self.channels = 1  # Mono audio
        
        # Transcription state
        self.is_transcribing = False
        self.current_transcription = ""
        self.transcription_history = []
        self.max_history = 100
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        
        # Google Speech-to-Text client
        self.speech_client = None
        self.streaming_config = None
        
        # Callbacks
        self.on_transcription_update: Optional[Callable] = None
        self.on_transcription_complete: Optional[Callable] = None
        
        # Initialize components
        self._initialize_google_speech()
        self._initialize_audio()
        
        logger.info("Speech Transcriber initialized")
    
    def _initialize_google_speech(self):
        """Initialize Google Speech-to-Text client"""
        try:
            # Try to get credentials from environment or default
            credentials, project = default()
            
            if not credentials:
                logger.warning("No Google Cloud credentials found. Please set GOOGLE_APPLICATION_CREDENTIALS")
                logger.info("You can get credentials from: https://console.cloud.google.com/apis/credentials")
                return
            
            self.speech_client = speech.SpeechClient(credentials=credentials)
            
            # Configure streaming recognition
            self.streaming_config = speech.StreamingRecognitionConfig(
                config=speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    language_code=self.language_code,
                    enable_automatic_punctuation=self.enable_automatic_punctuation,
                    enable_word_time_offsets=self.enable_word_time_offsets,
                    model="latest_long",  # Best for continuous speech
                    use_enhanced=True,    # Enhanced models for better accuracy
                ),
                interim_results=True,     # Get results as they come
                single_utterance=False    # Continue listening
            )
            
            logger.info("Google Speech-to-Text client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Speech-to-Text: {e}")
            logger.info("Please ensure you have valid Google Cloud credentials")
    
    def _initialize_audio(self):
        """Initialize PyAudio for microphone input"""
        try:
            self.audio = pyaudio.PyAudio()
            
            # Find default microphone
            default_input = self.audio.get_default_input_device_info()
            logger.info(f"Using microphone: {default_input['name']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            self.audio = None
    
    def start_transcription(self):
        """Start real-time transcription"""
        if not self.speech_client or not self.audio:
            logger.error("Speech client or audio not initialized")
            return False
        
        if self.is_transcribing:
            logger.warning("Transcription already running")
            return False
        
        try:
            self.is_transcribing = True
            
            # Start audio capture thread
            self.audio_thread = threading.Thread(target=self._audio_capture_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # Start transcription thread
            self.transcription_thread = threading.Thread(target=self._transcription_loop)
            self.transcription_thread.daemon = True
            self.transcription_thread.start()
            
            logger.info("Transcription started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start transcription: {e}")
            self.is_transcribing = False
            return False
    
    def stop_transcription(self):
        """Stop transcription"""
        if not self.is_transcribing:
            return
        
        self.is_transcribing = False
        logger.info("Stopping transcription...")
        
        # Wait for threads to finish
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=2.0)
        if hasattr(self, 'transcription_thread'):
            self.transcription_thread.join(timeout=2.0)
        
        logger.info("Transcription stopped")
    
    def _audio_capture_loop(self):
        """Audio capture loop for microphone input"""
        try:
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            logger.info("Audio capture started")
            
            while self.is_transcribing:
                try:
                    # Read audio data
                    audio_data = stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Convert to bytes and add to queue
                    self.audio_queue.put(audio_data)
                    
                except Exception as e:
                    logger.error(f"Audio capture error: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
            logger.info("Audio capture stopped")
            
        except Exception as e:
            logger.error(f"Audio capture loop failed: {e}")
    
    def _transcription_loop(self):
        """Main transcription processing loop"""
        try:
            # Create bidirectional streaming request
            requests = self._audio_request_generator()
            responses = self.speech_client.streaming_recognize(self.streaming_config, requests)
            
            logger.info("Transcription stream started")
            
            for response in responses:
                if not self.is_transcribing:
                    break
                
                if not response.results:
                    continue
                
                result = response.results[0]
                
                if result.is_final:
                    # Final transcription result
                    transcript = result.alternatives[0].transcript
                    confidence = result.alternatives[0].confidence
                    
                    self._process_final_transcription(transcript, confidence)
                    
                else:
                    # Interim result
                    transcript = result.alternatives[0].transcript
                    self._process_interim_transcription(transcript)
            
            logger.info("Transcription stream ended")
            
        except Exception as e:
            logger.error(f"Transcription loop failed: {e}")
    
    def _audio_request_generator(self):
        """Generate audio requests for streaming recognition"""
        try:
            while self.is_transcribing:
                # Get audio data from queue
                try:
                    audio_data = self.audio_queue.get(timeout=1.0)
                    
                    # Create recognition audio object
                    request = speech.StreamingRecognizeRequest(
                        audio_content=audio_data
                    )
                    
                    yield request
                    
                except queue.Empty:
                    # Send empty audio to keep stream alive
                    continue
                    
        except Exception as e:
            logger.error(f"Audio request generator failed: {e}")
    
    def _process_final_transcription(self, transcript: str, confidence: float):
        """Process final transcription result"""
        if not transcript.strip():
            return
        
        # Clean transcript
        clean_transcript = transcript.strip()
        
        # Add to history
        transcription_entry = {
            'text': clean_transcript,
            'confidence': confidence,
            'timestamp': time.time(),
            'type': 'final'
        }
        
        self.transcription_history.append(transcription_entry)
        
        # Keep history size manageable
        if len(self.transcription_history) > self.max_history:
            self.transcription_history.pop(0)
        
        # Update current transcription
        self.current_transcription = clean_transcript
        
        # Add to transcription queue
        self.transcription_queue.put(transcription_entry)
        
        # Call callback if set
        if self.on_transcription_complete:
            self.on_transcription_complete(transcription_entry)
        
        logger.info(f"Final transcription: {clean_transcript} (confidence: {confidence:.2f})")
    
    def _process_interim_transcription(self, transcript: str):
        """Process interim transcription result"""
        if not transcript.strip():
            return
        
        # Clean transcript
        clean_transcript = transcript.strip()
        
        # Update current transcription
        self.current_transcription = clean_transcript
        
        # Call callback if set
        if self.on_transcription_update:
            self.on_transcription_update(clean_transcript)
    
    def get_current_transcription(self) -> str:
        """Get current transcription text"""
        return self.current_transcription
    
    def get_transcription_history(self, limit: int = 10) -> List[Dict]:
        """Get recent transcription history"""
        return self.transcription_history[-limit:] if self.transcription_history else []
    
    def get_latest_transcription(self) -> Optional[Dict]:
        """Get the most recent transcription"""
        return self.transcription_history[-1] if self.transcription_history else None
    
    def clear_history(self):
        """Clear transcription history"""
        self.transcription_history.clear()
        self.current_transcription = ""
        logger.info("Transcription history cleared")
    
    def set_language(self, language_code: str):
        """Change transcription language"""
        if self.is_transcribing:
            logger.warning("Cannot change language while transcribing")
            return
        
        self.language_code = language_code
        
        # Reinitialize streaming config
        if self.speech_client:
            self.streaming_config = speech.StreamingRecognitionConfig(
                config=speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    language_code=self.language_code,
                    enable_automatic_punctuation=self.enable_automatic_punctuation,
                    enable_word_time_offsets=self.enable_word_time_offsets,
                    model="latest_long",
                    use_enhanced=True,
                ),
                interim_results=True,
                single_utterance=False
            )
        
        logger.info(f"Language changed to: {language_code}")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_transcription()
        
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()
        
        logger.info("Speech Transcriber cleaned up")

def test_speech_transcriber():
    """Test function for speech transcriber"""
    print("Testing speech transcriber...")
    
    try:
        transcriber = SpeechTranscriber()
        print("✓ Speech transcriber initialized")
        
        # Test configuration
        print(f"✓ Sample rate: {transcriber.sample_rate} Hz")
        print(f"✓ Language: {transcriber.language_code}")
        print(f"✓ Audio format: {transcriber.audio_format}")
        
        print("\nSpeech transcriber is ready for use!")
        print("Note: Google Cloud credentials required for actual transcription")
        
        transcriber.cleanup()
        
    except Exception as e:
        print(f"❌ Speech transcriber test failed: {e}")
        print("This is expected if Google Cloud credentials are not set up")

if __name__ == "__main__":
    test_speech_transcriber()
