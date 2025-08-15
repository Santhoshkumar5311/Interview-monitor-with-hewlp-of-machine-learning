"""
Whisper ASR Module for Streaming Audio Transcription
Provides real-time transcription using OpenAI's Whisper model with 5-second chunking
"""

import os
import time
import threading
import queue
import logging
import numpy as np
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from datetime import datetime
import whisper
import torch
import pyaudio
import wave
import tempfile

logger = logging.getLogger(__name__)

@dataclass
class AudioChunk:
    """Represents a 5-second audio chunk"""
    audio_data: np.ndarray
    timestamp: datetime
    sample_rate: int
    duration: float = 5.0

@dataclass
class TranscriptionResult:
    """Result of transcription analysis"""
    text: str
    confidence: float
    language: str
    timestamp: datetime
    duration: float
    is_final: bool = False

class WhisperASR:
    """Whisper-based ASR for real-time transcription"""
    
    def __init__(self, 
                 model_size: str = "base",
                 chunk_duration: float = 5.0,
                 sample_rate: int = 16000,
                 device: str = "auto"):
        """
        Initialize Whisper ASR
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            chunk_duration: Duration of audio chunks in seconds
            sample_rate: Audio sample rate
            device: Device to use (auto, cpu, cuda)
        """
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        
        # Initialize Whisper model
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        logger.info(f"Whisper model loaded on {self.device}")
        
        # Audio processing
        self.audio = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        
        # State
        self.is_running = False
        self.is_transcribing = False
        self.audio_thread = None
        self.transcription_thread = None
        
        # Callbacks
        self.on_transcription: Optional[Callable[[TranscriptionResult], None]] = None
        
        # Performance tracking
        self.chunk_count = 0
        self.total_processing_time = 0.0
        
        logger.info("Whisper ASR initialized successfully")
    
    def start_streaming(self, on_transcription: Optional[Callable] = None):
        """Start streaming audio transcription"""
        if self.is_running:
            logger.warning("ASR already running")
            return False
        
        if on_transcription:
            self.on_transcription = on_transcription
        
        try:
            # Start audio capture
            self.is_running = True
            self.audio_thread = threading.Thread(target=self._audio_capture_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # Start transcription processing
            self.is_transcribing = True
            self.transcription_thread = threading.Thread(target=self._transcription_loop)
            self.transcription_thread.daemon = True
            self.transcription_thread.start()
            
            logger.info("Whisper ASR streaming started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start ASR streaming: {e}")
            self.is_running = False
            return False
    
    def stop_streaming(self):
        """Stop streaming audio transcription"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.is_transcribing = False
        
        # Wait for threads to finish
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
        if self.transcription_thread:
            self.transcription_thread.join(timeout=2.0)
        
        logger.info("Whisper ASR streaming stopped")
    
    def _audio_capture_loop(self):
        """Audio capture loop for microphone input"""
        try:
            stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            stream.start_stream()
            logger.info("Audio capture started")
            
            while self.is_running:
                time.sleep(0.1)
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for real-time processing"""
        if not self.is_running:
            return (None, pyaudio.paComplete)
        
        try:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Create audio chunk
            chunk = AudioChunk(
                audio_data=audio_data,
                timestamp=datetime.now(),
                sample_rate=self.sample_rate,
                duration=self.chunk_duration
            )
            
            # Add to processing queue
            self.audio_queue.put(chunk)
            
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
        
        return (None, pyaudio.paContinue)
    
    def _transcription_loop(self):
        """Transcription processing loop"""
        while self.is_transcribing:
            try:
                # Get audio chunk from queue
                chunk = self.audio_queue.get(timeout=1.0)
                
                # Process chunk
                result = self._transcribe_chunk(chunk)
                
                if result:
                    # Add to transcription queue
                    self.transcription_queue.put(result)
                    
                    # Call callback if provided
                    if self.on_transcription:
                        self.on_transcription(result)
                    
                    # Update performance metrics
                    self.chunk_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Transcription loop error: {e}")
    
    def _transcribe_chunk(self, chunk: AudioChunk) -> Optional[TranscriptionResult]:
        """Transcribe a single audio chunk"""
        try:
            start_time = time.time()
            
            # Save audio chunk to temporary file (Whisper requires file input)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Write WAV file
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(4)  # 32-bit float
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(chunk.audio_data.tobytes())
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                temp_path,
                language="en",
                task="transcribe",
                fp16=False  # Use FP32 for better compatibility
            )
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Create result
            transcription_result = TranscriptionResult(
                text=result["text"].strip(),
                confidence=result.get("confidence", 0.0),
                language=result.get("language", "en"),
                timestamp=chunk.timestamp,
                duration=chunk.duration,
                is_final=True
            )
            
            logger.debug(f"Chunk transcribed in {processing_time:.2f}s: {result['text'][:50]}...")
            return transcription_result
            
        except Exception as e:
            logger.error(f"Chunk transcription failed: {e}")
            return None
    
    def get_latest_transcription(self) -> Optional[TranscriptionResult]:
        """Get the latest transcription result"""
        try:
            return self.transcription_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_transcription_history(self, limit: int = 10) -> List[TranscriptionResult]:
        """Get recent transcription history"""
        history = []
        temp_queue = queue.Queue()
        
        # Transfer items to temp queue while collecting history
        while not self.transcription_queue.empty():
            item = self.transcription_queue.get()
            temp_queue.put(item)
            history.append(item)
        
        # Restore items to original queue
        while not temp_queue.empty():
            self.transcription_queue.put(temp_queue.get())
        
        # Return limited history
        return history[-limit:] if len(history) > limit else history
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        avg_processing_time = (self.total_processing_time / self.chunk_count 
                             if self.chunk_count > 0 else 0.0)
        
        return {
            "chunks_processed": self.chunk_count,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": avg_processing_time,
            "chunks_per_second": self.chunk_count / (self.total_processing_time + 1e-6),
            "is_running": self.is_running,
            "is_transcribing": self.is_transcribing
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_streaming()
        if self.audio:
            self.audio.terminate()
        logger.info("Whisper ASR cleanup completed")
