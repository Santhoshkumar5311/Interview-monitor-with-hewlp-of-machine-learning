"""
Audio Prosody Analyzer Module
Extracts audio prosody features using pyAudioAnalysis for enhanced speech analysis
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import tempfile
import wave

try:
    from pyAudioAnalysis import audioBasicIO
    from pyAudioAnalysis import audioSegmentation
    from pyAudioAnalysis import audioFeatureExtraction
    from pyAudioAnalysis import audioTrainTest
    PYAUDIOANALYSIS_AVAILABLE = True
except ImportError:
    PYAUDIOANALYSIS_AVAILABLE = False
    logging.warning("pyAudioAnalysis not available. Using fallback methods.")

logger = logging.getLogger(__name__)

@dataclass
class ProsodyFeatures:
    """Audio prosody features extracted from audio chunks"""
    # Pitch features
    pitch_mean: float
    pitch_std: float
    pitch_range: float
    
    # Rate features
    speaking_rate: float  # words per second
    articulation_rate: float  # syllables per second
    pause_frequency: float  # pauses per second
    
    # Pause features
    total_pause_duration: float
    pause_ratio: float  # pause time / total time
    avg_pause_duration: float
    
    # Energy features
    energy_mean: float
    energy_std: float
    energy_entropy: float
    
    # Spectral features
    spectral_centroid: float
    spectral_rolloff: float
    spectral_bandwidth: float
    
    # MFCC features (first 5 coefficients)
    mfcc_1: float
    mfcc_2: float
    mfcc_3: float
    mfcc_4: float
    mfcc_5: float
    
    # Timestamp
    timestamp: datetime
    
    # Quality metrics
    confidence: float
    is_valid: bool = True

class AudioProsodyAnalyzer:
    """Analyzes audio prosody using pyAudioAnalysis"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 frame_size: float = 0.025,
                 hop_size: float = 0.010):
        """
        Initialize Audio Prosody Analyzer
        
        Args:
            sample_rate: Audio sample rate
            frame_size: Frame size in seconds for analysis
            hop_size: Hop size in seconds for analysis
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        
        if not PYAUDIOANALYSIS_AVAILABLE:
            logger.warning("pyAudioAnalysis not available. Prosody analysis will be limited.")
        
        logger.info("Audio Prosody Analyzer initialized")
    
    def analyze_audio_chunk(self, 
                           audio_data: np.ndarray, 
                           transcript: Optional[str] = None) -> ProsodyFeatures:
        """
        Analyze audio prosody from audio chunk
        
        Args:
            audio_data: Audio data as numpy array
            transcript: Optional transcript for rate calculations
            
        Returns:
            ProsodyFeatures object with extracted features
        """
        try:
            if not PYAUDIOANALYSIS_AVAILABLE:
                return self._fallback_analysis(audio_data, transcript)
            
            # Save audio to temporary file for pyAudioAnalysis
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Write WAV file
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
            
            # Extract features using pyAudioAnalysis
            features = self._extract_features_pyaudioanalysis(temp_path, transcript)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return features
            
        except Exception as e:
            logger.error(f"Audio prosody analysis failed: {e}")
            return self._create_default_features()
    
    def _extract_features_pyaudioanalysis(self, 
                                        audio_path: str, 
                                        transcript: Optional[str]) -> ProsodyFeatures:
        """Extract features using pyAudioAnalysis"""
        try:
            # Load audio
            [Fs, x] = audioBasicIO.readAudioFile(audio_path)
            
            # Extract short-term features
            F, f_names = audioFeatureExtraction.stFeatureExtraction(
                x, Fs, self.frame_size * Fs, self.hop_size * Fs
            )
            
            # Extract pitch features
            pitch_features = self._extract_pitch_features(F, f_names)
            
            # Extract energy features
            energy_features = self._extract_energy_features(F, f_names)
            
            # Extract spectral features
            spectral_features = self._extract_spectral_features(F, f_names)
            
            # Extract MFCC features
            mfcc_features = self._extract_mfcc_features(F, f_names)
            
            # Calculate speaking rate from transcript
            rate_features = self._calculate_rate_features(transcript, len(x) / Fs)
            
            # Calculate pause features
            pause_features = self._extract_pause_features(x, Fs)
            
            # Combine all features
            return ProsodyFeatures(
                # Pitch
                pitch_mean=pitch_features['mean'],
                pitch_std=pitch_features['std'],
                pitch_range=pitch_features['range'],
                
                # Rate
                speaking_rate=rate_features['speaking_rate'],
                articulation_rate=rate_features['articulation_rate'],
                pause_frequency=pause_features['frequency'],
                
                # Pause
                total_pause_duration=pause_features['total_duration'],
                pause_ratio=pause_features['ratio'],
                avg_pause_duration=pause_features['avg_duration'],
                
                # Energy
                energy_mean=energy_features['mean'],
                energy_std=energy_features['std'],
                energy_entropy=energy_features['entropy'],
                
                # Spectral
                spectral_centroid=spectral_features['centroid'],
                spectral_rolloff=spectral_features['rolloff'],
                spectral_bandwidth=spectral_features['bandwidth'],
                
                # MFCC
                mfcc_1=mfcc_features[0],
                mfcc_2=mfcc_features[1],
                mfcc_3=mfcc_features[2],
                mfcc_4=mfcc_features[3],
                mfcc_5=mfcc_features[4],
                
                # Metadata
                timestamp=datetime.now(),
                confidence=0.8,
                is_valid=True
            )
            
        except Exception as e:
            logger.error(f"pyAudioAnalysis feature extraction failed: {e}")
            return self._create_default_features()
    
    def _extract_pitch_features(self, F: np.ndarray, f_names: List[str]) -> Dict:
        """Extract pitch-related features"""
        try:
            # Find pitch feature indices
            pitch_idx = [i for i, name in enumerate(f_names) if 'pitch' in name.lower()]
            
            if pitch_idx:
                pitch_values = F[pitch_idx, :].flatten()
                pitch_values = pitch_values[pitch_values > 0]  # Remove zeros
                
                if len(pitch_values) > 0:
                    return {
                        'mean': float(np.mean(pitch_values)),
                        'std': float(np.std(pitch_values)),
                        'range': float(np.max(pitch_values) - np.min(pitch_values))
                    }
            
            # Fallback values
            return {'mean': 200.0, 'std': 50.0, 'range': 100.0}
            
        except Exception as e:
            logger.error(f"Pitch feature extraction failed: {e}")
            return {'mean': 200.0, 'std': 50.0, 'range': 100.0}
    
    def _extract_energy_features(self, F: np.ndarray, f_names: List[str]) -> Dict:
        """Extract energy-related features"""
        try:
            # Find energy feature indices
            energy_idx = [i for i, name in enumerate(f_names) if 'energy' in name.lower()]
            
            if energy_idx:
                energy_values = F[energy_idx, :].flatten()
                
                return {
                    'mean': float(np.mean(energy_values)),
                    'std': float(np.std(energy_values)),
                    'entropy': float(self._calculate_entropy(energy_values))
                }
            
            # Fallback values
            return {'mean': 0.5, 'std': 0.2, 'entropy': 0.7}
            
        except Exception as e:
            logger.error(f"Energy feature extraction failed: {e}")
            return {'mean': 0.5, 'std': 0.2, 'entropy': 0.7}
    
    def _extract_spectral_features(self, F: np.ndarray, f_names: List[str]) -> Dict:
        """Extract spectral features"""
        try:
            features = {}
            
            # Spectral centroid
            centroid_idx = [i for i, name in enumerate(f_names) if 'centroid' in name.lower()]
            if centroid_idx:
                features['centroid'] = float(np.mean(F[centroid_idx, :]))
            else:
                features['centroid'] = 2000.0
            
            # Spectral rolloff
            rolloff_idx = [i for i, name in enumerate(f_names) if 'rolloff' in name.lower()]
            if rolloff_idx:
                features['rolloff'] = float(np.mean(F[rolloff_idx, :]))
            else:
                features['rolloff'] = 4000.0
            
            # Spectral bandwidth
            bandwidth_idx = [i for i, name in enumerate(f_names) if 'bandwidth' in name.lower()]
            if bandwidth_idx:
                features['bandwidth'] = float(np.mean(F[bandwidth_idx, :]))
            else:
                features['bandwidth'] = 1000.0
            
            return features
            
        except Exception as e:
            logger.error(f"Spectral feature extraction failed: {e}")
            return {'centroid': 2000.0, 'rolloff': 4000.0, 'bandwidth': 1000.0}
    
    def _extract_mfcc_features(self, F: np.ndarray, f_names: List[str]) -> List[float]:
        """Extract MFCC features"""
        try:
            # Find MFCC feature indices
            mfcc_idx = [i for i, name in enumerate(f_names) if 'mfcc' in name.lower()]
            
            if len(mfcc_idx) >= 5:
                mfcc_values = [float(np.mean(F[i, :])) for i in mfcc_idx[:5]]
                return mfcc_values
            else:
                # Fallback values
                return [0.0, 0.0, 0.0, 0.0, 0.0]
                
        except Exception as e:
            logger.error(f"MFCC feature extraction failed: {e}")
            return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    def _calculate_rate_features(self, transcript: Optional[str], duration: float) -> Dict:
        """Calculate speaking rate features"""
        try:
            if not transcript or duration <= 0:
                return {
                    'speaking_rate': 2.0,
                    'articulation_rate': 4.0
                }
            
            # Count words and syllables
            words = transcript.split()
            word_count = len(words)
            
            # Simple syllable counting (approximate)
            syllable_count = sum(self._count_syllables(word) for word in words)
            
            return {
                'speaking_rate': word_count / duration,
                'articulation_rate': syllable_count / duration
            }
            
        except Exception as e:
            logger.error(f"Rate calculation failed: {e}")
            return {'speaking_rate': 2.0, 'articulation_rate': 4.0}
    
    def _extract_pause_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Extract pause-related features"""
        try:
            # Simple energy-based pause detection
            frame_size = int(0.025 * sample_rate)  # 25ms frames
            hop_size = int(0.010 * sample_rate)    # 10ms hop
            
            energy_frames = []
            for i in range(0, len(audio_data) - frame_size, hop_size):
                frame = audio_data[i:i + frame_size]
                energy = np.mean(frame ** 2)
                energy_frames.append(energy)
            
            if not energy_frames:
                return {'frequency': 0.0, 'total_duration': 0.0, 'ratio': 0.0, 'avg_duration': 0.0}
            
            # Calculate energy threshold (10th percentile)
            energy_threshold = np.percentile(energy_frames, 10)
            
            # Detect pauses
            pause_frames = [i for i, energy in enumerate(energy_frames) if energy < energy_threshold]
            
            if not pause_frames:
                return {'frequency': 0.0, 'total_duration': 0.0, 'ratio': 0.0, 'avg_duration': 0.0}
            
            # Calculate pause statistics
            pause_duration = len(pause_frames) * hop_size / sample_rate
            total_duration = len(audio_data) / sample_rate
            pause_ratio = pause_duration / total_duration
            
            # Count pause segments
            pause_segments = self._count_pause_segments(pause_frames)
            pause_frequency = pause_segments / total_duration
            
            avg_pause_duration = pause_duration / pause_segments if pause_segments > 0 else 0.0
            
            return {
                'frequency': pause_frequency,
                'total_duration': pause_duration,
                'ratio': pause_ratio,
                'avg_duration': avg_pause_duration
            }
            
        except Exception as e:
            logger.error(f"Pause feature extraction failed: {e}")
            return {'frequency': 0.0, 'total_duration': 0.0, 'ratio': 0.0, 'avg_duration': 0.0}
    
    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting (approximate)"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        return max(count, 1)
    
    def _count_pause_segments(self, pause_frames: List[int]) -> int:
        """Count continuous pause segments"""
        if not pause_frames:
            return 0
        
        segments = 1
        for i in range(1, len(pause_frames)):
            if pause_frames[i] != pause_frames[i-1] + 1:
                segments += 1
        
        return segments
    
    def _calculate_entropy(self, values: np.ndarray) -> float:
        """Calculate entropy of values"""
        try:
            # Normalize values to probabilities
            values = values - np.min(values)
            if np.sum(values) == 0:
                return 0.0
            
            probs = values / np.sum(values)
            probs = probs[probs > 0]  # Remove zero probabilities
            
            entropy = -np.sum(probs * np.log2(probs))
            return float(entropy)
            
        except Exception:
            return 0.0
    
    def _fallback_analysis(self, audio_data: np.ndarray, transcript: Optional[str]) -> ProsodyFeatures:
        """Fallback analysis when pyAudioAnalysis is not available"""
        logger.info("Using fallback audio analysis")
        
        # Basic energy analysis
        energy = np.mean(audio_data ** 2)
        energy_std = np.std(audio_data ** 2)
        
        # Basic rate calculation
        rate_features = self._calculate_rate_features(transcript, len(audio_data) / self.sample_rate)
        
        return ProsodyFeatures(
            # Pitch (fallback)
            pitch_mean=200.0,
            pitch_std=50.0,
            pitch_range=100.0,
            
            # Rate
            speaking_rate=rate_features['speaking_rate'],
            articulation_rate=rate_features['articulation_rate'],
            pause_frequency=2.0,
            
            # Pause (fallback)
            total_pause_duration=0.5,
            pause_ratio=0.1,
            avg_pause_duration=0.25,
            
            # Energy
            energy_mean=float(energy),
            energy_std=float(energy_std),
            energy_entropy=0.5,
            
            # Spectral (fallback)
            spectral_centroid=2000.0,
            spectral_rolloff=4000.0,
            spectral_bandwidth=1000.0,
            
            # MFCC (fallback)
            mfcc_1=0.0,
            mfcc_2=0.0,
            mfcc_3=0.0,
            mfcc_4=0.0,
            mfcc_5=0.0,
            
            # Metadata
            timestamp=datetime.now(),
            confidence=0.5,
            is_valid=True
        )
    
    def _create_default_features(self) -> ProsodyFeatures:
        """Create default features when analysis fails"""
        return ProsodyFeatures(
            pitch_mean=200.0, pitch_std=50.0, pitch_range=100.0,
            speaking_rate=2.0, articulation_rate=4.0, pause_frequency=2.0,
            total_pause_duration=0.5, pause_ratio=0.1, avg_pause_duration=0.25,
            energy_mean=0.5, energy_std=0.2, energy_entropy=0.5,
            spectral_centroid=2000.0, spectral_rolloff=4000.0, spectral_bandwidth=1000.0,
            mfcc_1=0.0, mfcc_2=0.0, mfcc_3=0.0, mfcc_4=0.0, mfcc_5=0.0,
            timestamp=datetime.now(), confidence=0.0, is_valid=False
        )
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return [
            'pitch_mean', 'pitch_std', 'pitch_range',
            'speaking_rate', 'articulation_rate', 'pause_frequency',
            'total_pause_duration', 'pause_ratio', 'avg_pause_duration',
            'energy_mean', 'energy_std', 'energy_entropy',
            'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth',
            'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5'
        ]
    
    def normalize_features(self, features: ProsodyFeatures) -> np.ndarray:
        """Convert features to normalized numpy array"""
        feature_values = [
            features.pitch_mean, features.pitch_std, features.pitch_range,
            features.speaking_rate, features.articulation_rate, features.pause_frequency,
            features.total_pause_duration, features.pause_ratio, features.avg_pause_duration,
            features.energy_mean, features.energy_std, features.energy_entropy,
            features.spectral_centroid, features.spectral_rolloff, features.spectral_bandwidth,
            features.mfcc_1, features.mfcc_2, features.mfcc_3, features.mfcc_4, features.mfcc_5
        ]
        
        return np.array(feature_values, dtype=np.float32)
