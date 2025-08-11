"""
Speech Analyzer Module
Analyzes speech characteristics for confidence assessment
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import librosa
import soundfile as sf
from scipy import signal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechAnalyzer:
    """Analyzes speech characteristics for confidence assessment"""
    
    def __init__(self):
        """Initialize speech analyzer"""
        self.sample_rate = 16000  # Standard sample rate
        
        # Speech analysis parameters
        self.volume_thresholds = {
            'very_low': 0.1,
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'very_high': 0.9
        }
        
        self.pace_thresholds = {
            'very_slow': 2.0,    # words per second
            'slow': 3.0,
            'normal': 4.0,
            'fast': 5.0,
            'very_fast': 6.0
        }
        
        logger.info("Speech Analyzer initialized")
    
    def analyze_speech_characteristics(self, audio_data: np.ndarray, transcription: str = "") -> Dict:
        """
        Analyze speech characteristics from audio data
        
        Args:
            audio_data: Audio data as numpy array
            transcription: Optional transcription text for pace analysis
            
        Returns:
            Dict containing speech analysis results
        """
        if audio_data is None or len(audio_data) == 0:
            return self._get_neutral_speech_data()
        
        try:
            # Analyze different speech characteristics
            volume_analysis = self._analyze_volume(audio_data)
            pace_analysis = self._analyze_pace(audio_data, transcription)
            clarity_analysis = self._analyze_clarity(audio_data)
            tone_analysis = self._analyze_tone(audio_data)
            
            # Calculate overall speech confidence score
            speech_confidence = self._calculate_speech_confidence(
                volume_analysis, pace_analysis, clarity_analysis, tone_analysis
            )
            
            return {
                'overall_confidence': speech_confidence,
                'volume': volume_analysis,
                'pace': pace_analysis,
                'clarity': clarity_analysis,
                'tone': tone_analysis,
                'recommendations': self._get_speech_recommendations(
                    volume_analysis, pace_analysis, clarity_analysis, tone_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Speech analysis failed: {str(e)}")
            return self._get_neutral_speech_data()
    
    def _analyze_volume(self, audio_data: np.ndarray) -> Dict:
        """Analyze speech volume"""
        try:
            # Calculate RMS (Root Mean Square) volume
            rms_volume = np.sqrt(np.mean(audio_data**2))
            
            # Normalize volume to 0-1 scale
            normalized_volume = min(rms_volume / 0.5, 1.0)  # 0.5 as reference max
            
            # Determine volume level
            if normalized_volume < self.volume_thresholds['very_low']:
                volume_level = 'very_low'
                volume_score = 0.2
            elif normalized_volume < self.volume_thresholds['low']:
                volume_level = 'low'
                volume_score = 0.4
            elif normalized_volume < self.volume_thresholds['medium']:
                volume_level = 'medium'
                volume_score = 0.7
            elif normalized_volume < self.volume_thresholds['high']:
                volume_level = 'high'
                volume_score = 0.9
            else:
                volume_level = 'very_high'
                volume_score = 0.8  # Slightly lower than high for optimal
            
            return {
                'level': volume_level,
                'score': volume_score,
                'rms_volume': rms_volume,
                'normalized_volume': normalized_volume
            }
            
        except Exception as e:
            logger.error(f"Volume analysis failed: {e}")
            return {'level': 'medium', 'score': 0.5, 'rms_volume': 0.0, 'normalized_volume': 0.5}
    
    def _analyze_pace(self, audio_data: np.ndarray, transcription: str = "") -> Dict:
        """Analyze speech pace (speed)"""
        try:
            if transcription:
                # Analyze pace from transcription (words per second)
                words = transcription.split()
                duration = len(audio_data) / self.sample_rate
                words_per_second = len(words) / duration if duration > 0 else 0
                
                # Determine pace level
                if words_per_second < self.pace_thresholds['very_slow']:
                    pace_level = 'very_slow'
                    pace_score = 0.3
                elif words_per_second < self.pace_thresholds['slow']:
                    pace_level = 'slow'
                    pace_score = 0.5
                elif words_per_second < self.pace_thresholds['normal']:
                    pace_level = 'normal'
                    pace_score = 0.9
                elif words_per_second < self.pace_thresholds['fast']:
                    pace_level = 'fast'
                    pace_score = 0.7
                else:
                    pace_level = 'very_fast'
                    pace_score = 0.4
                
                return {
                    'level': pace_level,
                    'score': pace_score,
                    'words_per_second': words_per_second,
                    'duration': duration,
                    'word_count': len(words)
                }
            else:
                # Analyze pace from audio characteristics (energy variations)
                # This is a simplified approach when transcription is not available
                energy = np.sum(audio_data**2)
                energy_variations = np.std(np.diff(energy))
                
                # Normalize to pace score
                pace_score = min(energy_variations / 0.1, 1.0)  # 0.1 as reference
                
                return {
                    'level': 'unknown',
                    'score': pace_score,
                    'energy_variations': energy_variations,
                    'note': 'Pace estimated from audio energy (transcription recommended)'
                }
                
        except Exception as e:
            logger.error(f"Pace analysis failed: {e}")
            return {'level': 'normal', 'score': 0.5, 'words_per_second': 0.0, 'duration': 0.0}
    
    def _analyze_clarity(self, audio_data: np.ndarray) -> Dict:
        """Analyze speech clarity (articulation)"""
        try:
            # Calculate spectral centroid (indicates clarity)
            # Higher centroid = clearer speech
            fft = np.fft.fft(audio_data)
            frequencies = np.fft.fftfreq(len(audio_data), 1/self.sample_rate)
            
            # Focus on speech frequency range (80Hz - 8000Hz)
            speech_mask = (frequencies >= 80) & (frequencies <= 8000)
            speech_fft = fft[speech_mask]
            speech_freqs = frequencies[speech_mask]
            
            if len(speech_freqs) > 0:
                spectral_centroid = np.sum(np.abs(speech_fft) * speech_freqs) / np.sum(np.abs(speech_fft))
                # Normalize centroid to 0-1 scale
                clarity_score = min(spectral_centroid / 4000, 1.0)  # 4000Hz as reference
            else:
                clarity_score = 0.5
            
            # Determine clarity level
            if clarity_score < 0.3:
                clarity_level = 'poor'
            elif clarity_score < 0.6:
                clarity_level = 'fair'
            elif clarity_score < 0.8:
                clarity_level = 'good'
            else:
                clarity_level = 'excellent'
            
            return {
                'level': clarity_level,
                'score': clarity_score,
                'spectral_centroid': spectral_centroid if 'spectral_centroid' in locals() else 0.0
            }
            
        except Exception as e:
            logger.error(f"Clarity analysis failed: {e}")
            return {'level': 'fair', 'score': 0.5, 'spectral_centroid': 0.0}
    
    def _analyze_tone(self, audio_data: np.ndarray) -> Dict:
        """Analyze speech tone (pitch and stability)"""
        try:
            # Calculate pitch using autocorrelation
            # This is a simplified pitch detection
            autocorr = signal.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find first peak after zero lag
            peaks = signal.find_peaks(autocorr, height=0.1*np.max(autocorr))[0]
            
            if len(peaks) > 0:
                # Calculate fundamental frequency
                fundamental_freq = self.sample_rate / peaks[0] if peaks[0] > 0 else 0
                
                # Normalize pitch to 0-1 scale (typical speech range: 80-400Hz)
                pitch_score = min(max((fundamental_freq - 80) / 320, 0), 1)
                
                # Analyze tone stability (lower std = more stable)
                pitch_stability = 1.0 - min(np.std(autocorr) / np.mean(autocorr), 1.0)
                
                # Combined tone score
                tone_score = (pitch_score + pitch_stability) / 2
                
                return {
                    'score': tone_score,
                    'fundamental_frequency': fundamental_freq,
                    'pitch_score': pitch_score,
                    'stability_score': pitch_stability
                }
            else:
                return {'score': 0.5, 'fundamental_frequency': 0.0, 'pitch_score': 0.5, 'stability_score': 0.5}
                
        except Exception as e:
            logger.error(f"Tone analysis failed: {e}")
            return {'score': 0.5, 'fundamental_frequency': 0.0, 'pitch_score': 0.5, 'stability_score': 0.5}
    
    def _calculate_speech_confidence(self, volume: Dict, pace: Dict, clarity: Dict, tone: Dict) -> float:
        """Calculate overall speech confidence score"""
        # Weighted combination of speech characteristics
        weights = {
            'volume': 0.25,
            'pace': 0.25,
            'clarity': 0.3,
            'tone': 0.2
        }
        
        overall_confidence = (
            weights['volume'] * volume.get('score', 0.5) +
            weights['pace'] * pace.get('score', 0.5) +
            weights['clarity'] * clarity.get('score', 0.5) +
            weights['tone'] * tone.get('score', 0.5)
        )
        
        return overall_confidence
    
    def _get_speech_recommendations(self, volume: Dict, pace: Dict, clarity: Dict, tone: Dict) -> List[str]:
        """Get speech improvement recommendations"""
        recommendations = []
        
        # Volume recommendations
        if volume.get('score', 0.5) < 0.6:
            if volume.get('level') in ['very_low', 'low']:
                recommendations.append("Speak louder and more clearly")
            elif volume.get('level') == 'very_high':
                recommendations.append("Lower your voice slightly for better clarity")
        
        # Pace recommendations
        if pace.get('score', 0.5) < 0.6:
            if pace.get('level') in ['very_slow', 'slow']:
                recommendations.append("Try to speak at a more natural pace")
            elif pace.get('level') in ['fast', 'very_fast']:
                recommendations.append("Slow down your speech for better understanding")
        
        # Clarity recommendations
        if clarity.get('score', 0.5) < 0.6:
            recommendations.append("Focus on clear articulation of words")
            recommendations.append("Take your time with difficult words")
        
        # Tone recommendations
        if tone.get('score', 0.5) < 0.6:
            recommendations.append("Maintain a steady, confident tone")
            recommendations.append("Avoid monotone speech")
        
        if not recommendations:
            recommendations.append("Excellent speech characteristics! Keep it up!")
        
        return recommendations
    
    def _get_neutral_speech_data(self) -> Dict:
        """Return neutral speech data when analysis fails"""
        return {
            'overall_confidence': 0.5,
            'volume': {'level': 'medium', 'score': 0.5, 'rms_volume': 0.0, 'normalized_volume': 0.5},
            'pace': {'level': 'normal', 'score': 0.5, 'words_per_second': 0.0, 'duration': 0.0},
            'clarity': {'level': 'fair', 'score': 0.5, 'spectral_centroid': 0.0},
            'tone': {'score': 0.5, 'fundamental_frequency': 0.0, 'pitch_score': 0.5, 'stability_score': 0.5},
            'recommendations': ['Speech analysis not available']
        }

def test_speech_analyzer():
    """Test function for speech analyzer"""
    print("Testing speech analyzer...")
    
    analyzer = SpeechAnalyzer()
    print("✓ Speech analyzer initialized")
    
    # Create dummy audio data for testing
    sample_rate = 16000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate test audio (sine wave + noise)
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    
    # Test speech analysis
    analysis = analyzer.analyze_speech_characteristics(test_audio, "This is a test transcription for speech analysis")
    
    print(f"✓ Overall speech confidence: {analysis['overall_confidence']:.2f}")
    print(f"✓ Volume: {analysis['volume']['level']} (score: {analysis['volume']['score']:.2f})")
    print(f"✓ Pace: {analysis['pace']['level']} (score: {analysis['pace']['score']:.2f})")
    print(f"✓ Clarity: {analysis['clarity']['level']} (score: {analysis['clarity']['score']:.2f})")
    print(f"✓ Tone score: {analysis['tone']['score']:.2f}")
    
    print("\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  • {rec}")
    
    print("\nSpeech analyzer is ready for use!")

if __name__ == "__main__":
    test_speech_analyzer()
