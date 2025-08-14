"""
Enhanced Confidence Classifier Module
Integrates facial expressions, speech characteristics, and advanced sentiment analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json

# Import our modules
from .sentiment_analyzer import SentimentAnalyzer
from .speech_analyzer import SpeechAnalyzer
from .advanced_sentiment_analyzer import AdvancedSentimentAnalyzer, AdvancedSentimentResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence levels for candidates"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class RelevanceLevel(Enum):
    """Relevance levels for responses"""
    IRRELEVANT = "irrelevant"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class FacialMetrics:
    """Facial expression metrics"""
    smile_intensity: float  # 0 to 1
    frown_intensity: float  # 0 to 1
    eyebrow_raise: float    # 0 to 1
    eye_contact: float      # 0 to 1 (percentage of time looking at camera)
    head_tilt: float        # -1 to 1 (left to right)
    head_nod: float         # -1 to 1 (down to up)
    blink_rate: float       # blinks per second
    mouth_open: float       # 0 to 1

@dataclass
class SpeechMetrics:
    """Speech characteristics metrics"""
    volume: float           # 0 to 1 (normalized)
    pace: float            # 0 to 1 (slow to fast)
    clarity: float         # 0 to 1 (unclear to clear)
    tone: float            # -1 to 1 (monotone to expressive)
    pause_frequency: float # 0 to 1 (frequent to rare)
    pitch_variation: float # 0 to 1 (monotone to varied)

@dataclass
class FusionMetrics:
    """Combined metrics with EMA smoothing"""
    confidence_score: float      # 0 to 1
    confidence_level: ConfidenceLevel
    relevance_score: float       # 0 to 1
    relevance_level: RelevanceLevel
    sentiment_score: float       # -1 to 1
    emotion_label: str
    toxicity_score: float        # 0 to 1
    politeness_score: float      # 0 to 1
    overall_score: float         # 0 to 1 (combined metric)
    
    # Timestamps for tracking
    timestamp: datetime
    chunk_duration: float        # seconds
    
    # Raw component scores
    facial_score: float
    speech_score: float
    sentiment_component: float

class EMASmoother:
    """Exponential Moving Average for smoothing metrics"""
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize EMA smoother
        
        Args:
            alpha: Smoothing factor (0 to 1). Lower = more smoothing
        """
        self.alpha = alpha
        self.value = None
        self.initialized = False
    
    def update(self, new_value: float) -> float:
        """Update EMA with new value"""
        if not self.initialized:
            self.value = new_value
            self.initialized = True
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        
        return self.value
    
    def get_value(self) -> Optional[float]:
        """Get current EMA value"""
        return self.value

class ConfidenceClassifier:
    """Enhanced confidence classifier with fusion layer"""
    
    def __init__(self, use_advanced_sentiment: bool = True):
        """
        Initialize confidence classifier
        
        Args:
            use_advanced_sentiment: Whether to use advanced sentiment analysis
        """
        # Initialize analyzers
        self.sentiment_analyzer = SentimentAnalyzer()
        self.speech_analyzer = SpeechAnalyzer()
        
        if use_advanced_sentiment:
            try:
                self.advanced_sentiment_analyzer = AdvancedSentimentAnalyzer()
                self.use_advanced = True
                logger.info("✓ Advanced sentiment analyzer initialized")
            except Exception as e:
                logger.warning(f"Advanced sentiment analyzer failed: {e}")
                self.use_advanced = False
        else:
            self.use_advanced = False
        
        # Initialize EMA smoothers for each metric
        self.confidence_smoother = EMASmoother(alpha=0.3)
        self.relevance_smoother = EMASmoother(alpha=0.3)
        self.sentiment_smoother = EMASmoother(alpha=0.3)
        self.facial_smoother = EMASmoother(alpha=0.4)
        self.speech_smoother = EMASmoother(alpha=0.4)
        
        # Session tracking
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.chunk_count = 0
        
        # Performance tracking
        self.performance_history = []
        
        logger.info(f"Confidence Classifier initialized (Session: {self.session_id})")
    
    def analyze_chunk(self, 
                     transcript: str,
                     facial_metrics: Optional[FacialMetrics] = None,
                     speech_metrics: Optional[SpeechMetrics] = None,
                     chunk_duration: float = 5.0) -> FusionMetrics:
        """
        Analyze a transcript chunk and produce fusion metrics
        
        Args:
            transcript: Text transcript to analyze
            facial_metrics: Facial expression metrics
            speech_metrics: Speech characteristics metrics
            chunk_duration: Duration of the chunk in seconds
            
        Returns:
            FusionMetrics with all combined analysis
        """
        try:
            self.chunk_count += 1
            timestamp = datetime.now()
            
            # 1. Advanced sentiment analysis
            if self.use_advanced and transcript.strip():
                sentiment_result = self.advanced_sentiment_analyzer.analyze_sentiment(transcript)
                sentiment_score = sentiment_result.overall_sentiment
                emotion_label = sentiment_result.primary_emotion.value
                toxicity_score = sentiment_result.toxicity_score
                politeness_score = sentiment_result.politeness_score
            else:
                # Fallback to basic sentiment
                basic_sentiment = self.sentiment_analyzer.analyze_sentiment(transcript)
                sentiment_score = basic_sentiment.get('polarity', 0.0)
                emotion_label = "neutral"
                toxicity_score = 0.0
                politeness_score = 0.5
            
            # 2. Calculate component scores
            facial_score = self._calculate_facial_score(facial_metrics) if facial_metrics else 0.5
            speech_score = self._calculate_speech_score(speech_metrics) if speech_metrics else 0.5
            sentiment_component = self._normalize_sentiment(sentiment_score)
            
            # 3. Apply EMA smoothing
            smoothed_facial = self.facial_smoother.update(facial_score)
            smoothed_speech = self.speech_smoother.update(speech_score)
            smoothed_sentiment = self.sentiment_smoother.update(sentiment_component)
            
            # 4. Fusion algorithm - weighted combination
            confidence_score = self._fuse_confidence_metrics(
                smoothed_facial, smoothed_speech, smoothed_sentiment
            )
            relevance_score = self._fuse_relevance_metrics(
                transcript, sentiment_score, toxicity_score
            )
            
            # 5. Apply final smoothing
            smoothed_confidence = self.confidence_smoother.update(confidence_score)
            smoothed_relevance = self.relevance_smoother.update(relevance_score)
            
            # 6. Determine levels
            confidence_level = self._classify_confidence(smoothed_confidence)
            relevance_level = self._classify_relevance(smoothed_relevance)
            
            # 7. Calculate overall score
            overall_score = (smoothed_confidence + smoothed_relevance + smoothed_sentiment) / 3
            
            # 8. Create fusion metrics
            fusion_metrics = FusionMetrics(
                confidence_score=smoothed_confidence,
                confidence_level=confidence_level,
                relevance_score=smoothed_relevance,
                relevance_level=relevance_level,
                sentiment_score=sentiment_score,
                emotion_label=emotion_label,
                toxicity_score=toxicity_score,
                politeness_score=politeness_score,
                overall_score=overall_score,
                timestamp=timestamp,
                chunk_duration=chunk_duration,
                facial_score=smoothed_facial,
                speech_score=smoothed_speech,
                sentiment_component=smoothed_sentiment
            )
            
            # 9. Log performance data
            self._log_performance(fusion_metrics, transcript)
            
            return fusion_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing chunk: {e}")
            return self._get_default_metrics(timestamp, chunk_duration)
    
    def _calculate_facial_score(self, metrics: FacialMetrics) -> float:
        """Calculate facial confidence score from metrics"""
        # Positive indicators
        positive_score = (
            metrics.smile_intensity * 0.3 +
            metrics.eye_contact * 0.4 +
            (1 - abs(metrics.head_tilt)) * 0.2 +
            (1 - abs(metrics.head_nod)) * 0.1
        )
        
        # Negative indicators
        negative_score = (
            metrics.frown_intensity * 0.3 +
            metrics.eyebrow_raise * 0.2 +
            metrics.blink_rate * 0.2 +
            metrics.mouth_open * 0.3
        )
        
        # Normalize to 0-1
        facial_score = max(0, positive_score - negative_score)
        return min(1.0, facial_score)
    
    def _calculate_speech_score(self, metrics: SpeechMetrics) -> float:
        """Calculate speech confidence score from metrics"""
        # Positive indicators
        positive_score = (
            metrics.volume * 0.2 +
            metrics.clarity * 0.4 +
            metrics.tone * 0.3 +
            (1 - metrics.pause_frequency) * 0.1
        )
        
        # Pace should be moderate (not too fast, not too slow)
        pace_score = 1 - abs(metrics.pace - 0.5) * 2
        
        speech_score = (positive_score + pace_score) / 2
        return max(0, min(1, speech_score))
    
    def _normalize_sentiment(self, sentiment: float) -> float:
        """Normalize sentiment from [-1, 1] to [0, 1]"""
        return (sentiment + 1) / 2
    
    def _fuse_confidence_metrics(self, facial: float, speech: float, sentiment: float) -> float:
        """Fuse facial, speech, and sentiment metrics for confidence"""
        # Weighted combination - facial expressions are most important
        weights = {
            'facial': 0.5,
            'speech': 0.3,
            'sentiment': 0.2
        }
        
        confidence_score = (
            facial * weights['facial'] +
            speech * weights['speech'] +
            sentiment * weights['sentiment']
        )
        
        return max(0, min(1, confidence_score))
    
    def _fuse_relevance_metrics(self, transcript: str, sentiment: float, toxicity: float) -> float:
        """Fuse metrics for relevance scoring"""
        # Base relevance from transcript length and content
        word_count = len(transcript.split())
        base_relevance = min(1.0, word_count / 20.0)  # Normalize to 20 words
        
        # Sentiment contribution (positive sentiment suggests relevance)
        sentiment_contribution = (sentiment + 1) / 2
        
        # Toxicity penalty (toxic content is less relevant)
        toxicity_penalty = 1 - toxicity
        
        # Combine metrics
        relevance_score = (
            base_relevance * 0.4 +
            sentiment_contribution * 0.4 +
            toxicity_penalty * 0.2
        )
        
        return max(0, min(1, relevance_score))
    
    def _classify_confidence(self, score: float) -> ConfidenceLevel:
        """Classify confidence level from score"""
        if score < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif score < 0.4:
            return ConfidenceLevel.LOW
        elif score < 0.6:
            return ConfidenceLevel.MEDIUM
        elif score < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    def _classify_relevance(self, score: float) -> RelevanceLevel:
        """Classify relevance level from score"""
        if score < 0.2:
            return RelevanceLevel.IRRELEVANT
        elif score < 0.4:
            return RelevanceLevel.LOW
        elif score < 0.6:
            return RelevanceLevel.MEDIUM
        elif score < 0.8:
            return RelevanceLevel.HIGH
        else:
            return RelevanceLevel.VERY_HIGH
    
    def _log_performance(self, metrics: FusionMetrics, transcript: str):
        """Log performance data for analysis"""
        log_entry = {
            "session_id": self.session_id,
            "chunk_id": self.chunk_count,
            "timestamp": metrics.timestamp.isoformat(),
            "chunk_duration": metrics.chunk_duration,
            "text": transcript,
            "sentiment": round(metrics.sentiment_score, 3),
            "emotion": metrics.emotion_label,
            "toxicity": round(metrics.toxicity_score, 3),
            "politeness": round(metrics.politeness_score, 3),
            "confidence": round(metrics.confidence_score, 3),
            "relevance": round(metrics.relevance_score, 3),
            "overall_score": round(metrics.overall_score, 3),
            "facial_score": round(metrics.facial_score, 3),
            "speech_score": round(metrics.speech_score, 3),
            "sentiment_component": round(metrics.sentiment_component, 3)
        }
        
        self.performance_history.append(log_entry)
        
        # Log to console for debugging
        logger.info(f"Chunk {self.chunk_count}: Confidence={metrics.confidence_level.value} "
                   f"({metrics.confidence_score:.2f}), Relevance={metrics.relevance_level.value} "
                   f"({metrics.relevance_score:.2f}), Emotion={metrics.emotion_label}")
    
    def _get_default_metrics(self, timestamp: datetime, chunk_duration: float) -> FusionMetrics:
        """Return default metrics when analysis fails"""
        return FusionMetrics(
            confidence_score=0.5,
            confidence_level=ConfidenceLevel.MEDIUM,
            relevance_score=0.5,
            relevance_level=RelevanceLevel.MEDIUM,
            sentiment_score=0.0,
            emotion_label="neutral",
            toxicity_score=0.0,
            politeness_score=0.5,
            overall_score=0.5,
            timestamp=timestamp,
            chunk_duration=chunk_duration,
            facial_score=0.5,
            speech_score=0.5,
            sentiment_component=0.5
        )
    
    def get_session_summary(self) -> Dict:
        """Get summary statistics for the current session"""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        # Calculate averages
        avg_confidence = np.mean([m['confidence'] for m in self.performance_history])
        avg_relevance = np.mean([m['relevance'] for m in self.performance_history])
        avg_sentiment = np.mean([m['sentiment'] for m in self.performance_history])
        avg_overall = np.mean([m['overall_score'] for m in self.performance_history])
        
        # Count emotions
        emotion_counts = {}
        for m in self.performance_history:
            emotion = m['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Session duration
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        return {
            "session_id": self.session_id,
            "session_duration_seconds": session_duration,
            "total_chunks": self.chunk_count,
            "averages": {
                "confidence": round(avg_confidence, 3),
                "relevance": round(avg_relevance, 3),
                "sentiment": round(avg_sentiment, 3),
                "overall_score": round(avg_overall, 3)
            },
            "emotion_distribution": emotion_counts,
            "chunks_per_minute": round(self.chunk_count / (session_duration / 60), 2)
        }
    
    def export_performance_log(self, filename: str = None) -> str:
        """Export performance log to JSONL file"""
        if not filename:
            filename = f"performance_log_{self.session_id}.jsonl"
        
        try:
            with open(filename, 'w') as f:
                for entry in self.performance_history:
                    f.write(json.dumps(entry) + '\n')
            
            logger.info(f"Performance log exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export performance log: {e}")
            return ""

def test_confidence_classifier():
    """Test function for enhanced confidence classifier"""
    print("Testing Enhanced Confidence Classifier...")
    
    try:
        classifier = ConfidenceClassifier(use_advanced_sentiment=True)
        print("✓ Enhanced confidence classifier initialized")
        
        # Test with sample data
        test_transcript = "I am very confident in my abilities and I believe I can excel in this role."
        
        # Mock facial metrics
        facial_metrics = FacialMetrics(
            smile_intensity=0.8,
            frown_intensity=0.1,
            eyebrow_raise=0.2,
            eye_contact=0.9,
            head_tilt=0.0,
            head_nod=0.1,
            blink_rate=0.3,
            mouth_open=0.2
        )
        
        # Mock speech metrics
        speech_metrics = SpeechMetrics(
            volume=0.8,
            pace=0.7,
            clarity=0.9,
            tone=0.8,
            pause_frequency=0.2,
            pitch_variation=0.7
        )
        
        # Analyze chunk
        result = classifier.analyze_chunk(
            transcript=test_transcript,
            facial_metrics=facial_metrics,
            speech_metrics=speech_metrics,
            chunk_duration=5.0
        )
        
        print(f"\nAnalysis Result:")
        print(f"Confidence: {result.confidence_level.value} ({result.confidence_score:.2f})")
        print(f"Relevance: {result.relevance_level.value} ({result.relevance_score:.2f})")
        print(f"Sentiment: {result.sentiment_score:.2f}")
        print(f"Emotion: {result.emotion_label}")
        print(f"Toxicity: {result.toxicity_score:.2f}")
        print(f"Politeness: {result.politeness_score:.2f}")
        print(f"Overall Score: {result.overall_score:.2f}")
        
        # Get session summary
        summary = classifier.get_session_summary()
        print(f"\nSession Summary:")
        print(f"Total Chunks: {summary['total_chunks']}")
        print(f"Avg Confidence: {summary['averages']['confidence']}")
        print(f"Avg Relevance: {summary['averages']['relevance']}")
        
        print("\nEnhanced confidence classifier is ready for use!")
        
    except Exception as e:
        print(f"❌ Enhanced confidence classifier test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_confidence_classifier()
