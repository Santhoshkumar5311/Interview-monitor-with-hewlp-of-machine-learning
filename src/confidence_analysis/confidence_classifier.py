"""
Confidence Classifier Module
Combines facial expressions, head pose, and speech characteristics to assess candidate confidence
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from enum import Enum
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence level enumeration"""
    VERY_LOW = "Very Low"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"

@dataclass
class ConfidenceMetrics:
    """Structured confidence metrics"""
    overall_confidence: float
    confidence_level: ConfidenceLevel
    facial_confidence: float
    posture_confidence: float
    speech_confidence: float
    sentiment_score: float
    detailed_breakdown: Dict

class ConfidenceClassifier:
    """Classifies candidate confidence based on multiple factors"""
    
    def __init__(self):
        """Initialize confidence classifier with weights and thresholds"""
        
        # Weights for different confidence factors
        self.weights = {
            'facial_expressions': 0.35,    # 35% weight
            'head_posture': 0.25,          # 25% weight
            'speech_characteristics': 0.25, # 25% weight
            'sentiment': 0.15              # 15% weight
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            ConfidenceLevel.VERY_LOW: 0.0,
            ConfidenceLevel.LOW: 0.2,
            ConfidenceLevel.MEDIUM: 0.4,
            ConfidenceLevel.HIGH: 0.6,
            ConfidenceLevel.VERY_HIGH: 0.8
        }
        
        # Facial expression confidence scoring
        self.facial_scores = {
            'smile': {'positive': 0.8, 'negative': 0.2},
            'frown': {'positive': 0.2, 'negative': 0.8},
            'raised_eyebrows': {'positive': 0.7, 'negative': 0.3},
            'eye_contact': {'positive': 0.9, 'negative': 0.1},
            'neutral': {'positive': 0.5, 'negative': 0.5}
        }
        
        # Head posture confidence scoring
        self.posture_scores = {
            'upright': 0.9,      # Head straight, looking forward
            'slight_tilt': 0.7,  # Small head tilt
            'moderate_tilt': 0.5, # Moderate head tilt
            'significant_tilt': 0.3, # Large head tilt
            'looking_down': 0.2,  # Head down
            'looking_away': 0.1   # Head turned away
        }
        
        # Speech confidence scoring
        self.speech_scores = {
            'clear_articulation': 0.9,
            'steady_pace': 0.8,
            'appropriate_volume': 0.8,
            'minimal_fillers': 0.7,
            'consistent_tone': 0.8
        }
        
        logger.info("Confidence Classifier initialized")
    
    def classify_confidence(self, 
                          facial_data: Dict,
                          head_pose: Dict,
                          speech_data: Dict,
                          sentiment_score: float) -> ConfidenceMetrics:
        """
        Classify overall confidence based on all available data
        
        Args:
            facial_data: Facial expression analysis results
            head_pose: Head pose data (pitch, yaw, roll)
            speech_data: Speech characteristics data
            sentiment_score: Sentiment analysis score (-1 to 1)
            
        Returns:
            ConfidenceMetrics object with detailed confidence assessment
        """
        try:
            # Calculate individual confidence scores
            facial_confidence = self._calculate_facial_confidence(facial_data)
            posture_confidence = self._calculate_posture_confidence(head_pose)
            speech_confidence = self._calculate_speech_confidence(speech_data)
            
            # Normalize sentiment score to 0-1 range
            normalized_sentiment = (sentiment_score + 1) / 2
            
            # Calculate weighted overall confidence
            overall_confidence = (
                self.weights['facial_expressions'] * facial_confidence +
                self.weights['head_posture'] * posture_confidence +
                self.weights['speech_characteristics'] * speech_confidence +
                self.weights['sentiment'] * normalized_sentiment
            )
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(overall_confidence)
            
            # Create detailed breakdown
            detailed_breakdown = {
                'facial_expressions': {
                    'score': facial_confidence,
                    'weight': self.weights['facial_expressions'],
                    'contribution': facial_confidence * self.weights['facial_expressions']
                },
                'head_posture': {
                    'score': posture_confidence,
                    'weight': self.weights['head_posture'],
                    'contribution': posture_confidence * self.weights['head_posture']
                },
                'speech_characteristics': {
                    'score': speech_confidence,
                    'weight': self.weights['speech_characteristics'],
                    'contribution': speech_confidence * self.weights['speech_characteristics']
                },
                'sentiment': {
                    'score': normalized_sentiment,
                    'weight': self.weights['sentiment'],
                    'contribution': normalized_sentiment * self.weights['sentiment']
                }
            }
            
            return ConfidenceMetrics(
                overall_confidence=overall_confidence,
                confidence_level=confidence_level,
                facial_confidence=facial_confidence,
                posture_confidence=posture_confidence,
                speech_confidence=speech_confidence,
                sentiment_score=sentiment_score,
                detailed_breakdown=detailed_breakdown
            )
            
        except Exception as e:
            logger.error(f"Confidence classification failed: {str(e)}")
            # Return neutral confidence on error
            return ConfidenceMetrics(
                overall_confidence=0.5,
                confidence_level=ConfidenceLevel.MEDIUM,
                facial_confidence=0.5,
                posture_confidence=0.5,
                speech_confidence=0.5,
                sentiment_score=0.0,
                detailed_breakdown={}
            )
    
    def _calculate_facial_confidence(self, facial_data: Dict) -> float:
        """Calculate confidence score from facial expressions"""
        if not facial_data:
            return 0.5  # Neutral score
        
        total_score = 0.0
        expression_count = 0
        
        # Analyze each detected expression
        for expression_name, expression_data in facial_data.items():
            if expression_data.get('detected', False):
                confidence = expression_data.get('confidence', 0.0)
                
                # Get base score for this expression
                if expression_name in self.facial_scores:
                    base_score = self.facial_scores[expression_name]['positive']
                    # Weight by confidence
                    weighted_score = base_score * confidence
                    total_score += weighted_score
                    expression_count += 1
        
        # If no expressions detected, return neutral score
        if expression_count == 0:
            return 0.5
        
        # Return average score
        return total_score / expression_count
    
    def _calculate_posture_confidence(self, head_pose: Dict) -> float:
        """Calculate confidence score from head posture"""
        if not head_pose:
            return 0.5  # Neutral score
        
        # Extract pose angles
        pitch = abs(head_pose.get('pitch', 0))
        yaw = abs(head_pose.get('yaw', 0))
        roll = abs(head_pose.get('roll', 0))
        
        # Score based on head position
        # Ideal: head straight (low angles)
        # Poor: head tilted or turned (high angles)
        
        # Normalize angles to 0-1 scale (0 = ideal, 1 = poor)
        max_angle = 45.0  # degrees
        
        pitch_score = 1.0 - min(pitch / max_angle, 1.0)
        yaw_score = 1.0 - min(yaw / max_angle, 1.0)
        roll_score = 1.0 - min(roll / max_angle, 1.0)
        
        # Weight the scores (pitch and yaw more important than roll)
        posture_score = (0.4 * pitch_score + 0.4 * yaw_score + 0.2 * roll_score)
        
        return posture_score
    
    def _calculate_speech_confidence(self, speech_data: Dict) -> float:
        """Calculate confidence score from speech characteristics"""
        if not speech_data:
            return 0.5  # Neutral score
        
        total_score = 0.0
        factor_count = 0
        
        # Analyze speech factors
        for factor_name, factor_score in speech_data.items():
            if factor_name in self.speech_scores:
                # Normalize factor score to 0-1 if needed
                normalized_score = min(max(factor_score, 0.0), 1.0)
                total_score += normalized_score
                factor_count += 1
        
        # If no speech factors, return neutral score
        if factor_count == 0:
            return 0.5
        
        # Return average score
        return total_score / factor_count
    
    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determine confidence level from numerical score"""
        # Find the highest threshold that the score meets
        for level in reversed(list(ConfidenceLevel)):
            if confidence_score >= self.confidence_thresholds[level]:
                return level
        
        return ConfidenceLevel.VERY_LOW
    
    def get_confidence_summary(self, metrics: ConfidenceMetrics) -> str:
        """Get human-readable confidence summary"""
        level = metrics.confidence_level.value
        overall = f"{metrics.overall_confidence:.1%}"
        
        summary = f"Confidence Level: {level} ({overall})\n"
        summary += f"Facial: {metrics.facial_confidence:.1%}\n"
        summary += f"Posture: {metrics.posture_confidence:.1%}\n"
        summary += f"Speech: {metrics.speech_confidence:.1%}\n"
        summary += f"Sentiment: {metrics.sentiment_score:.2f}"
        
        return summary
    
    def get_recommendations(self, metrics: ConfidenceMetrics) -> List[str]:
        """Get improvement recommendations based on confidence analysis"""
        recommendations = []
        
        # Facial expression recommendations
        if metrics.facial_confidence < 0.6:
            recommendations.append("Try to maintain positive facial expressions")
            recommendations.append("Practice confident eye contact")
        
        # Posture recommendations
        if metrics.posture_confidence < 0.6:
            recommendations.append("Keep your head upright and facing forward")
            recommendations.append("Avoid excessive head tilting or turning")
        
        # Speech recommendations
        if metrics.speech_confidence < 0.6:
            recommendations.append("Speak clearly and at a steady pace")
            recommendations.append("Maintain appropriate volume and tone")
        
        # Sentiment recommendations
        if metrics.sentiment_score < 0.0:
            recommendations.append("Try to maintain a positive tone")
            recommendations.append("Focus on confident and optimistic language")
        
        if not recommendations:
            recommendations.append("Excellent confidence! Keep up the great work!")
        
        return recommendations

def test_confidence_classifier():
    """Test function for confidence classifier"""
    print("Testing confidence classifier...")
    
    classifier = ConfidenceClassifier()
    print("✓ Confidence classifier initialized")
    
    # Test with sample data
    facial_data = {
        'smile': {'detected': True, 'confidence': 0.8},
        'raised_eyebrows': {'detected': False, 'confidence': 0.0}
    }
    
    head_pose = {
        'pitch': 5.0,  # Small forward tilt
        'yaw': 2.0,    # Slight turn
        'roll': 1.0    # Minimal tilt
    }
    
    speech_data = {
        'clear_articulation': 0.8,
        'steady_pace': 0.7,
        'appropriate_volume': 0.9
    }
    
    sentiment_score = 0.3  # Slightly positive
    
    # Classify confidence
    metrics = classifier.classify_confidence(facial_data, head_pose, speech_data, sentiment_score)
    
    print(f"✓ Overall confidence: {metrics.overall_confidence:.1%}")
    print(f"✓ Confidence level: {metrics.confidence_level.value}")
    print(f"✓ Facial confidence: {metrics.facial_confidence:.1%}")
    print(f"✓ Posture confidence: {metrics.posture_confidence:.1%}")
    print(f"✓ Speech confidence: {metrics.speech_confidence:.1%}")
    
    # Get recommendations
    recommendations = classifier.get_recommendations(metrics)
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"  • {rec}")
    
    print("\nConfidence classifier is ready for use!")

if __name__ == "__main__":
    test_confidence_classifier()
