"""
Advanced Sentiment Analyzer Module
Uses Transformers models for advanced sentiment, emotion, toxicity, and politeness analysis
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

# Advanced ML imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Using fallback methods.")

# Fallback imports
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionType(Enum):
    """Emotion types for classification"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"

class ToxicityLevel(Enum):
    """Toxicity levels"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class PolitenessLevel(Enum):
    """Politeness levels"""
    VERY_IMPOLITE = "very_impolite"
    IMPOLITE = "impolite"
    NEUTRAL = "neutral"
    POLITE = "polite"
    VERY_POLITE = "very_polite"

@dataclass
class AdvancedSentimentResult:
    """Structured result for advanced sentiment analysis"""
    # Basic sentiment
    overall_sentiment: float  # -1 to 1
    
    # Advanced analysis
    emotions: Dict[str, float]  # emotion -> confidence
    primary_emotion: EmotionType
    toxicity_score: float  # 0 to 1
    toxicity_level: ToxicityLevel
    politeness_score: float  # 0 to 1
    politeness_level: PolitenessLevel
    
    # Confidence indicators
    confidence_keywords: List[str]
    filler_word_count: int
    hedge_indicators: List[str]
    
    # Text analysis
    word_count: int
    sentence_count: int
    complexity_score: float
    
    # Raw scores
    vader_scores: Dict[str, float]
    textblob_scores: Dict[str, float]

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analyzer using Transformers models"""
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize advanced sentiment analyzer
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Initialize models
        self._initialize_models()
        
        # Confidence-related keywords
        self.confidence_keywords = {
            'positive': [
                'confident', 'sure', 'certain', 'definitely', 'absolutely',
                'excellent', 'great', 'strong', 'solid', 'proven',
                'successful', 'achieved', 'accomplished', 'demonstrated',
                'expertise', 'experience', 'qualified', 'capable'
            ],
            'negative': [
                'unsure', 'uncertain', 'maybe', 'perhaps', 'possibly',
                'weak', 'poor', 'struggled', 'failed', 'difficult',
                'challenging', 'problematic', 'concerning', 'worried',
                'doubt', 'hesitant', 'unclear', 'confused'
            ]
        }
        
        # Filler words and hedge indicators
        self.filler_words = [
            'um', 'uh', 'ah', 'er', 'like', 'you know', 'sort of',
            'kind of', 'basically', 'actually', 'literally', 'well',
            'i mean', 'right', 'okay', 'so', 'and stuff'
        ]
        
        self.hedge_indicators = [
            'maybe', 'perhaps', 'possibly', 'might', 'could',
            'sort of', 'kind of', 'basically', 'generally',
            'i think', 'i believe', 'it seems', 'appears to',
            'roughly', 'approximately', 'more or less'
        ]
        
        logger.info(f"Advanced Sentiment Analyzer initialized (device: {self.device})")
    
    def _initialize_models(self):
        """Initialize all required models"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("Transformers not available. Using fallback methods.")
                self._initialize_fallback_models()
                return
            
            logger.info("Initializing Transformers models...")
            
            # 1. Emotion classification model
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if self.use_gpu else -1
            )
            logger.info("✓ Emotion classifier loaded")
            
            # 2. Toxicity detection model
            self.toxicity_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=0 if self.use_gpu else -1
            )
            logger.info("✓ Toxicity classifier loaded")
            
            # 3. Sentiment analysis model
            self.sentiment_classifier = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.use_gpu else -1
            )
            logger.info("✓ Sentiment classifier loaded")
            
            # 4. Sentence embeddings for politeness analysis
            self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✓ Sentence encoder loaded")
            
            # Initialize fallback models as backup
            self._initialize_fallback_models()
            
        except Exception as e:
            logger.error(f"Failed to initialize Transformers models: {e}")
            logger.info("Falling back to basic models")
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Initialize fallback models when Transformers are not available"""
        try:
            # Download required NLTK data
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            
            self.vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("Fallback models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize fallback models: {e}")
    
    def analyze_sentiment(self, text: str) -> AdvancedSentimentResult:
        """
        Perform comprehensive sentiment analysis
        
        Args:
            text: Text to analyze
            
        Returns:
            AdvancedSentimentResult with all analysis data
        """
        if not text or not text.strip():
            return self._get_neutral_result()
        
        try:
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            
            # Basic text analysis
            text_analysis = self._analyze_text_structure(cleaned_text)
            
            # Advanced sentiment analysis
            if TRANSFORMERS_AVAILABLE and hasattr(self, 'sentiment_classifier'):
                sentiment_result = self._analyze_with_transformers(cleaned_text)
            else:
                sentiment_result = self._analyze_with_fallback(cleaned_text)
            
            # Emotion analysis
            if TRANSFORMERS_AVAILABLE and hasattr(self, 'emotion_classifier'):
                emotions = self._analyze_emotions(cleaned_text)
            else:
                emotions = self._analyze_emotions_fallback(cleaned_text)
            
            # Toxicity analysis
            if TRANSFORMERS_AVAILABLE and hasattr(self, 'toxicity_classifier'):
                toxicity_result = self._analyze_toxicity(cleaned_text)
            else:
                toxicity_result = self._analyze_toxicity_fallback(cleaned_text)
            
            # Politeness analysis
            politeness_result = self._analyze_politeness(cleaned_text)
            
            # Confidence indicators
            confidence_indicators = self._analyze_confidence_indicators(cleaned_text)
            
            # Combine all results
            return AdvancedSentimentResult(
                overall_sentiment=sentiment_result['overall_sentiment'],
                emotions=emotions,
                primary_emotion=self._get_primary_emotion(emotions),
                toxicity_score=toxicity_result['score'],
                toxicity_level=toxicity_result['level'],
                politeness_score=politeness_result['score'],
                politeness_level=politeness_result['level'],
                confidence_keywords=confidence_indicators['keywords'],
                filler_word_count=confidence_indicators['filler_count'],
                hedge_indicators=confidence_indicators['hedge_count'],
                word_count=text_analysis['word_count'],
                sentence_count=text_analysis['sentence_count'],
                complexity_score=text_analysis['complexity_score'],
                vader_scores=sentiment_result['vader_scores'],
                textblob_scores=sentiment_result['textblob_scores']
            )
            
        except Exception as e:
            logger.error(f"Advanced sentiment analysis failed: {str(e)}")
            return self._get_neutral_result()
    
    def _analyze_with_transformers(self, text: str) -> Dict:
        """Analyze sentiment using Transformers models"""
        try:
            # Get sentiment classification
            sentiment_result = self.sentiment_classifier(text)
            
            # Map sentiment labels to scores
            sentiment_mapping = {
                'LABEL_0': -1.0,  # Negative
                'LABEL_1': 0.0,   # Neutral
                'LABEL_2': 1.0    # Positive
            }
            
            overall_sentiment = sentiment_mapping.get(sentiment_result[0]['label'], 0.0)
            
            # Get VADER scores as backup
            vader_scores = self._get_vader_sentiment(text)
            textblob_scores = self._get_textblob_sentiment(text)
            
            return {
                'overall_sentiment': overall_sentiment,
                'vader_scores': vader_scores,
                'textblob_scores': textblob_scores,
                'transformers_confidence': sentiment_result[0]['score']
            }
            
        except Exception as e:
            logger.warning(f"Transformers sentiment analysis failed: {e}")
            return self._analyze_with_fallback(text)
    
    def _analyze_with_fallback(self, text: str) -> Dict:
        """Analyze sentiment using fallback methods"""
        vader_scores = self._get_vader_sentiment(text)
        textblob_scores = self._get_textblob_sentiment(text)
        
        # Calculate overall sentiment
        overall_sentiment = self._calculate_overall_sentiment_fallback(vader_scores, textblob_scores)
        
        return {
            'overall_sentiment': overall_sentiment,
            'vader_scores': vader_scores,
            'textblob_scores': textblob_scores
        }
    
    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Analyze emotions using Transformers model"""
        try:
            if not hasattr(self, 'emotion_classifier'):
                return self._analyze_emotions_fallback(text)
            
            # Get emotion classification
            emotion_result = self.emotion_classifier(text)
            
            # Convert to emotion -> confidence mapping
            emotions = {}
            for result in emotion_result:
                emotion_label = result['label'].lower()
                confidence = result['score']
                emotions[emotion_label] = confidence
            
            return emotions
            
        except Exception as e:
            logger.warning(f"Transformers emotion analysis failed: {e}")
            return self._analyze_emotions_fallback(text)
    
    def _analyze_emotions_fallback(self, text: str) -> Dict[str, float]:
        """Fallback emotion analysis using keyword matching"""
        emotions = {
            'joy': 0.0, 'sadness': 0.0, 'anger': 0.0,
            'fear': 0.0, 'surprise': 0.0, 'disgust': 0.0, 'neutral': 0.0
        }
        
        # Simple keyword-based emotion detection
        joy_words = ['happy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic']
        sadness_words = ['sad', 'disappointed', 'unfortunate', 'sorry', 'regret']
        anger_words = ['angry', 'frustrated', 'annoyed', 'upset', 'mad']
        fear_words = ['worried', 'concerned', 'afraid', 'scared', 'nervous']
        surprise_words = ['surprised', 'shocked', 'amazed', 'unexpected']
        
        text_lower = text.lower()
        
        for word in joy_words:
            if word in text_lower:
                emotions['joy'] += 0.2
        
        for word in sadness_words:
            if word in text_lower:
                emotions['sadness'] += 0.2
        
        for word in anger_words:
            if word in text_lower:
                emotions['anger'] += 0.2
        
        for word in fear_words:
            if word in text_lower:
                emotions['fear'] += 0.2
        
        for word in surprise_words:
            if word in text_lower:
                emotions['surprise'] += 0.2
        
        # Normalize scores
        total = sum(emotions.values())
        if total > 0:
            for emotion in emotions:
                emotions[emotion] = min(emotions[emotion] / total, 1.0)
        else:
            emotions['neutral'] = 1.0
        
        return emotions
    
    def _analyze_toxicity(self, text: str) -> Dict:
        """Analyze toxicity using Transformers model"""
        try:
            if not hasattr(self, 'toxicity_classifier'):
                return self._analyze_toxicity_fallback(text)
            
            # Get toxicity classification
            toxicity_result = self.toxicity_classifier(text)
            
            # Calculate overall toxicity score
            toxicity_score = 0.0
            for result in toxicity_result:
                if result['label'] in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
                    toxicity_score = max(toxicity_score, result['score'])
            
            # Determine toxicity level
            if toxicity_score < 0.1:
                level = ToxicityLevel.SAFE
            elif toxicity_score < 0.3:
                level = ToxicityLevel.LOW
            elif toxicity_score < 0.6:
                level = ToxicityLevel.MEDIUM
            elif toxicity_score < 0.8:
                level = ToxicityLevel.HIGH
            else:
                level = ToxicityLevel.EXTREME
            
            return {
                'score': toxicity_score,
                'level': level,
                'details': toxicity_result
            }
            
        except Exception as e:
            logger.warning(f"Transformers toxicity analysis failed: {e}")
            return self._analyze_toxicity_fallback(text)
    
    def _analyze_toxicity_fallback(self, text: str) -> Dict:
        """Fallback toxicity analysis using keyword matching"""
        toxic_words = [
            'hate', 'stupid', 'idiot', 'dumb', 'terrible', 'awful',
            'horrible', 'disgusting', 'offensive', 'inappropriate'
        ]
        
        text_lower = text.lower()
        toxic_count = sum(1 for word in toxic_words if word in text_lower)
        
        # Simple scoring based on toxic word count
        toxicity_score = min(toxic_count * 0.1, 1.0)
        
        if toxicity_score < 0.1:
            level = ToxicityLevel.SAFE
        elif toxicity_score < 0.3:
            level = ToxicityLevel.LOW
        elif toxicity_score < 0.6:
            level = ToxicityLevel.MEDIUM
        elif toxicity_score < 0.8:
            level = ToxicityLevel.HIGH
        else:
            level = ToxicityLevel.EXTREME
        
        return {
            'score': toxicity_score,
            'level': level,
            'details': {'toxic_word_count': toxic_count}
        }
    
    def _analyze_politeness(self, text: str) -> Dict:
        """Analyze politeness using sentence embeddings and rules"""
        try:
            # Simple rule-based politeness analysis
            polite_indicators = [
                'please', 'thank you', 'thanks', 'excuse me', 'sorry',
                'would you', 'could you', 'may i', 'if you don\'t mind'
            ]
            
            impolite_indicators = [
                'shut up', 'get lost', 'go away', 'i don\'t care',
                'whatever', 'who cares', 'nobody cares'
            ]
            
            text_lower = text.lower()
            
            polite_count = sum(1 for phrase in polite_indicators if phrase in text_lower)
            impolite_count = sum(1 for phrase in impolite_indicators if phrase in text_lower)
            
            # Calculate politeness score
            if polite_count == 0 and impolite_count == 0:
                politeness_score = 0.5  # Neutral
            else:
                total = polite_count + impolite_count
                politeness_score = polite_count / total
            
            # Determine politeness level
            if politeness_score < 0.2:
                level = PolitenessLevel.VERY_IMPOLITE
            elif politeness_score < 0.4:
                level = PolitenessLevel.IMPOLITE
            elif politeness_score < 0.6:
                level = PolitenessLevel.NEUTRAL
            elif politeness_score < 0.8:
                level = PolitenessLevel.POLITE
            else:
                level = PolitenessLevel.VERY_POLITE
            
            return {
                'score': politeness_score,
                'level': level,
                'polite_count': polite_count,
                'impolite_count': impolite_count
            }
            
        except Exception as e:
            logger.error(f"Politeness analysis failed: {e}")
            return {
                'score': 0.5,
                'level': PolitenessLevel.NEUTRAL,
                'polite_count': 0,
                'impolite_count': 0
            }
    
    def _analyze_confidence_indicators(self, text: str) -> Dict:
        """Analyze confidence-related indicators in text"""
        words = text.split()
        
        # Count confidence keywords
        positive_count = 0
        negative_count = 0
        found_keywords = []
        
        for word in words:
            if word in self.confidence_keywords['positive']:
                positive_count += 1
                found_keywords.append(word)
            elif word in self.confidence_keywords['negative']:
                negative_count += 1
                found_keywords.append(word)
        
        # Count filler words
        filler_count = sum(1 for word in words if word in self.filler_words)
        
        # Count hedge indicators
        hedge_count = sum(1 for word in words if word in self.hedge_indicators)
        
        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'keywords': found_keywords,
            'filler_count': filler_count,
            'hedge_count': hedge_count
        }
    
    def _analyze_text_structure(self, text: str) -> Dict:
        """Analyze basic text structure"""
        sentences = text.split('.')
        words = text.split()
        
        # Calculate complexity score (simple heuristic)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        complexity_score = min(avg_word_length / 10.0, 1.0)  # Normalize to 0-1
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': avg_word_length,
            'complexity_score': complexity_score
        }
    
    def _get_primary_emotion(self, emotions: Dict[str, float]) -> EmotionType:
        """Get the primary emotion from emotion scores"""
        if not emotions:
            return EmotionType.NEUTRAL
        
        primary_emotion = max(emotions.items(), key=lambda x: x[1])
        
        # Map to EmotionType enum
        emotion_mapping = {
            'joy': EmotionType.JOY,
            'sadness': EmotionType.SADNESS,
            'anger': EmotionType.ANGER,
            'fear': EmotionType.FEAR,
            'surprise': EmotionType.SURPRISE,
            'disgust': EmotionType.DISGUST,
            'neutral': EmotionType.NEUTRAL
        }
        
        return emotion_mapping.get(primary_emotion[0], EmotionType.NEUTRAL)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation (keep basic structure)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return text.strip()
    
    def _get_vader_sentiment(self, text: str) -> Dict:
        """Get VADER sentiment scores"""
        if not hasattr(self, 'vader_analyzer'):
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 0.0}
        
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return scores
        except Exception as e:
            logger.warning(f"VADER analysis failed: {e}")
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 0.0}
    
    def _get_textblob_sentiment(self, text: str) -> Dict:
        """Get TextBlob sentiment scores"""
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,  # -1 to 1
                'subjectivity': blob.sentiment.subjectivity  # 0 to 1
            }
        except Exception as e:
            logger.warning(f"TextBlob analysis failed: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.5}
    
    def _calculate_overall_sentiment_fallback(self, vader_scores: Dict, textblob_sentiment: Dict) -> float:
        """Calculate overall sentiment score using fallback methods"""
        # VADER compound score (-1 to 1)
        vader_score = vader_scores.get('compound', 0.0)
        
        # TextBlob polarity (-1 to 1)
        textblob_score = textblob_sentiment.get('polarity', 0.0)
        
        # Weighted combination
        overall_score = (0.6 * vader_score + 0.4 * textblob_score)
        
        # Ensure score is in [-1, 1] range
        return max(-1.0, min(1.0, overall_score))
    
    def _get_neutral_result(self) -> AdvancedSentimentResult:
        """Return neutral result when analysis fails"""
        return AdvancedSentimentResult(
            overall_sentiment=0.0,
            emotions={'neutral': 1.0},
            primary_emotion=EmotionType.NEUTRAL,
            toxicity_score=0.0,
            toxicity_level=ToxicityLevel.SAFE,
            politeness_score=0.5,
            politeness_level=PolitenessLevel.NEUTRAL,
            confidence_keywords=[],
            filler_word_count=0,
            hedge_indicators=0,
            word_count=0,
            sentence_count=0,
            complexity_score=0.5,
            vader_scores={'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0},
            textblob_scores={'polarity': 0.0, 'subjectivity': 0.5}
        )
    
    def get_analysis_summary(self, result: AdvancedSentimentResult) -> str:
        """Get human-readable analysis summary"""
        summary = f"Advanced Sentiment Analysis Summary\n"
        summary += f"{'='*40}\n"
        summary += f"Overall Sentiment: {result.overall_sentiment:.2f}\n"
        summary += f"Primary Emotion: {result.primary_emotion.value}\n"
        summary += f"Toxicity: {result.toxicity_level.value} ({result.toxicity_score:.2f})\n"
        summary += f"Politeness: {result.politeness_level.value} ({result.politeness_score:.2f})\n"
        summary += f"Word Count: {result.word_count}\n"
        summary += f"Filler Words: {result.filler_word_count}\n"
        summary += f"Hedge Indicators: {result.hedge_indicators}\n"
        
        if result.confidence_keywords:
            summary += f"Confidence Keywords: {', '.join(result.confidence_keywords)}\n"
        
        return summary

def test_advanced_sentiment_analyzer():
    """Test function for advanced sentiment analyzer"""
    print("Testing Advanced Sentiment Analyzer...")
    
    try:
        analyzer = AdvancedSentimentAnalyzer()
        print("✓ Advanced sentiment analyzer initialized")
        
        # Test with sample texts
        test_texts = [
            "I am very confident in my abilities and I believe I can excel in this role.",
            "I'm not sure about this, maybe I could try but it might be difficult.",
            "The project was challenging but I successfully completed it with excellent results.",
            "Please, could you help me understand this better? Thank you so much!"
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text[:50]}...")
            result = analyzer.analyze_sentiment(text)
            summary = analyzer.get_analysis_summary(result)
            print(summary)
        
        print("\nAdvanced sentiment analyzer is ready for use!")
        
    except Exception as e:
        print(f"❌ Advanced sentiment analyzer test failed: {e}")
        print("This is expected if some models are not fully configured")

if __name__ == "__main__":
    test_advanced_sentiment_analyzer()
