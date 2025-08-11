"""
Sentiment Analyzer Module
Analyzes text sentiment for confidence assessment
"""

import re
from typing import Dict, List, Tuple, Optional
import logging
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyzes sentiment of transcribed speech"""
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        try:
            # Download required NLTK data
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            # Confidence-related keywords
            self.confidence_keywords = {
                'positive': [
                    'confident', 'sure', 'certain', 'definitely', 'absolutely',
                    'excellent', 'great', 'strong', 'solid', 'proven',
                    'successful', 'achieved', 'accomplished', 'demonstrated'
                ],
                'negative': [
                    'unsure', 'uncertain', 'maybe', 'perhaps', 'possibly',
                    'weak', 'poor', 'struggled', 'failed', 'difficult',
                    'challenging', 'problematic', 'concerning', 'worried'
                ]
            }
            
            # Filler words that indicate uncertainty
            self.filler_words = [
                'um', 'uh', 'ah', 'er', 'like', 'you know', 'sort of',
                'kind of', 'basically', 'actually', 'literally'
            ]
            
            logger.info("Sentiment Analyzer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {e}")
            self.vader_analyzer = None
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of given text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict containing sentiment scores and analysis
        """
        if not text or not text.strip():
            return self._get_neutral_sentiment()
        
        try:
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Get VADER sentiment scores
            vader_scores = self._get_vader_sentiment(cleaned_text)
            
            # Get TextBlob sentiment
            textblob_sentiment = self._get_textblob_sentiment(cleaned_text)
            
            # Analyze confidence indicators
            confidence_indicators = self._analyze_confidence_indicators(cleaned_text)
            
            # Calculate overall sentiment score
            overall_sentiment = self._calculate_overall_sentiment(
                vader_scores, textblob_sentiment, confidence_indicators
            )
            
            return {
                'overall_sentiment': overall_sentiment,
                'vader_scores': vader_scores,
                'textblob_sentiment': textblob_sentiment,
                'confidence_indicators': confidence_indicators,
                'text_analysis': {
                    'word_count': len(cleaned_text.split()),
                    'filler_word_count': self._count_filler_words(cleaned_text),
                    'confidence_keywords': self._extract_confidence_keywords(cleaned_text)
                }
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return self._get_neutral_sentiment()
    
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
        if not self.vader_analyzer:
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 0.0}
        
        scores = self.vader_analyzer.polarity_scores(text)
        return scores
    
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
    
    def _analyze_confidence_indicators(self, text: str) -> Dict:
        """Analyze confidence-related indicators in text"""
        words = text.split()
        
        positive_count = 0
        negative_count = 0
        
        # Count confidence keywords
        for word in words:
            if word in self.confidence_keywords['positive']:
                positive_count += 1
            elif word in self.confidence_keywords['negative']:
                negative_count += 1
        
        # Calculate confidence score
        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            confidence_score = 0.0
        else:
            confidence_score = (positive_count - negative_count) / total_keywords
        
        return {
            'positive_keywords': positive_count,
            'negative_keywords': negative_count,
            'confidence_score': confidence_score,
            'total_keywords': total_keywords
        }
    
    def _count_filler_words(self, text: str) -> int:
        """Count filler words in text"""
        words = text.split()
        filler_count = 0
        
        for word in words:
            if word in self.filler_words:
                filler_count += 1
        
        return filler_count
    
    def _extract_confidence_keywords(self, text: str) -> List[str]:
        """Extract confidence-related keywords from text"""
        words = text.split()
        found_keywords = []
        
        for word in words:
            if word in self.confidence_keywords['positive'] or word in self.confidence_keywords['negative']:
                found_keywords.append(word)
        
        return found_keywords
    
    def _calculate_overall_sentiment(self, vader_scores: Dict, textblob_sentiment: Dict, confidence_indicators: Dict) -> float:
        """Calculate overall sentiment score"""
        # VADER compound score (-1 to 1)
        vader_score = vader_scores.get('compound', 0.0)
        
        # TextBlob polarity (-1 to 1)
        textblob_score = textblob_sentiment.get('polarity', 0.0)
        
        # Confidence indicators score (-1 to 1)
        confidence_score = confidence_indicators.get('confidence_score', 0.0)
        
        # Weighted combination
        overall_score = (
            0.4 * vader_score +      # VADER gets highest weight
            0.3 * textblob_score +   # TextBlob gets medium weight
            0.3 * confidence_score   # Confidence indicators get medium weight
        )
        
        # Ensure score is in [-1, 1] range
        return max(-1.0, min(1.0, overall_score))
    
    def _get_neutral_sentiment(self) -> Dict:
        """Return neutral sentiment when analysis fails"""
        return {
            'overall_sentiment': 0.0,
            'vader_scores': {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0},
            'textblob_sentiment': {'polarity': 0.0, 'subjectivity': 0.5},
            'confidence_indicators': {'positive_keywords': 0, 'negative_keywords': 0, 'confidence_score': 0.0, 'total_keywords': 0},
            'text_analysis': {'word_count': 0, 'filler_word_count': 0, 'confidence_keywords': []}
        }
    
    def get_sentiment_summary(self, analysis: Dict) -> str:
        """Get human-readable sentiment summary"""
        sentiment = analysis['overall_sentiment']
        
        if sentiment > 0.3:
            sentiment_label = "Positive"
        elif sentiment < -0.3:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        summary = f"Sentiment: {sentiment_label} ({sentiment:.2f})\n"
        summary += f"VADER: {analysis['vader_scores']['compound']:.2f}\n"
        summary += f"TextBlob: {analysis['textblob_sentiment']['polarity']:.2f}\n"
        summary += f"Confidence Score: {analysis['confidence_indicators']['confidence_score']:.2f}\n"
        summary += f"Word Count: {analysis['text_analysis']['word_count']}\n"
        summary += f"Filler Words: {analysis['text_analysis']['filler_word_count']}"
        
        return summary

def test_sentiment_analyzer():
    """Test function for sentiment analyzer"""
    print("Testing sentiment analyzer...")
    
    analyzer = SentimentAnalyzer()
    print("✓ Sentiment analyzer initialized")
    
    # Test with sample texts
    test_texts = [
        "I am very confident in my abilities and I believe I can excel in this role.",
        "I'm not sure about this, maybe I could try but it might be difficult.",
        "The project was challenging but I successfully completed it with excellent results."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text[:50]}...")
        analysis = analyzer.analyze_sentiment(text)
        summary = analyzer.get_sentiment_summary(analysis)
        print(summary)
    
    print("\nSentiment analyzer is ready for use!")

if __name__ == "__main__":
    test_sentiment_analyzer()
