"""
Enhanced NLP-Integrated CRM System
Comprehensive text analysis with sentiment, entities, summarization, and priority scoring
"""

import spacy
from textblob import TextBlob
from transformers import pipeline
# Optional language detection (avoid hard dependency)
try:
    from langdetect import detect as _detect, DetectorFactory as _DetectorFactory
    _langdetect_available = True
except Exception:
    _langdetect_available = False
    _detect = None
    _DetectorFactory = None
# Optional translation (avoid breaking httpx/httpcore stack)
try:
    from googletrans import Translator as _GoogleTranslator
    _googletrans_available = True
except Exception:
    _googletrans_available = False
    _GoogleTranslator = None
import re
import warnings
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import logging

# Set seed for consistent language detection if available
if _langdetect_available and _DetectorFactory is not None:
    try:
        _DetectorFactory.seed = 0
    except Exception:
        pass
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class EnhancedNLPCRM:
    def __init__(self):
        """Initialize the Enhanced NLP CRM System"""
        logger.info("Initializing Enhanced NLP-Integrated CRM System...")
        logger.info("Loading models and components...")

        try:
            # Load models with error handling
            self.nlp_spacy = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded")

            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # Use CPU to avoid GPU issues
            )
            logger.info("BART summarization model loaded")

            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            logger.info("Advanced sentiment analyzer loaded")

            # Initialize translator only if library is available
            self.translator = _GoogleTranslator() if _googletrans_available else None
            if self.translator is not None:
                logger.info("Google Translator initialized")
            else:
                logger.info("Google Translator not available; skipping translation")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

        logger.info("All models loaded successfully!")

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text"""
        if not _langdetect_available or _detect is None:
            return 'en'
        try:
            return _detect(text)
        except Exception:
            return 'en'  # Default to English if detection fails

    def translate_text(self, text: str, target_lang: str = 'en') -> Tuple[str, str]:
        """Translate text to target language"""
        detected_lang = self.detect_language(text)
        # If already target or translator unavailable, return original
        if detected_lang == target_lang or not _googletrans_available or self.translator is None:
            return text, detected_lang
        try:
            translated = self.translator.translate(text, dest=target_lang).text
            return translated, detected_lang
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text, detected_lang or 'unknown'

    def enhanced_sentiment_analysis(self, text: str) -> Dict:
        """Perform comprehensive sentiment analysis"""
        try:
            # TextBlob sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Advanced transformer-based sentiment
            advanced_sentiment = self.sentiment_analyzer(text)[0]

            # Normalize advanced label
            adv_label = str(advanced_sentiment.get('label', '')).lower()
            adv_score = float(advanced_sentiment.get('score', 0.0))

            # Determine overall sentiment
            if polarity > 0.1:
                overall_sentiment = "Positive"
                emoji = "😊"
            elif polarity < -0.1:
                overall_sentiment = "Negative"
                emoji = "😟"
            else:
                overall_sentiment = "Neutral"
                emoji = "😐"

            # If transformer strongly contradicts TextBlob around neutral, trust transformer
            if overall_sentiment == "Neutral":
                if adv_label.startswith('neg') and adv_score >= 0.6:
                    overall_sentiment = "Negative"
                    emoji = "😟"
                elif adv_label.startswith('pos') and adv_score >= 0.6:
                    overall_sentiment = "Positive"
                    emoji = "😊"

            return {
                'polarity': round(polarity, 3),
                'subjectivity': round(subjectivity, 3),
                'overall_sentiment': overall_sentiment,
                'emoji': emoji,
                'advanced_sentiment': advanced_sentiment['label'],
                'confidence': round(adv_score, 3),
                'tone': 'Opinionated' if subjectivity > 0.5 else 'Factual'
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {'error': str(e)}

    def extract_entities(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract named entities with enhanced categorization"""
        try:
            doc = self.nlp_spacy(text)
            entities = []

            for ent in doc.ents:
                # Enhanced entity descriptions
                entity_descriptions = {
                    'PERSON': 'Person Name',
                    'ORG': 'Organization',
                    'GPE': 'Location',
                    'MONEY': 'Monetary Value',
                    'DATE': 'Date/Time',
                    'PRODUCT': 'Product',
                    'EMAIL': 'Email Address',
                    'PHONE': 'Phone Number',
                    'CARDINAL': 'Number',
                    'ORDINAL': 'Ordinal Number'
                }

                description = entity_descriptions.get(ent.label_, ent.label_)
                entities.append((ent.text.strip(), ent.label_, description))

            # Extract email addresses and phone numbers using regex
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'

            emails = re.findall(email_pattern, text)
            phones = re.findall(phone_pattern, text)

            for email in emails:
                entities.append((email, 'EMAIL', 'Email Address'))

            for phone in phones:
                entities.append((phone, 'PHONE', 'Phone Number'))

            return entities

        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return []

    def extract_keywords(self, text: str) -> List[str]:
        """Extract enhanced keywords and key phrases"""
        try:
            doc = self.nlp_spacy(text)

            # Extract noun chunks
            noun_chunks = [chunk.text.strip().lower() for chunk in doc.noun_chunks
                          if len(chunk.text.strip()) > 2]

            # Extract important single words (nouns, adjectives)
            important_words = [token.lemma_.lower() for token in doc
                             if token.pos_ in ['NOUN', 'ADJ']
                             and not token.is_stop
                             and not token.is_punct
                             and len(token.text) > 2]

            # Combine and deduplicate
            keywords = list(set(noun_chunks + important_words))

            # Sort by frequency and relevance
            return sorted(keywords, key=len, reverse=True)[:10]

        except Exception as e:
            logger.error(f"Keyword extraction error: {e}")
            return []

    def generate_summary(self, text: str) -> str:
        """Generate intelligent summary based on text length"""
        try:
            word_count = len(text.split())

            if word_count < 20:
                return "Text too short for meaningful summarization."

            elif word_count < 50:
                # For short texts, create a simple summary
                doc = self.nlp_spacy(text)
                sentences = [sent.text.strip() for sent in doc.sents]
                return sentences[0] if sentences else "Unable to generate summary."

            else:
                # Chunk long text to avoid model truncation and aggregate summaries
                def chunk_tokens(s: str, max_words: int = 350) -> List[str]:
                    words = s.split()
                    chunks: List[str] = []
                    for i in range(0, len(words), max_words):
                        chunks.append(" ".join(words[i:i+max_words]))
                    return chunks

                chunks = chunk_tokens(text, max_words=350)
                partial_summaries: List[str] = []
                for chunk in chunks:
                    c_wc = len(chunk.split())
                    max_length = min(130, max(60, c_wc // 2))
                    min_length = min(60, max(25, c_wc // 4))
                    out = self.summarizer(
                        chunk,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )[0]['summary_text']
                    partial_summaries.append(out)

                # If multiple chunk summaries, summarize the summaries for a final concise result
                if len(partial_summaries) == 1:
                    return partial_summaries[0]
                else:
                    joined = "\n".join(partial_summaries)
                    jwc = len(joined.split())
                    max_length = min(140, max(70, jwc // 2))
                    min_length = min(70, max(30, jwc // 4))
                    final = self.summarizer(
                        joined,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )[0]['summary_text']
                    return final

        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return "Unable to generate summary due to processing error."

    def classify_intent(self, text: str) -> str:
        """Classify the intent/purpose of the text"""
        text_lower = text.lower()

        # Intent classification based on keywords
        if any(word in text_lower for word in ['complaint', 'issue', 'problem', 'wrong', 'error', 'fix']):
            return "🚨 Complaint/Issue"
        elif any(word in text_lower for word in ['thank', 'appreciate', 'grateful', 'excellent', 'great']):
            return "💝 Appreciation/Praise"
        elif any(word in text_lower for word in ['question', 'help', 'how', 'what', 'when', 'where', 'support']):
            return "❓ Inquiry/Support"
        elif any(word in text_lower for word in ['order', 'buy', 'purchase', 'payment', 'invoice']):
            return "🛍️ Sales/Order"
        elif any(word in text_lower for word in ['refund', 'return', 'cancel', 'money back']):
            return "💰 Refund/Return"
        else:
            return "📧 General Communication"

    def calculate_priority_score(self, sentiment: Dict, entities: List, text: str) -> Tuple[str, int]:
        """Calculate text priority based on various factors"""
        score = 0

        # Sentiment-based scoring
        if sentiment.get('overall_sentiment') == 'Negative':
            score += 3
        elif sentiment.get('overall_sentiment') == 'Positive':
            score += 1

        # Entity-based scoring
        if any('MONEY' in entity[1] for entity in entities):
            score += 2
        if any('DATE' in entity[1] for entity in entities):
            score += 1

        tl = text.lower()

        # Urgency keywords
        urgency_words = ['urgent', 'asap', 'immediately', 'emergency', 'critical', 'important']
        if any(word in tl for word in urgency_words):
            score += 3

        # Escalation / unacceptable signal
        if 'escalate' in tl or 'escalation' in tl:
            score += 2
        if 'unacceptable' in tl:
            score += 2

        # Failed fulfillment (damage/refund/replacement)
        failed_terms = ['damaged', 'broken', 'defective', 'refund', 'replacement', 'not received', 'never received']
        if any(term in tl for term in failed_terms):
            score += 2

        # Multiple attempts signal
        multiple_attempts = ['multiple emails', 'several emails', 'many emails', 'multiple times', 'several times']
        if any(term in tl for term in multiple_attempts):
            score += 1

        # Determine priority level
        if score >= 6:
            return "🔴 HIGH", score
        elif score >= 3:
            return "🟡 MEDIUM", score
        else:
            return "🟢 LOW", score

    def analyze_text(self, text: str) -> Dict:
        """Comprehensive text analysis"""
        logger.info("Starting comprehensive text analysis...")

        # Language detection and translation
        translated_text, detected_lang = self.translate_text(text)

        # Perform all analyses
        sentiment = self.enhanced_sentiment_analysis(translated_text)
        entities = self.extract_entities(translated_text)
        keywords = self.extract_keywords(translated_text)
        summary = self.generate_summary(translated_text)
        intent = self.classify_intent(translated_text)
        priority, priority_score = self.calculate_priority_score(sentiment, entities, translated_text)

        return {
            'original_text': text,
            'translated_text': translated_text if detected_lang != 'en' else None,
            'detected_language': detected_lang,
            'summary': summary,
            'sentiment': sentiment,
            'entities': entities,
            'keywords': keywords,
            'intent': intent,
            'priority': priority,
            'priority_score': priority_score,
            'word_count': len(translated_text.split()),
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# Global instance
enhanced_nlp_crm = None

def get_enhanced_nlp_crm():
    """Get or create the enhanced NLP CRM instance"""
    global enhanced_nlp_crm
    if enhanced_nlp_crm is None:
        enhanced_nlp_crm = EnhancedNLPCRM()
    return enhanced_nlp_crm

