"""
Query Transformation Module for NLP CRM System
Handles ASR output processing and query enhancement.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import re as _re
try:
    from transformers import pipeline
    _hf_available = True
except Exception:
    _hf_available = False
from datetime import datetime
import re as re2
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryTransformer:
    """Transforms and enhances user queries for better retrieval."""
    
    def __init__(self):
        """Initialize the query transformer."""
        self.common_misspellings = {
            "refund": ["refund", "refunded", "refunding"],
            "shipping": ["shipping", "ship", "delivery", "deliver"],
            "warranty": ["warranty", "guarantee", "guaranteed"],
            "password": ["password", "pass", "login", "signin"],
            "payment": ["payment", "pay", "billing", "charge"],
            "support": ["support", "help", "assistance", "service"],
            "account": ["account", "profile", "user"],
            "order": ["order", "purchase", "buy", "bought"]
        }
        
        self.query_intent_patterns = {
            "refund": [
                r"refund", r"return", r"money back", r"cancel order",
                r"get my money back", r"don't want", r"change my mind"
            ],
            "shipping": [
                r"shipping", r"delivery", r"when will", r"track", r"ship",
                r"free shipping", r"express", r"standard"
            ],
            "warranty": [
                r"warranty", r"guarantee", r"broken", r"defective", r"not working",
                r"repair", r"replace", r"damaged"
            ],
            "password": [
                r"password", r"login", r"sign in", r"forgot", r"reset",
                r"can't access", r"locked out", r"account"
            ],
            "payment": [
                r"payment", r"billing", r"charge", r"credit card", r"pay",
                r"invoice", r"bill", r"cost"
            ],
            "support": [
                r"help", r"support", r"assistance", r"contact", r"speak to",
                r"talk to", r"customer service", r"agent"
            ],
            "greeting": [
                r"hello", r"hi", r"hey", r"good morning", r"good afternoon",
                r"good evening", r"greetings"
            ],
            "goodbye": [
                r"bye", r"goodbye", r"see you", r"thanks", r"thank you",
                r"that's all", r"nothing else"
            ]
        }
        
        self._sentiment_pipe = None
        self._zeroshot_pipe = None
        self._ner_pipe = None
        if _hf_available:
            try:
                self._sentiment_pipe = pipeline("sentiment-analysis")
            except Exception:
                self._sentiment_pipe = None
            try:
                self._zeroshot_pipe = pipeline("zero-shot-classification")
            except Exception:
                self._zeroshot_pipe = None
            try:
                self._ner_pipe = pipeline("ner", grouped_entities=True)
            except Exception:
                self._ner_pipe = None
        logger.info("Query transformer initialized")
        # Optional spaCy NER
        self._spacy = None
        try:
            import spacy  # type: ignore
            try:
                # Try to load small English model if available
                self._spacy = spacy.load("en_core_web_sm")
            except Exception:
                # Fallback to blank English with NER disabled if model missing
                self._spacy = None
        except Exception:
            self._spacy = None
    
    def clean_asr_output(self, asr_text: str) -> str:
        """
        Clean and normalize ASR output.
        
        Args:
            asr_text: Raw ASR transcription
            
        Returns:
            Cleaned text
        """
        if not asr_text:
            return ""
        
        # Convert to lowercase
        cleaned = asr_text.lower().strip()
        
        # Remove common ASR artifacts
        artifacts = [
            r'\b(um|uh|er|ah)\b',  # Filler words
            r'\b(like|you know|basically|literally)\b',  # Common filler phrases
            r'[^\w\s\?\!\.\,]',  # Remove special characters except basic punctuation
            r'\s+',  # Normalize whitespace
        ]
        
        for pattern in artifacts:
            cleaned = re.sub(pattern, ' ', cleaned)
        
        # Clean up punctuation
        cleaned = re.sub(r'\s*([\?\!\.\,])\s*', r'\1 ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def detect_query_intent(self, query: str) -> List[str]:
        """
        Detect the intent(s) of the query.
        
        Args:
            query: User query
            
        Returns:
            List of detected intents
        """
        # Prefer zero-shot classification if available
        if self._zeroshot_pipe:
            try:
                labels = [
                    "refund", "shipping", "warranty", "password",
                    "payment", "support", "greeting", "goodbye", "complaint", "general_inquiry"
                ]
                res = self._zeroshot_pipe(query, candidate_labels=labels, multi_label=True)
                preds = [lbl for lbl, score in zip(res["labels"], res["scores"]) if score >= 0.3]
                return preds or ["general_inquiry"]
            except Exception:
                pass
        # Fallback heuristic
        query_lower = query.lower()
        detected_intents = []
        for intent, patterns in self.query_intent_patterns.items():
            for pattern in patterns:
                if _re.search(pattern, query_lower):
                    detected_intents.append(intent)
                    break
        return detected_intents or ["general_inquiry"]
    
    def expand_query_with_synonyms(self, query: str) -> str:
        """
        Expand query with synonyms and related terms.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query
        """
        expanded_terms = []
        query_words = query.lower().split()
        
        for word in query_words:
            # Check if word has synonyms
            synonyms_found = False
            for key, synonyms in self.common_misspellings.items():
                if word in synonyms:
                    expanded_terms.append(key)
                    synonyms_found = True
                    break
            
            if not synonyms_found:
                expanded_terms.append(word)
        
        return " ".join(expanded_terms)
    
    def extract_key_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract key entities from the query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {
            "time_entities": [],
            "product_entities": [],
            "action_entities": [],
            "contact_entities": [],
            # Added specialized buckets for UI summary
            "person_entities": [],
            "product_ids": [],
            "customer_ids": [],
            "emails": [],
            "phone_numbers": []
        }
        
        query_lower = query.lower()
        
        # Time entities
        time_patterns = [
            r'\b(30 days?|7 days?|24 hours?|immediately|asap|soon)\b',
            r'\b(today|tomorrow|yesterday|this week|next week)\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, query_lower)
            entities["time_entities"].extend(matches)
        
        # Product entities
        product_patterns = [
            r'\b(product|item|order|purchase|goods)\b',
            r'\b(phone|computer|laptop|tablet|book|clothing)\b'
        ]
        
        for pattern in product_patterns:
            matches = re.findall(pattern, query_lower)
            entities["product_entities"].extend(matches)
        
        # Action entities
        action_patterns = [
            r'\b(refund|return|cancel|exchange|replace|repair)\b',
            r'\b(ship|deliver|track|send|receive)\b',
            r'\b(contact|call|email|speak|talk)\b'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, query_lower)
            entities["action_entities"].extend(matches)
        
        # Contact entities
        contact_patterns = [
            r'\b(support|service|help|assistance)\b',
            r'\b(agent|representative|manager|supervisor)\b'
        ]
        
        for pattern in contact_patterns:
            matches = re.findall(pattern, query_lower)
            entities["contact_entities"].extend(matches)
        
        # Simple SKU capture like SKU1234
        for sku in _re.findall(r"\bSKU\d{3,6}\b", query.upper()):
            entities["product_entities"].append(sku)
            entities["product_ids"].append(sku)

        # Capture patterns like "product id 12345" or "productID: 12345"
        for m in _re.findall(r"\b(product\s*id|productid|sku)\s*[:#-]?\s*(\d{3,10})\b", query_lower):
            entities["product_entities"].append(m[1])
            entities["product_ids"].append(m[1])

        # Capture order/ticket numbers if provided
        for m in _re.findall(r"\b(order|ticket|case)\s*[:#-]?\s*(\d{4,12})\b", query_lower):
            entities.setdefault("order_entities", []).append(m[1])

        # Capture customer id patterns
        for m in _re.findall(r"\b(customer\s*id|customerid|cust\s*id)\s*[:#-]?\s*(\d{3,20})\b", query_lower):
            entities["customer_ids"].append(m[1])

        # Capture emails
        for em in re2.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", query):
            entities["emails"].append(em)

        # Capture phone numbers (lenient)
        for ph in re2.findall(r"\+?\d[\d \-()]{7,}\d", query):
            cleaned = re2.sub(r"[^\d+]", "", ph)
            if len(re2.sub(r"\D", "", cleaned)) >= 8:
                entities["phone_numbers"].append(cleaned)

        # If HF NER available, add detected entities
        if self._ner_pipe:
            try:
                ner = self._ner_pipe(query)
                for ent in ner:
                    label = ent.get("entity_group") or ent.get("entity")
                    text = ent.get("word") or ent.get("text")
                    if not text:
                        continue
                    if label and "PER" in label:
                        entities.setdefault("person_entities", []).append(text)
                    elif label and ("ORG" in label or "PROD" in label or "MISC" in label):
                        entities.setdefault("product_entities", []).append(text)
            except Exception:
                pass

        # spaCy NER (prefer PERSON for names)
        if self._spacy:
            try:
                doc = self._spacy(query)
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        entities["person_entities"].append(ent.text)
            except Exception:
                pass

        # Remove duplicates
        for key in entities:
            if isinstance(entities.get(key), list):
                entities[key] = list(set(entities[key]))
        
        return entities
    
    def enhance_query_for_retrieval(self, query: str) -> Dict[str, Any]:
        """
        Enhance query for better retrieval performance.
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query information
        """
        # Clean the query
        cleaned_query = self.clean_asr_output(query)
        
        # Detect intent
        intents = self.detect_query_intent(cleaned_query)
        
        # Expand with synonyms
        expanded_query = self.expand_query_with_synonyms(cleaned_query)
        
        # Extract entities
        entities = self.extract_key_entities(cleaned_query)
        
        # Create multiple query variations
        query_variations = [
            cleaned_query,
            expanded_query
        ]
        
        # Add intent-specific variations
        for intent in intents:
            if intent == "refund":
                query_variations.append(f"refund policy return money back")
            elif intent == "shipping":
                query_variations.append(f"shipping delivery track order")
            elif intent == "warranty":
                query_variations.append(f"warranty guarantee repair replace")
            elif intent == "password":
                query_variations.append(f"password reset login account access")
            elif intent == "payment":
                query_variations.append(f"payment billing charge credit card")
            elif intent == "support":
                query_variations.append(f"customer support help contact")
        
        # Remove duplicates and empty queries
        query_variations = list(set([q for q in query_variations if q.strip()]))
        
        # Lightweight sentiment (HF if available, otherwise heuristic)
        sentiment_label, sentiment_score = self._detect_sentiment(cleaned_query)

        # Language detection (very lightweight heuristic)
        language_code = self._detect_language(cleaned_query)

        return {
            "original_query": query,
            "cleaned_query": cleaned_query,
            "expanded_query": expanded_query,
            "query_variations": query_variations,
            "detected_intents": intents,
            "extracted_entities": entities,
            "primary_intent": intents[0] if intents else "general_inquiry",
            "confidence": self._calculate_intent_confidence(cleaned_query, intents),
            "sentiment": {
                "label": sentiment_label,
                "score": sentiment_score
            },
            "language": language_code,
        }
    
    def _calculate_intent_confidence(self, query: str, intents: List[str]) -> float:
        """
        Calculate confidence score for intent detection.
        
        Args:
            query: User query
            intents: Detected intents
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not intents or intents == ["general_inquiry"]:
            return 0.3
        
        query_lower = query.lower()
        total_matches = 0
        total_patterns = 0
        
        for intent in intents:
            if intent in self.query_intent_patterns:
                patterns = self.query_intent_patterns[intent]
                total_patterns += len(patterns)
                
                for pattern in patterns:
                    if re.search(pattern, query_lower):
                        total_matches += 1
        
        if total_patterns == 0:
            return 0.3
        
        confidence = total_matches / total_patterns
        return min(confidence, 1.0)
    
    def transform_for_retrieval(self, query: str, max_variations: int = 3) -> List[str]:
        """
        Transform query for optimal retrieval.
        
        Args:
            query: Original user query
            max_variations: Maximum number of query variations to return
            
        Returns:
            List of query variations for retrieval
        """
        enhanced_info = self.enhance_query_for_retrieval(query)
        
        # Prioritize query variations based on confidence and intent
        variations = enhanced_info["query_variations"]
        
        # Sort by relevance (original query first, then by confidence)
        if enhanced_info["cleaned_query"] in variations:
            variations.remove(enhanced_info["cleaned_query"])
            variations.insert(0, enhanced_info["cleaned_query"])
        
        # Return top variations
        return variations[:max_variations]
    
    def get_query_metadata(self, query: str) -> Dict[str, Any]:
        """
        Get comprehensive metadata about the query.
        
        Args:
            query: User query
            
        Returns:
            Query metadata
        """
        enhanced_info = self.enhance_query_for_retrieval(query)
        
        # Derive a simple priority from escalation + confidence
        # 1 (lowest) .. 5 (highest)
        priority = 3
        if enhanced_info["confidence"] < 0.2:
            priority = 4
        if self._should_escalate(enhanced_info):
            priority = 5

        return {
            "timestamp": datetime.now().isoformat(),
            "original_query": query,
            "cleaned_query": enhanced_info["cleaned_query"],
            "detected_intents": enhanced_info["detected_intents"],
            "primary_intent": enhanced_info["primary_intent"],
            "intent_confidence": enhanced_info["confidence"],
            "extracted_entities": enhanced_info["extracted_entities"],
            "sentiment": enhanced_info.get("sentiment", {"label": "neutral", "score": 0.0}),
            "language": enhanced_info.get("language", "en"),
            "query_length": len(query),
            "word_count": len(query.split()),
            "has_question_mark": "?" in query,
            "has_exclamation": "!" in query,
            "is_greeting": "greeting" in enhanced_info["detected_intents"],
            "is_goodbye": "goodbye" in enhanced_info["detected_intents"],
            "requires_escalation": self._should_escalate(enhanced_info),
            "priority": priority
        }
    
    def _should_escalate(self, enhanced_info: Dict[str, Any]) -> bool:
        """
        Determine if query should be escalated to human agent.
        
        Args:
            enhanced_info: Enhanced query information
            
        Returns:
            True if escalation is recommended
        """
        intents = enhanced_info["detected_intents"]
        confidence = enhanced_info["confidence"]
        
        # Escalate if confidence is very low
        if confidence < 0.2:
            return True
        
        # Escalate for complex intents
        complex_intents = ["complaint", "dispute", "legal", "urgent"]
        if any(intent in intents for intent in complex_intents):
            return True
        
        return False

    def _detect_language(self, text: str) -> str:
        """Very lightweight language detection for UI routing.
        - Returns 'hi' if Devanagari characters present (Hindi/Marathi)
        - Otherwise returns 'en'
        """
        if not text:
            return "en"
        # Devanagari Unicode range: \u0900-\u097F (Hindi/Marathi and others)
        if _re.search(r"[\u0900-\u097F]", text):
            # Heuristic split between Hindi and Marathi
            t = text.lower()
            marathi_markers = ["मी", "आहे", "तुम्ही", "करा", "कृपया", "धन्यवाद", "होय"]
            hindi_markers = ["मैं", "हूँ", "आप", "करें", "कृपया", "धन्यवाद", "हाँ"]
            m_hits = sum(1 for w in marathi_markers if w.lower() in t)
            h_hits = sum(1 for w in hindi_markers if w.lower() in t)
            if m_hits > h_hits:
                return "mr"
            return "hi"
        return "en"

    def _detect_sentiment(self, text: str) -> Tuple[str, float]:
        """Detect sentiment label and score.
        If HF pipeline is available, use it. Otherwise use a tiny heuristic.
        """
        try:
            if self._sentiment_pipe:
                out = self._sentiment_pipe(text[:512])  # limit length
                if isinstance(out, list) and out:
                    item = out[0]
                    label = str(item.get("label", "NEUTRAL")).lower()
                    score = float(item.get("score", 0.5))
                    if label.startswith("pos"):
                        return ("positive", score)
                    if label.startswith("neg"):
                        return ("negative", score)
                    return ("neutral", score)
        except Exception:
            pass

        # Heuristic fallback
        t = text.lower()
        pos_words = ["great", "good", "thanks", "awesome", "love", "helpful"]
        neg_words = ["bad", "terrible", "angry", "upset", "refund", "complaint", "broken"]
        pos = sum(1 for w in pos_words if w in t)
        neg = sum(1 for w in neg_words if w in t)
        if neg > pos:
            return ("negative", 0.7)
        if pos > neg:
            return ("positive", 0.7)
        return ("neutral", 0.5)
