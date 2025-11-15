"""
Cross-Cultural, Personality-Aware AI Negotiator with Multilingual Transcription
Advanced CRM system component for multilingual customer interaction
"""

import json
import re
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)

class Language(Enum):
    ENGLISH = "en"
    HINDI = "hi"
    MARATHI = "mr"
    MIXED = "mixed"

class PersonalityTrait(Enum):
    ANALYTICAL = "analytical"
    IMPULSIVE = "impulsive"
    RELATIONSHIP_DRIVEN = "relationship_driven"
    PRICE_SENSITIVE = "price_sensitive"
    NEGOTIATOR = "negotiator"
    FORMAL = "formal"
    CASUAL = "casual"
    BARGAIN_SEEKER = "bargain_seeker"
    TRUST_FOCUSED = "trust_focused"
    URGENCY_DRIVEN = "urgency_driven"

class CulturalContext(Enum):
    INDIAN_URBAN = "indian_urban"
    INDIAN_RURAL = "indian_rural"
    INDIAN_BUSINESS = "indian_business"
    WESTERN_BUSINESS = "western_business"
    MIXED_CULTURE = "mixed_culture"

class NegotiationStrategy(Enum):
    PRICE_FOCUSED = "price_focused"
    DATA_DRIVEN = "data_driven"
    RELATIONSHIP_FIRST = "relationship_first"
    URGENCY_BASED = "urgency_based"
    TRUST_BUILDING = "trust_building"
    VALUE_PROPOSITION = "value_proposition"

@dataclass
class TranscriptSegment:
    speaker: str
    text: str
    language: Language
    confidence: float
    timestamp: float
    sentiment: float = 0.0
    personality_indicators: List[str] = field(default_factory=list)

@dataclass
class MultilingualProfile:
    customer_id: str
    primary_language: Language
    cultural_context: CulturalContext
    personality_traits: List[PersonalityTrait]
    communication_style: str  # formal, casual, mixed
    negotiation_preferences: Dict[str, Any]
    past_interactions: List[Dict] = field(default_factory=list)
    preferred_negotiation_language: Language = Language.ENGLISH

@dataclass
class NegotiationRecommendation:
    customer_id: str
    context: str
    transcript: List[TranscriptSegment]
    detected_language: str
    detected_personality: str
    culture: str
    recommended_message: str
    justification: str
    negotiation_strategy: str
    cultural_considerations: List[str]
    tone_guidelines: Dict[str, Any]
    confidence_score: float
    recommended_message_hindi: Optional[str] = None
    recommended_message_marathi: Optional[str] = None

class MultilingualSTT:
    """Multilingual Speech-to-Text with language detection"""
    
    def __init__(self):
        self.language_indicators = {
            Language.HINDI: {
                'keywords': ['है', 'हैं', 'था', 'थी', 'थे', 'कर', 'करना', 'होना', 'जाना', 'आना'],
                'phrases': ['कैसे हैं', 'क्या कर रहे', 'कितना', 'कब', 'कहाँ', 'क्यों'],
                'patterns': [r'[क-ह]', r'[अ-औ]', r'[०-९]']
            },
            Language.MARATHI: {
                'keywords': ['आहे', 'आहोत', 'होते', 'करत', 'करणे', 'असत', 'जात', 'येत'],
                'phrases': ['कसे आहात', 'काय करत आहात', 'किती', 'केव्हा', 'कुठे', 'का'],
                'patterns': [r'[क-ह]', r'[अ-औ]', r'[०-९]']
            },
            Language.ENGLISH: {
                'keywords': ['the', 'and', 'is', 'are', 'was', 'were', 'have', 'has', 'had'],
                'phrases': ['how are you', 'what are you', 'how much', 'when', 'where', 'why'],
                'patterns': [r'[a-zA-Z]', r'\b\w+\b']
            }
        }
        
        # Mixed language patterns (Hinglish, Marathi-English)
        self.mixed_patterns = {
            'hinglish': [r'[क-ह].*[a-zA-Z]', r'[a-zA-Z].*[क-ह]'],
            'marathi_english': [r'[क-ह].*[a-zA-Z]', r'[a-zA-Z].*[क-ह]']
        }
    
    def detect_language(self, text: str) -> Language:
        """Detect primary language of the text"""
        text_lower = text.lower()
        
        # Check for mixed patterns first
        for mixed_type, patterns in self.mixed_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return Language.MIXED
        
        # Score each language
        language_scores = {}
        
        for language, indicators in self.language_indicators.items():
            score = 0
            
            # Check keywords
            for keyword in indicators['keywords']:
                if keyword in text_lower:
                    score += 1
            
            # Check phrases
            for phrase in indicators['phrases']:
                if phrase in text_lower:
                    score += 2
            
            # Check patterns
            for pattern in indicators['patterns']:
                matches = len(re.findall(pattern, text_lower))
                score += matches * 0.1
            
            language_scores[language] = score
        
        # Return language with highest score
        if language_scores:
            best_language = max(language_scores, key=language_scores.get)
            if language_scores[best_language] > 0:
                return best_language
        
        return Language.ENGLISH  # Default to English
    
    def transcribe_audio(self, audio_file_path: str) -> List[TranscriptSegment]:
        """Transcribe audio file and return segments with language detection"""
        try:
            import whisper
            import os
            
            # Load Whisper model
            model = whisper.load_model("base")
            
            # Transcribe audio
            result = model.transcribe(audio_file_path, language="hi")  # Default to Hindi, auto-detect if not
            
            # Process transcription result
            segments = []
            for i, segment in enumerate(result["segments"]):
                # Determine speaker (simple heuristic - alternate between Customer and Agent)
                speaker = "Customer" if i % 2 == 0 else "Agent"
                
                # Detect language of the segment
                language = self.detect_language(segment["text"])
                
                transcript_segment = TranscriptSegment(
                    speaker=speaker,
                    text=segment["text"],
                    language=language,
                    confidence=segment.get("no_speech_prob", 0.1),  # Use no_speech_prob as confidence
                    timestamp=segment["start"]
                )
                segments.append(transcript_segment)
            
            return segments
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {str(e)}")
            # Fallback to simulated data if transcription fails
            simulated_transcript = [
                {"speaker": "Customer", "text": "Mujhe thoda discount milega kya?", "timestamp": 0.0},
                {"speaker": "Agent", "text": "We can explore that option.", "timestamp": 5.2},
                {"speaker": "Customer", "text": "Kitna discount de sakte hain?", "timestamp": 8.5},
                {"speaker": "Agent", "text": "Let me check our current offers.", "timestamp": 12.1}
            ]
            
            segments = []
            for item in simulated_transcript:
                language = self.detect_language(item["text"])
                segment = TranscriptSegment(
                    speaker=item["speaker"],
                    text=item["text"],
                    language=language,
                    confidence=0.85,
                    timestamp=item["timestamp"]
                )
                segments.append(segment)
            
            return segments

class MultilingualPersonalityDetector:
    """Enhanced personality detection for multilingual content"""
    
    def __init__(self):
        self.trait_indicators = {
            PersonalityTrait.ANALYTICAL: {
                'hindi': ['डेटा', 'विश्लेषण', 'तुलना', 'आंकड़े', 'रिपोर्ट', 'अध्ययन'],
                'marathi': ['डेटा', 'विश्लेषण', 'तुलना', 'आकडे', 'अहवाल', 'अभ्यास'],
                'english': ['data', 'analysis', 'comparison', 'metrics', 'report', 'study'],
                'phrases': ['दिखाइए', 'साबित करें', 'तुलना करें', 'show me', 'prove it', 'compare']
            },
            PersonalityTrait.PRICE_SENSITIVE: {
                'hindi': ['कीमत', 'मूल्य', 'छूट', 'सस्ता', 'महंगा', 'बजट'],
                'marathi': ['किंमत', 'मूल्य', 'सूट', 'स्वस्त', 'महाग', 'बजेट'],
                'english': ['price', 'cost', 'discount', 'cheap', 'expensive', 'budget'],
                'phrases': ['कितना', 'कितनी', 'कितने', 'how much', 'what price', 'discount']
            },
            PersonalityTrait.IMPULSIVE: {
                'hindi': ['अभी', 'तुरंत', 'जल्दी', 'आज', 'अभी तुरंत'],
                'marathi': ['आत्ता', 'लगेच', 'झटपट', 'आज', 'आत्ता लगेच'],
                'english': ['now', 'immediate', 'quick', 'today', 'right now'],
                'phrases': ['अभी करते हैं', 'जल्दी करो', 'do it now', 'quickly']
            },
            PersonalityTrait.RELATIONSHIP_DRIVEN: {
                'hindi': ['रिश्ता', 'भरोसा', 'साझेदारी', 'दोस्ती', 'संबंध'],
                'marathi': ['नाते', 'विश्वास', 'भागीदारी', 'मैत्री', 'संबंध'],
                'english': ['relationship', 'trust', 'partnership', 'friendship', 'connection'],
                'phrases': ['लंबे समय तक', 'भरोसेमंद', 'long term', 'trustworthy']
            },
            PersonalityTrait.BARGAIN_SEEKER: {
                'hindi': ['सौदा', 'मोलभाव', 'छूट', 'छूट मिलेगी', 'कम कीमत'],
                'marathi': ['सौदा', 'मोलभाव', 'सूट', 'सूट मिळेल', 'कमी किंमत'],
                'english': ['deal', 'bargain', 'discount', 'cheaper', 'best price'],
                'phrases': ['कितना छूट', 'कम करो', 'how much discount', 'reduce price']
            },
            PersonalityTrait.TRUST_FOCUSED: {
                'hindi': ['भरोसा', 'विश्वास', 'गारंटी', 'सुरक्षित', 'भरोसेमंद'],
                'marathi': ['विश्वास', 'भरोसा', 'हमी', 'सुरक्षित', 'विश्वसनीय'],
                'english': ['trust', 'confidence', 'guarantee', 'safe', 'reliable'],
                'phrases': ['भरोसा है', 'विश्वास करें', 'trust me', 'guaranteed']
            }
        }
    
    def detect_personality_traits(self, transcript: List[TranscriptSegment]) -> List[PersonalityTrait]:
        """Detect personality traits from multilingual transcript"""
        trait_scores = {}
        
        for segment in transcript:
            text_lower = segment.text.lower()
            language = segment.language
            
            for trait, indicators in self.trait_indicators.items():
                score = 0
                
                # Check language-specific keywords
                if language in indicators:
                    for keyword in indicators[language]:
                        if keyword in text_lower:
                            score += 1
                
                # Check common phrases
                for phrase in indicators['phrases']:
                    if phrase in text_lower:
                        score += 2
                
                trait_scores[trait] = trait_scores.get(trait, 0) + score
        
        # Return traits above threshold
        threshold = 2
        detected_traits = [trait for trait, score in trait_scores.items() if score >= threshold]
        
        return detected_traits

class IndianCulturalAdapter:
    """Cultural adaptation for Indian market contexts"""
    
    def __init__(self):
        self.cultural_norms = {
            CulturalContext.INDIAN_URBAN: {
                'formality_level': 'medium',
                'politeness_indicators': ['जी', 'आप', 'कृपया', 'धन्यवाद'],
                'negotiation_style': 'relationship_first',
                'decision_making': 'family_consensus',
                'time_orientation': 'flexible',
                'greeting': 'namaste',
                'business_etiquette': ['personal_relationships', 'flexibility', 'hospitality']
            },
            CulturalContext.INDIAN_RURAL: {
                'formality_level': 'high',
                'politeness_indicators': ['जी', 'आप', 'कृपया', 'धन्यवाद', 'महोदय'],
                'negotiation_style': 'respect_based',
                'decision_making': 'community_consensus',
                'time_orientation': 'flexible',
                'greeting': 'namaste',
                'business_etiquette': ['respect', 'patience', 'community_consideration']
            },
            CulturalContext.INDIAN_BUSINESS: {
                'formality_level': 'high',
                'politeness_indicators': ['जी', 'आप', 'कृपया', 'धन्यवाद', 'सर'],
                'negotiation_style': 'hierarchy_respect',
                'decision_making': 'senior_approval',
                'time_orientation': 'business_hours',
                'greeting': 'namaste',
                'business_etiquette': ['hierarchy', 'formality', 'professional_distance']
            }
        }
        
        self.language_preferences = {
            CulturalContext.INDIAN_URBAN: [Language.HINDI, Language.ENGLISH, Language.MIXED],
            CulturalContext.INDIAN_RURAL: [Language.HINDI, Language.MARATHI],
            CulturalContext.INDIAN_BUSINESS: [Language.ENGLISH, Language.HINDI]
        }
    
    def adapt_message_tone(self, message: str, cultural_context: CulturalContext, 
                          personality_traits: List[PersonalityTrait]) -> str:
        """Adapt message tone based on cultural context and personality"""
        adapted_message = message
        
        # Apply cultural modifications
        if cultural_context == CulturalContext.INDIAN_URBAN:
            # More casual, friendly tone
            adapted_message = self._make_casual_indian(adapted_message)
        elif cultural_context == CulturalContext.INDIAN_RURAL:
            # More respectful, formal tone
            adapted_message = self._make_formal_indian(adapted_message)
        elif cultural_context == CulturalContext.INDIAN_BUSINESS:
            # Professional, respectful tone
            adapted_message = self._make_professional_indian(adapted_message)
        
        # Apply personality-based modifications
        if PersonalityTrait.FORMAL in personality_traits:
            adapted_message = self._increase_formality(adapted_message)
        elif PersonalityTrait.CASUAL in personality_traits:
            adapted_message = self._make_casual(adapted_message)
        
        return adapted_message
    
    def _make_casual_indian(self, message: str) -> str:
        """Make message more casual for urban Indian context"""
        casual_replacements = {
            'I understand': 'मैं समझता हूं',
            'Thank you': 'धन्यवाद',
            'Please': 'कृपया',
            'Sir': 'भाई',
            'Madam': 'दीदी'
        }
        
        for formal, casual in casual_replacements.items():
            message = message.replace(formal, casual)
        
        return message
    
    def _make_formal_indian(self, message: str) -> str:
        """Make message more formal for rural Indian context"""
        formal_replacements = {
            'I understand': 'मैं समझता हूं',
            'Thank you': 'आपका धन्यवाद',
            'Please': 'कृपया',
            'Sir': 'महोदय',
            'Madam': 'महोदया'
        }
        
        for casual, formal in formal_replacements.items():
            message = message.replace(casual, formal)
        
        return message
    
    def _make_professional_indian(self, message: str) -> str:
        """Make message professional for Indian business context"""
        professional_replacements = {
            'I understand': 'मैं समझता हूं',
            'Thank you': 'धन्यवाद',
            'Please': 'कृपया',
            'Sir': 'सर',
            'Madam': 'मैडम'
        }
        
        for informal, professional in professional_replacements.items():
            message = message.replace(informal, professional)
        
        return message
    
    def _increase_formality(self, message: str) -> str:
        """Increase formality of message"""
        formal_replacements = {
            'I can': 'I would be able to',
            'We can': 'We would be able to',
            'Let me': 'Allow me to',
            'I think': 'I believe'
        }
        
        for casual, formal in formal_replacements.items():
            message = message.replace(casual, formal)
        
        return message
    
    def _make_casual(self, message: str) -> str:
        """Make message more casual"""
        casual_replacements = {
            'I would be able to': 'I can',
            'We would be able to': 'We can',
            'Allow me to': 'Let me',
            'I believe': 'I think'
        }
        
        for formal, casual in casual_replacements.items():
            message = message.replace(formal, casual)
        
        return message

class MultilingualTranslator:
    """Real-time translation with cultural sensitivity"""
    
    def __init__(self):
        # Use language codes ('hi','mr','en') as keys
        self.translation_templates = {
            'price_sensitive': {
                'hi': 'मैं आपकी चिंता समझता हूं। आपके लिए {discount}% की छूट दे सकते हैं।',
                'mr': 'मी तुमची चिंता समजतो। तुम्हाला {discount}% सूट देऊ शकतो।',
                'en': 'I understand your concern. We can offer you a {discount}% discount.'
            },
            'analytical': {
                'hi': 'मैं आपके लिए विस्तृत विश्लेषण तैयार करूंगा।',
                'mr': 'मी तुमच्यासाठी तपशीलवार विश्लेषण तयार करेन।',
                'en': 'I will prepare a detailed analysis for you.'
            },
            'relationship_driven': {
                'hi': 'हमारा रिश्ता महत्वपूर्ण है। आइए एक साथ काम करें।',
                'mr': 'आमचे नाते महत्वाचे आहे। आपण एकत्र काम करूया।',
                'en': 'Our relationship is important. Let us work together.'
            },
            'impulsive': {
                'hi': 'यह एक सीमित समय का प्रस्ताव है। जल्दी निर्णय लें।',
                'mr': 'हा मर्यादित वेळेचा प्रस्ताव आहे। लवकर निर्णय घ्या।',
                'en': 'This is a limited time offer. Please decide quickly.'
            }
        }
    
    def translate_message(self, message: str, target_language: Language, 
                         personality_traits: List[PersonalityTrait]) -> str:
        """Translate message to target language with personality adaptation"""
        
        # Determine personality-based template
        template_key = 'analytical'  # Default
        if PersonalityTrait.PRICE_SENSITIVE in personality_traits:
            template_key = 'price_sensitive'
        elif PersonalityTrait.RELATIONSHIP_DRIVEN in personality_traits:
            template_key = 'relationship_driven'
        elif PersonalityTrait.IMPULSIVE in personality_traits:
            template_key = 'impulsive'
        
        # Get template for target language (keys are codes: 'en', 'hi', 'mr')
        code = target_language.value
        if code in self.translation_templates[template_key]:
            template = self.translation_templates[template_key][code]
            
            # Replace placeholders
            if '{discount}' in template:
                template = template.replace('{discount}', '5')
            
            return template
        
        return message  # Return original if no translation available

class MultilingualNegotiationEngine:
    """Main negotiation engine for multilingual scenarios"""
    
    def __init__(self):
        self.stt = MultilingualSTT()
        self.personality_detector = MultilingualPersonalityDetector()
        self.cultural_adapter = IndianCulturalAdapter()
        self.translator = MultilingualTranslator()
        
        self.negotiation_strategies = {
            PersonalityTrait.PRICE_SENSITIVE: {
                'strategy': NegotiationStrategy.PRICE_FOCUSED,
                'tactics': ['emphasize_discounts', 'show_value', 'offer_bundles'],
                'language_style': 'value_oriented',
                'cultural_considerations': ['respect_budget', 'show_understanding']
            },
            PersonalityTrait.ANALYTICAL: {
                'strategy': NegotiationStrategy.DATA_DRIVEN,
                'tactics': ['provide_data', 'show_comparisons', 'present_metrics'],
                'language_style': 'technical',
                'cultural_considerations': ['detailed_analysis', 'professional_tone']
            },
            PersonalityTrait.RELATIONSHIP_DRIVEN: {
                'strategy': NegotiationStrategy.RELATIONSHIP_FIRST,
                'tactics': ['build_trust', 'emphasize_partnership', 'personal_connection'],
                'language_style': 'warm',
                'cultural_considerations': ['respect_relationship', 'personal_touch']
            },
            PersonalityTrait.IMPULSIVE: {
                'strategy': NegotiationStrategy.URGENCY_BASED,
                'tactics': ['create_urgency', 'limited_time', 'immediate_benefits'],
                'language_style': 'energetic',
                'cultural_considerations': ['quick_response', 'clear_deadlines']
            }
        }
    
    def analyze_negotiation(self, audio_file_path: str, customer_profile: MultilingualProfile, 
                          context: str) -> NegotiationRecommendation:
        """Main analysis function for multilingual negotiation"""
        
        # Step 1: Transcribe audio
        transcript = self.stt.transcribe_audio(audio_file_path)
        
        # Step 2: Detect personality traits
        personality_traits = self.personality_detector.detect_personality_traits(transcript)
        
        # Step 3: Determine primary language
        languages = [segment.language for segment in transcript]
        primary_language = self._determine_primary_language(languages)
        
        # Step 4: Generate negotiation strategy
        strategy = self._get_negotiation_strategy(personality_traits)
        
        # Step 5: Generate recommended message (in primary language)
        base_message = self._generate_recommendation_message(
            personality_traits, customer_profile.cultural_context, context
        )
        
        # Step 6: Adapt message culturally
        adapted_message = self.cultural_adapter.adapt_message_tone(
            base_message, customer_profile.cultural_context, personality_traits
        )

        # Translate the adapted message into the detected primary language for the main recommendation
        primary_message = self.translator.translate_message(
            adapted_message, primary_language, personality_traits
        )
        
        # Step 7: Generate translations
        hindi_message = self.translator.translate_message(
            adapted_message, Language.HINDI, personality_traits
        )
        marathi_message = self.translator.translate_message(
            adapted_message, Language.MARATHI, personality_traits
        )
        
        # Step 8: Generate justification
        justification = self._generate_justification(
            personality_traits, customer_profile.cultural_context, primary_language
        )
        
        # Step 9: Generate cultural considerations
        cultural_considerations = self._generate_cultural_considerations(
            customer_profile.cultural_context, personality_traits
        )
        
        return NegotiationRecommendation(
            customer_id=customer_profile.customer_id,
            context=context,
            transcript=transcript,
            detected_language=primary_language.value,
            detected_personality=personality_traits[0].value if personality_traits else 'analytical',
            culture=customer_profile.cultural_context.value,
            recommended_message=primary_message,
            recommended_message_hindi=hindi_message,
            recommended_message_marathi=marathi_message,
            justification=justification,
            negotiation_strategy=strategy['strategy'].value,
            cultural_considerations=cultural_considerations,
            tone_guidelines=strategy,
            confidence_score=self._calculate_confidence_score(transcript, personality_traits)
        )
    
    def _determine_primary_language(self, languages: List[Language]) -> Language:
        """Determine primary language from transcript"""
        if not languages:
            return Language.ENGLISH
        
        # Count language occurrences
        language_counts = {}
        for lang in languages:
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Return most common language
        return max(language_counts, key=language_counts.get)
    
    def _get_negotiation_strategy(self, personality_traits: List[PersonalityTrait]) -> Dict[str, Any]:
        """Get negotiation strategy based on personality traits"""
        if not personality_traits:
            return self.negotiation_strategies[PersonalityTrait.ANALYTICAL]
        
        # Use the most prominent trait
        primary_trait = personality_traits[0]
        return self.negotiation_strategies.get(primary_trait, self.negotiation_strategies[PersonalityTrait.ANALYTICAL])
    
    def _generate_recommendation_message(self, personality_traits: List[PersonalityTrait], 
                                       cultural_context: CulturalContext, context: str) -> str:
        """Generate recommended negotiation message"""
        if PersonalityTrait.PRICE_SENSITIVE in personality_traits:
            return "I completely understand your concern about pricing. Since you've been a valued customer, I can offer you a special 5% discount if we finalize this within the next 3 days."
        elif PersonalityTrait.ANALYTICAL in personality_traits:
            return "I have prepared a detailed analysis of our proposal. Let me walk you through the data and show you the ROI calculations that support this decision."
        elif PersonalityTrait.RELATIONSHIP_DRIVEN in personality_traits:
            return "Our partnership means a lot to us. Let's work together to find a solution that benefits both our companies in the long term."
        elif PersonalityTrait.IMPULSIVE in personality_traits:
            return "This is a limited-time offer with immediate benefits. I recommend we move forward quickly to secure these advantages."
        else:
            return "I appreciate your interest. Let me present our best offer and explain how it can benefit your business."
    
    def _generate_justification(self, personality_traits: List[PersonalityTrait], 
                              cultural_context: CulturalContext, primary_language: Language) -> str:
        """Generate justification for the recommendation"""
        trait_names = [trait.value.replace('_', ' ') for trait in personality_traits]
        cultural_name = cultural_context.value.replace('_', ' ')
        
        justification = f"Detected {', '.join(trait_names)} traits in {primary_language.value} conversation. "
        justification += f"Customer shows {cultural_name} cultural context. "
        justification += f"Recommended approach matches personality and cultural expectations."
        
        return justification
    
    def _generate_cultural_considerations(self, cultural_context: CulturalContext, 
                                       personality_traits: List[PersonalityTrait]) -> List[str]:
        """Generate cultural considerations for the interaction"""
        considerations = []
        
        if cultural_context == CulturalContext.INDIAN_URBAN:
            considerations.extend([
                "Use friendly, approachable tone",
                "Mix Hindi and English naturally",
                "Show understanding of local market"
            ])
        elif cultural_context == CulturalContext.INDIAN_RURAL:
            considerations.extend([
                "Use respectful, formal language",
                "Be patient with decision-making",
                "Consider community impact"
            ])
        elif cultural_context == CulturalContext.INDIAN_BUSINESS:
            considerations.extend([
                "Maintain professional hierarchy",
                "Use formal business language",
                "Respect seniority and authority"
            ])
        
        if PersonalityTrait.FORMAL in personality_traits:
            considerations.append("Maintain formal tone throughout")
        elif PersonalityTrait.CASUAL in personality_traits:
            considerations.append("Use casual, friendly approach")
        
        return considerations
    
    def _calculate_confidence_score(self, transcript: List[TranscriptSegment], 
                                  personality_traits: List[PersonalityTrait]) -> float:
        """Calculate confidence score for the analysis"""
        base_confidence = 0.5
        
        # Increase confidence with more transcript data
        if len(transcript) > 5:
            base_confidence += 0.2
        elif len(transcript) > 2:
            base_confidence += 0.1
        
        # Increase confidence with clear personality traits
        if len(personality_traits) > 2:
            base_confidence += 0.2
        elif len(personality_traits) > 0:
            base_confidence += 0.1
        
        # Increase confidence with mixed language (shows cultural awareness)
        languages = [segment.language for segment in transcript]
        if Language.MIXED in languages:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)

# Example usage and testing
def create_sample_multilingual_profile() -> MultilingualProfile:
    """Create a sample multilingual profile for testing"""
    return MultilingualProfile(
        customer_id="A102",
        primary_language=Language.HINDI,
        cultural_context=CulturalContext.INDIAN_URBAN,
        personality_traits=[PersonalityTrait.PRICE_SENSITIVE, PersonalityTrait.BARGAIN_SEEKER],
        communication_style="mixed",
        negotiation_preferences={
            "preferred_language": "hindi",
            "formality_level": "medium",
            "negotiation_style": "relationship_first"
        }
    )

if __name__ == "__main__":
    # Test the multilingual negotiator
    negotiator = MultilingualNegotiationEngine()
    
    # Create sample data
    customer_profile = create_sample_multilingual_profile()
    context = "Software subscription renewal call"
    audio_file_path = "sample_call.wav"  # Simulated audio file
    
    # Generate recommendation
    recommendation = negotiator.analyze_negotiation(audio_file_path, customer_profile, context)
    
    # Print results
    print("=== Multilingual AI Negotiator Recommendation ===")
    print(f"Customer ID: {recommendation.customer_id}")
    print(f"Context: {recommendation.context}")
    print(f"Detected Language: {recommendation.detected_language}")
    print(f"Detected Personality: {recommendation.detected_personality}")
    print(f"Culture: {recommendation.culture}")
    print(f"Recommended Message (English): {recommendation.recommended_message}")
    print(f"Recommended Message (Hindi): {recommendation.recommended_message_hindi}")
    print(f"Recommended Message (Marathi): {recommendation.recommended_message_marathi}")
    print(f"Justification: {recommendation.justification}")
    print(f"Negotiation Strategy: {recommendation.negotiation_strategy}")
    print(f"Cultural Considerations: {recommendation.cultural_considerations}")
    print(f"Confidence Score: {recommendation.confidence_score:.1%}")
    
    print("\n=== Transcript ===")
    for segment in recommendation.transcript:
        print(f"[{segment.timestamp:.1f}s] {segment.speaker}: {segment.text} ({segment.language.value})")
