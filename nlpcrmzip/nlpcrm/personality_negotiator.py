"""
Cross-Cultural, Personality-Aware AI Negotiator Module
Advanced CRM system component for intelligent customer interaction
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PersonalityType(Enum):
    ANALYTICAL = "analytical"
    RELATIONSHIP_DRIVEN = "relationship_driven"
    PRICE_SENSITIVE = "price_sensitive"
    IMPULSIVE = "impulsive"
    FORMAL = "formal"
    CASUAL = "casual"
    DECISION_MAKER = "decision_maker"
    INFLUENCER = "influencer"

class CulturalRegion(Enum):
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    LATIN_AMERICA = "latin_america"
    MIDDLE_EAST = "middle_east"
    AFRICA = "africa"

class CommunicationStyle(Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    HIGH_CONTEXT = "high_context"
    LOW_CONTEXT = "low_context"
    FORMAL = "formal"
    INFORMAL = "informal"

@dataclass
class CustomerProfile:
    customer_id: str
    age_range: Optional[str] = None
    region: Optional[str] = None
    culture: Optional[str] = None
    personality_traits: List[PersonalityType] = None
    communication_preferences: Dict[str, Any] = None
    past_interactions: List[Dict] = None
    negotiation_history: List[Dict] = None
    
    def __post_init__(self):
        if self.personality_traits is None:
            self.personality_traits = []
        if self.communication_preferences is None:
            self.communication_preferences = {}
        if self.past_interactions is None:
            self.past_interactions = []
        if self.negotiation_history is None:
            self.negotiation_history = []

@dataclass
class NegotiationContext:
    intent: str  # sales_negotiation, support_resolution, partnership_deal
    product_service: str
    current_stage: str  # initial_contact, proposal, negotiation, closing
    customer_objections: List[str]
    budget_range: Optional[str]
    timeline: Optional[str]
    decision_makers: List[str]

@dataclass
class NegotiationRecommendation:
    customer_id: str
    context: str
    detected_personality: str
    culture: str
    recommended_message: str
    justification: str
    negotiation_strategy: str
    tone_guidelines: Dict[str, Any]
    cultural_considerations: List[str]
    alternative_approaches: List[str]
    confidence_score: float

class PersonalityDetector:
    """Detects customer personality traits from communication patterns"""
    
    def __init__(self):
        self.personality_indicators = {
            PersonalityType.ANALYTICAL: {
                'keywords': ['data', 'analysis', 'metrics', 'roi', 'comparison', 'evidence', 'research', 'statistics'],
                'phrases': ['show me the numbers', 'what are the facts', 'prove it', 'based on data'],
                'patterns': [r'\d+%', r'\$\d+', r'roi', r'kpi', r'metrics']
            },
            PersonalityType.RELATIONSHIP_DRIVEN: {
                'keywords': ['trust', 'relationship', 'partnership', 'long-term', 'collaboration', 'team'],
                'phrases': ['let\'s work together', 'build trust', 'long-term partnership', 'mutual benefit'],
                'patterns': [r'relationship', r'partnership', r'collaboration', r'trust']
            },
            PersonalityType.PRICE_SENSITIVE: {
                'keywords': ['price', 'cost', 'budget', 'expensive', 'cheap', 'discount', 'deal', 'affordable'],
                'phrases': ['what\'s the price', 'too expensive', 'budget constraints', 'best deal'],
                'patterns': [r'\$\d+', r'price', r'cost', r'budget', r'discount']
            },
            PersonalityType.IMPULSIVE: {
                'keywords': ['urgent', 'immediate', 'now', 'quick', 'fast', 'limited time', 'deadline'],
                'phrases': ['need it now', 'urgent', 'immediate action', 'limited time offer'],
                'patterns': [r'urgent', r'immediate', r'now', r'quick', r'deadline']
            },
            PersonalityType.FORMAL: {
                'keywords': ['please', 'thank you', 'sincerely', 'respectfully', 'regards'],
                'phrases': ['I would appreciate', 'thank you for your time', 'looking forward to'],
                'patterns': [r'please', r'thank you', r'sincerely', r'respectfully']
            },
            PersonalityType.CASUAL: {
                'keywords': ['hey', 'hi', 'thanks', 'cool', 'awesome', 'yeah', 'sure'],
                'phrases': ['no problem', 'sounds good', 'let\'s do it', 'that works'],
                'patterns': [r'hey', r'hi', r'thanks', r'cool', r'awesome']
            }
        }
    
    def detect_personality(self, text: str, customer_profile: CustomerProfile) -> List[PersonalityType]:
        """Detect personality traits from text and profile"""
        text_lower = text.lower()
        detected_traits = []
        
        for trait, indicators in self.personality_indicators.items():
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
                if re.search(pattern, text_lower):
                    score += 1
            
            # Consider customer profile
            if trait in customer_profile.personality_traits:
                score += 1
            
            # Threshold for detection
            if score >= 2:
                detected_traits.append(trait)
        
        return detected_traits

class CulturalNormsDatabase:
    """Database of cultural communication norms and preferences"""
    
    def __init__(self):
        self.cultural_norms = {
            'japanese': {
                'communication_style': CommunicationStyle.INDIRECT,
                'formality_level': 'high',
                'politeness_indicators': ['honorifics', 'apologies', 'gratitude'],
                'negotiation_style': 'consensus_building',
                'decision_making': 'group_consensus',
                'time_orientation': 'long_term',
                'greeting': 'formal_bow',
                'business_etiquette': ['business_cards', 'formal_meetings', 'hierarchy_respect']
            },
            'american': {
                'communication_style': CommunicationStyle.DIRECT,
                'formality_level': 'medium',
                'politeness_indicators': ['please', 'thank_you'],
                'negotiation_style': 'direct_negotiation',
                'decision_making': 'individual',
                'time_orientation': 'short_term',
                'greeting': 'handshake',
                'business_etiquette': ['punctuality', 'efficiency', 'results_focused']
            },
            'german': {
                'communication_style': CommunicationStyle.DIRECT,
                'formality_level': 'high',
                'politeness_indicators': ['formal_language', 'titles'],
                'negotiation_style': 'systematic',
                'decision_making': 'thorough_analysis',
                'time_orientation': 'long_term',
                'greeting': 'formal_handshake',
                'business_etiquette': ['punctuality', 'preparation', 'detail_oriented']
            },
            'british': {
                'communication_style': CommunicationStyle.INDIRECT,
                'formality_level': 'high',
                'politeness_indicators': ['understatement', 'apologies', 'politeness'],
                'negotiation_style': 'diplomatic',
                'decision_making': 'consensus',
                'time_orientation': 'medium_term',
                'greeting': 'formal_handshake',
                'business_etiquette': ['small_talk', 'hierarchy', 'professional_distance']
            },
            'chinese': {
                'communication_style': CommunicationStyle.HIGH_CONTEXT,
                'formality_level': 'high',
                'politeness_indicators': ['respect', 'face_saving', 'harmony'],
                'negotiation_style': 'relationship_first',
                'decision_making': 'group_consensus',
                'time_orientation': 'long_term',
                'greeting': 'formal_nod',
                'business_etiquette': ['guanxi', 'face', 'hierarchy', 'gift_giving']
            },
            'indian': {
                'communication_style': CommunicationStyle.HIGH_CONTEXT,
                'formality_level': 'medium',
                'politeness_indicators': ['respect', 'family_mentions'],
                'negotiation_style': 'relationship_building',
                'decision_making': 'family_consensus',
                'time_orientation': 'flexible',
                'greeting': 'namaste',
                'business_etiquette': ['personal_relationships', 'flexibility', 'hospitality']
            },
            'brazilian': {
                'communication_style': CommunicationStyle.HIGH_CONTEXT,
                'formality_level': 'medium',
                'politeness_indicators': ['warmth', 'personal_connection'],
                'negotiation_style': 'relationship_focused',
                'decision_making': 'personal_trust',
                'time_orientation': 'flexible',
                'greeting': 'warm_handshake',
                'business_etiquette': ['personal_relationships', 'flexibility', 'warmth']
            },
            'arab': {
                'communication_style': CommunicationStyle.HIGH_CONTEXT,
                'formality_level': 'high',
                'politeness_indicators': ['hospitality', 'respect', 'honor'],
                'negotiation_style': 'relationship_honor',
                'decision_making': 'family_consensus',
                'time_orientation': 'flexible',
                'greeting': 'formal_greeting',
                'business_etiquette': ['hospitality', 'respect', 'family_importance']
            }
        }
    
    def get_cultural_norms(self, culture: str) -> Dict[str, Any]:
        """Get cultural norms for a specific culture"""
        return self.cultural_norms.get(culture.lower(), self.cultural_norms['american'])

class NegotiationStrategies:
    """Psychology-based negotiation strategies"""
    
    def __init__(self):
        self.strategies = {
            PersonalityType.ANALYTICAL: {
                'approach': 'data_driven',
                'tactics': ['provide_roi_analysis', 'show_comparisons', 'present_metrics', 'offer_trial_data'],
                'language_style': 'formal_technical',
                'objection_handling': 'address_with_facts',
                'closing_technique': 'logical_conclusion'
            },
            PersonalityType.RELATIONSHIP_DRIVEN: {
                'approach': 'trust_building',
                'tactics': ['emphasize_partnership', 'highlight_collaboration', 'show_success_stories', 'offer_support'],
                'language_style': 'warm_professional',
                'objection_handling': 'address_concerns_personally',
                'closing_technique': 'relationship_based'
            },
            PersonalityType.PRICE_SENSITIVE: {
                'approach': 'value_focused',
                'tactics': ['emphasize_discounts', 'show_total_value', 'offer_bundles', 'highlight_savings'],
                'language_style': 'value_oriented',
                'objection_handling': 'address_cost_concerns',
                'closing_technique': 'value_proposition'
            },
            PersonalityType.IMPULSIVE: {
                'approach': 'urgency_driven',
                'tactics': ['create_urgency', 'offer_limited_time', 'show_immediate_benefits', 'simplify_decision'],
                'language_style': 'energetic_direct',
                'objection_handling': 'address_quickly',
                'closing_technique': 'urgency_close'
            },
            PersonalityType.FORMAL: {
                'approach': 'professional',
                'tactics': ['maintain_formality', 'use_proper_titles', 'follow_protocol', 'show_respect'],
                'language_style': 'formal_polite',
                'objection_handling': 'address_formally',
                'closing_technique': 'professional_close'
            },
            PersonalityType.CASUAL: {
                'approach': 'friendly',
                'tactics': ['be_conversational', 'use_casual_language', 'show_personality', 'build_rapport'],
                'language_style': 'casual_friendly',
                'objection_handling': 'address_casually',
                'closing_technique': 'friendly_close'
            }
        }
    
    def get_strategy(self, personality_traits: List[PersonalityType]) -> Dict[str, Any]:
        """Get negotiation strategy based on personality traits"""
        if not personality_traits:
            return self.strategies[PersonalityType.ANALYTICAL]
        
        # Use the most prominent trait
        primary_trait = personality_traits[0]
        return self.strategies.get(primary_trait, self.strategies[PersonalityType.ANALYTICAL])

class ToneAdapter:
    """Adapts communication tone based on personality and culture"""
    
    def __init__(self):
        self.tone_templates = {
            'formal_technical': {
                'greeting': 'Good [time], [title] [name].',
                'body': 'I would like to present [content] for your consideration.',
                'closing': 'I look forward to your response. Best regards, [sender]'
            },
            'warm_professional': {
                'greeting': 'Hello [name], I hope this message finds you well.',
                'body': 'I wanted to share [content] with you, as I believe it aligns with our partnership goals.',
                'closing': 'I\'m excited about the possibilities. Warm regards, [sender]'
            },
            'value_oriented': {
                'greeting': 'Hi [name], I have some great news about [content].',
                'body': 'I wanted to highlight the significant value and savings this offers.',
                'closing': 'This is an opportunity you won\'t want to miss. Best, [sender]'
            },
            'energetic_direct': {
                'greeting': 'Hey [name]!',
                'body': 'I have something exciting to share - [content] - and it\'s time-sensitive!',
                'closing': 'Let\'s make this happen! [sender]'
            },
            'casual_friendly': {
                'greeting': 'Hi [name]!',
                'body': 'Hope you\'re doing well! I wanted to tell you about [content].',
                'closing': 'Let me know what you think! [sender]'
            }
        }
    
    def adapt_tone(self, base_message: str, personality_traits: List[PersonalityType], 
                   cultural_norms: Dict[str, Any]) -> str:
        """Adapt message tone based on personality and culture"""
        # Determine tone style
        if PersonalityType.FORMAL in personality_traits:
            tone_style = 'formal_technical'
        elif PersonalityType.RELATIONSHIP_DRIVEN in personality_traits:
            tone_style = 'warm_professional'
        elif PersonalityType.PRICE_SENSITIVE in personality_traits:
            tone_style = 'value_oriented'
        elif PersonalityType.IMPULSIVE in personality_traits:
            tone_style = 'energetic_direct'
        else:
            tone_style = 'casual_friendly'
        
        # Apply cultural modifications
        adapted_message = self._apply_cultural_modifications(base_message, cultural_norms)
        
        return adapted_message
    
    def _apply_cultural_modifications(self, message: str, cultural_norms: Dict[str, Any]) -> str:
        """Apply cultural-specific modifications to message"""
        if cultural_norms.get('communication_style') == CommunicationStyle.INDIRECT:
            # Make language more indirect and polite
            message = self._make_indirect(message)
        
        if cultural_norms.get('formality_level') == 'high':
            # Increase formality
            message = self._increase_formality(message)
        
        return message
    
    def _make_indirect(self, message: str) -> str:
        """Make message more indirect and polite"""
        indirect_phrases = {
            'I need': 'I would appreciate',
            'You must': 'It would be helpful if',
            'This is required': 'This would be beneficial',
            'You should': 'I would suggest'
        }
        
        for direct, indirect in indirect_phrases.items():
            message = message.replace(direct, indirect)
        
        return message
    
    def _increase_formality(self, message: str) -> str:
        """Increase formality of message"""
        formal_phrases = {
            'Hi': 'Hello',
            'Thanks': 'Thank you',
            'Sure': 'Certainly',
            'No problem': 'It would be my pleasure'
        }
        
        for casual, formal in formal_phrases.items():
            message = message.replace(casual, formal)
        
        return message

class CrossCulturalNegotiator:
    """Main negotiator class that orchestrates all components"""
    
    def __init__(self):
        self.personality_detector = PersonalityDetector()
        self.cultural_norms_db = CulturalNormsDatabase()
        self.negotiation_strategies = NegotiationStrategies()
        self.tone_adapter = ToneAdapter()
    
    def analyze_customer(self, text: str, customer_profile: CustomerProfile) -> Dict[str, Any]:
        """Analyze customer personality and cultural context"""
        # Detect personality traits
        personality_traits = self.personality_detector.detect_personality(text, customer_profile)
        
        # Get cultural norms
        culture = customer_profile.culture or 'american'
        cultural_norms = self.cultural_norms_db.get_cultural_norms(culture)
        
        # Get negotiation strategy
        strategy = self.negotiation_strategies.get_strategy(personality_traits)
        
        return {
            'personality_traits': [trait.value for trait in personality_traits],
            'cultural_norms': cultural_norms,
            'negotiation_strategy': strategy,
            'confidence_score': self._calculate_confidence(personality_traits, text)
        }
    
    def generate_recommendation(self, customer_profile: CustomerProfile, 
                              context: NegotiationContext, 
                              base_message: str) -> NegotiationRecommendation:
        """Generate personalized negotiation recommendation"""
        
        # Analyze customer
        analysis = self.analyze_customer(base_message, customer_profile)
        
        # Adapt tone
        adapted_message = self.tone_adapter.adapt_tone(
            base_message, 
            [PersonalityType(trait) for trait in analysis['personality_traits']],
            analysis['cultural_norms']
        )
        
        # Generate justification
        justification = self._generate_justification(analysis, context)
        
        # Generate alternative approaches
        alternatives = self._generate_alternatives(analysis, context)
        
        return NegotiationRecommendation(
            customer_id=customer_profile.customer_id,
            context=context.intent,
            detected_personality=analysis['personality_traits'][0] if analysis['personality_traits'] else 'analytical',
            culture=customer_profile.culture or 'american',
            recommended_message=adapted_message,
            justification=justification,
            negotiation_strategy=analysis['negotiation_strategy']['approach'],
            tone_guidelines=analysis['negotiation_strategy'],
            cultural_considerations=self._get_cultural_considerations(analysis['cultural_norms']),
            alternative_approaches=alternatives,
            confidence_score=analysis['confidence_score']
        )
    
    def _calculate_confidence(self, personality_traits: List[PersonalityType], text: str) -> float:
        """Calculate confidence score for personality detection"""
        if not personality_traits:
            return 0.3
        
        # Base confidence on number of traits detected and text length
        base_confidence = min(0.9, 0.5 + (len(personality_traits) * 0.1))
        text_factor = min(1.0, len(text) / 200)  # More text = higher confidence
        
        return base_confidence * text_factor
    
    def _generate_justification(self, analysis: Dict[str, Any], context: NegotiationContext) -> str:
        """Generate justification for the recommended approach"""
        personality = analysis['personality_traits'][0] if analysis['personality_traits'] else 'analytical'
        culture = analysis['cultural_norms']
        
        justification = f"Customer shows {personality} traits and {culture['communication_style'].value} communication style. "
        justification += f"Recommended {analysis['negotiation_strategy']['approach']} approach with "
        justification += f"{culture['formality_level']} formality level to match cultural expectations."
        
        return justification
    
    def _generate_alternatives(self, analysis: Dict[str, Any], context: NegotiationContext) -> List[str]:
        """Generate alternative approaches"""
        alternatives = []
        
        # Add cultural alternatives
        if analysis['cultural_norms']['communication_style'] == CommunicationStyle.DIRECT:
            alternatives.append("Consider more indirect approach if customer seems uncomfortable")
        
        # Add personality alternatives
        if 'analytical' in analysis['personality_traits']:
            alternatives.append("Provide additional data points if customer requests more evidence")
        
        if 'relationship_driven' in analysis['personality_traits']:
            alternatives.append("Emphasize long-term partnership benefits")
        
        return alternatives
    
    def _get_cultural_considerations(self, cultural_norms: Dict[str, Any]) -> List[str]:
        """Get cultural considerations for the interaction"""
        considerations = []
        
        if cultural_norms['communication_style'] == CommunicationStyle.INDIRECT:
            considerations.append("Use indirect language and avoid direct confrontation")
        
        if cultural_norms['formality_level'] == 'high':
            considerations.append("Maintain formal tone and use proper titles")
        
        if cultural_norms['negotiation_style'] == 'relationship_first':
            considerations.append("Focus on building relationship before discussing business")
        
        if cultural_norms['time_orientation'] == 'long_term':
            considerations.append("Emphasize long-term benefits and sustainability")
        
        return considerations

# Example usage and testing
def create_sample_customer_profile() -> CustomerProfile:
    """Create a sample customer profile for testing"""
    return CustomerProfile(
        customer_id="7890",
        age_range="35-45",
        region="Asia Pacific",
        culture="japanese",
        personality_traits=[PersonalityType.ANALYTICAL, PersonalityType.FORMAL],
        communication_preferences={
            "formality": "high",
            "directness": "low",
            "context": "high"
        },
        past_interactions=[
            {"date": "2024-01-15", "type": "email", "tone": "formal", "outcome": "positive"},
            {"date": "2024-01-20", "type": "call", "tone": "professional", "outcome": "neutral"}
        ],
        negotiation_history=[
            {"date": "2024-01-15", "stage": "proposal", "outcome": "interested", "objections": ["price", "timeline"]}
        ]
    )

def create_sample_context() -> NegotiationContext:
    """Create a sample negotiation context for testing"""
    return NegotiationContext(
        intent="sales_negotiation",
        product_service="Software License",
        current_stage="proposal",
        customer_objections=["price", "implementation timeline"],
        budget_range="$50,000-$100,000",
        timeline="Q2 2024",
        decision_makers=["CTO", "CFO"]
    )

if __name__ == "__main__":
    # Test the negotiator
    negotiator = CrossCulturalNegotiator()
    
    # Create sample data
    customer_profile = create_sample_customer_profile()
    context = create_sample_context()
    base_message = "Thank you for your interest in our software solution. I have prepared a detailed proposal that addresses your requirements and demonstrates the ROI potential."
    
    # Generate recommendation
    recommendation = negotiator.generate_recommendation(customer_profile, context, base_message)
    
    # Print results
    print("=== Cross-Cultural AI Negotiator Recommendation ===")
    print(f"Customer ID: {recommendation.customer_id}")
    print(f"Context: {recommendation.context}")
    print(f"Detected Personality: {recommendation.detected_personality}")
    print(f"Culture: {recommendation.culture}")
    print(f"Recommended Message: {recommendation.recommended_message}")
    print(f"Justification: {recommendation.justification}")
    print(f"Negotiation Strategy: {recommendation.negotiation_strategy}")
    print(f"Cultural Considerations: {recommendation.cultural_considerations}")
    print(f"Confidence Score: {recommendation.confidence_score:.2f}")
