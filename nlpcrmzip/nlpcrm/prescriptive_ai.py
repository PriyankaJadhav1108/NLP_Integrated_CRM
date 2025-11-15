"""
Prescriptive AI Module for Advanced CRM System
Predictive & Prescriptive Intelligence beyond basic CRM predictions
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import math

logger = logging.getLogger(__name__)

class PsychologyTrait(Enum):
    RISK_AVERSE = "risk_averse"
    BARGAIN_SEEKER = "bargain_seeker"
    IMPULSIVE_BUYER = "impulsive_buyer"
    ANALYTICAL = "analytical"
    RELATIONSHIP_FOCUSED = "relationship_focused"
    PRICE_SENSITIVE = "price_sensitive"
    QUICK_DECISION_MAKER = "quick_decision_maker"
    NEGOTIATOR = "negotiator"
    RELATIONSHIP_DRIVEN = "relationship_driven"
    STATUS_CONSCIOUS = "status_conscious"
    VALUE_ORIENTED = "value_oriented"

class BuyingStyle(Enum):
    QUICK_DECISION = "quick_decision"
    NEGOTIATION_HEAVY = "negotiation_heavy"
    RELATIONSHIP_FIRST = "relationship_first"
    RESEARCH_INTENSIVE = "research_intensive"
    PRICE_FOCUSED = "price_focused"
    FEATURE_FOCUSED = "feature_focused"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ActionType(Enum):
    RETENTION = "retention"
    CONVERSION = "conversion"
    ENGAGEMENT = "engagement"
    NEGOTIATION = "negotiation"
    RELATIONSHIP_BUILDING = "relationship_building"
    URGENCY_CREATION = "urgency_creation"

@dataclass
class CustomerProfile:
    customer_id: str
    demographics: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    sentiment_score: float = 0.0
    psychology_traits: List[PsychologyTrait] = field(default_factory=list)
    buying_style: Optional[BuyingStyle] = None
    lead_score: float = 0.0
    churn_risk: float = 0.0
    conversion_probability: float = 0.0
    engagement_score: float = 0.0
    last_interaction: Optional[datetime] = None
    total_interactions: int = 0
    successful_deals: int = 0
    avg_deal_value: float = 0.0
    response_rate: float = 0.0
    preferred_communication: str = "email"
    timezone: str = "UTC"

@dataclass
class InteractionHistory:
    interaction_id: str
    customer_id: str
    interaction_type: str  # email, call, chat, meeting, demo
    timestamp: datetime
    content: str
    sentiment: float
    outcome: str  # positive, negative, neutral
    response_time_hours: float
    engagement_level: float  # 0-1
    offer_presented: Optional[str] = None
    discount_offered: Optional[float] = None
    decision_made: Optional[bool] = None
    deal_value: Optional[float] = None

@dataclass
class PrescriptiveRecommendation:
    customer_id: str
    conversion_probability: float
    churn_risk: float
    recommended_action: str
    action_type: ActionType
    suggested_timeframe: str
    justification: str
    confidence_score: float
    expected_outcome: str
    priority: str  # low, medium, high, critical
    required_resources: List[str]
    alternative_actions: List[str]
    success_metrics: Dict[str, Any]

class PsychologyProfiler:
    """Advanced psychology profiling using ML-based trait detection"""
    
    def __init__(self):
        self.trait_indicators = {
            PsychologyTrait.RISK_AVERSE: {
                'keywords': ['safe', 'secure', 'guaranteed', 'proven', 'established', 'reliable'],
                'phrases': ['need to be sure', 'what if it fails', 'is this safe', 'guarantee'],
                'behavioral_patterns': ['asks_for_references', 'requests_trial', 'wants_detailed_contract']
            },
            PsychologyTrait.BARGAIN_SEEKER: {
                'keywords': ['discount', 'deal', 'cheap', 'affordable', 'budget', 'price'],
                'phrases': ['best price', 'can you do better', 'what\'s the discount', 'special offer'],
                'behavioral_patterns': ['asks_for_discounts', 'compares_prices', 'negotiates_aggressively']
            },
            PsychologyTrait.IMPULSIVE_BUYER: {
                'keywords': ['now', 'immediate', 'urgent', 'quick', 'fast', 'today'],
                'phrases': ['need it now', 'can we start today', 'immediate action', 'right away'],
                'behavioral_patterns': ['makes_quick_decisions', 'responds_to_urgency', 'short_decision_cycle']
            },
            PsychologyTrait.ANALYTICAL: {
                'keywords': ['data', 'analysis', 'metrics', 'roi', 'comparison', 'evidence'],
                'phrases': ['show me the numbers', 'what\'s the data', 'prove it', 'analysis'],
                'behavioral_patterns': ['requests_data', 'asks_for_comparisons', 'wants_detailed_reports']
            },
            PsychologyTrait.RELATIONSHIP_FOCUSED: {
                'keywords': ['partnership', 'relationship', 'trust', 'collaboration', 'team'],
                'phrases': ['long-term partnership', 'build trust', 'work together', 'mutual benefit'],
                'behavioral_patterns': ['values_relationships', 'prefers_personal_contact', 'builds_rapport']
            },
            PsychologyTrait.PRICE_SENSITIVE: {
                'keywords': ['expensive', 'cost', 'budget', 'affordable', 'value', 'worth'],
                'phrases': ['too expensive', 'budget constraints', 'cost effective', 'value for money'],
                'behavioral_patterns': ['price_focused', 'budget_conscious', 'cost_optimization']
            },
            PsychologyTrait.QUICK_DECISION_MAKER: {
                'keywords': ['decide', 'choice', 'option', 'select', 'choose', 'pick'],
                'phrases': ['let\'s decide', 'I choose', 'go with', 'that works'],
                'behavioral_patterns': ['fast_decisions', 'minimal_analysis', 'trusts_intuition']
            },
            PsychologyTrait.NEGOTIATOR: {
                'keywords': ['negotiate', 'deal', 'terms', 'conditions', 'agreement', 'contract'],
                'phrases': ['let\'s negotiate', 'work out a deal', 'find middle ground', 'compromise'],
                'behavioral_patterns': ['negotiates_terms', 'asks_for_concessions', 'bargains_aggressively']
            },
            PsychologyTrait.STATUS_CONSCIOUS: {
                'keywords': ['premium', 'exclusive', 'elite', 'vip', 'luxury', 'high-end'],
                'phrases': ['best quality', 'premium service', 'exclusive access', 'top tier'],
                'behavioral_patterns': ['values_status', 'prefers_premium', 'brand_conscious']
            },
            PsychologyTrait.VALUE_ORIENTED: {
                'keywords': ['value', 'worth', 'benefit', 'advantage', 'return', 'investment'],
                'phrases': ['good value', 'worth the investment', 'return on investment', 'benefits'],
                'behavioral_patterns': ['evaluates_value', 'considers_benefits', 'roi_focused']
            }
        }
    
    def analyze_psychology(self, customer_profile: CustomerProfile, 
                          interaction_history: List[InteractionHistory]) -> List[PsychologyTrait]:
        """Analyze customer psychology traits from profile and history"""
        trait_scores = {}
        
        # Analyze from interaction history
        for interaction in interaction_history:
            content_lower = interaction.content.lower()
            
            for trait, indicators in self.trait_indicators.items():
                score = 0
                
                # Check keywords
                for keyword in indicators['keywords']:
                    if keyword in content_lower:
                        score += 1
                
                # Check phrases
                for phrase in indicators['phrases']:
                    if phrase in content_lower:
                        score += 2
                
                # Check behavioral patterns
                for pattern in indicators['behavioral_patterns']:
                    if self._check_behavioral_pattern(interaction, pattern):
                        score += 3
                
                trait_scores[trait] = trait_scores.get(trait, 0) + score
        
        # Analyze from customer profile
        if customer_profile.preferences:
            for trait, indicators in self.trait_indicators.items():
                score = 0
                for keyword in indicators['keywords']:
                    if keyword in str(customer_profile.preferences).lower():
                        score += 1
                trait_scores[trait] = trait_scores.get(trait, 0) + score
        
        # Determine buying style based on traits
        buying_style = self._determine_buying_style(trait_scores)
        customer_profile.buying_style = buying_style
        
        # Return traits above threshold
        threshold = 3
        detected_traits = [trait for trait, score in trait_scores.items() if score >= threshold]
        
        return detected_traits
    
    def _check_behavioral_pattern(self, interaction: InteractionHistory, pattern: str) -> bool:
        """Check if interaction matches behavioral pattern"""
        pattern_mapping = {
            'asks_for_references': interaction.content.lower().count('reference') > 0,
            'requests_trial': 'trial' in interaction.content.lower(),
            'wants_detailed_contract': 'contract' in interaction.content.lower() and 'detail' in interaction.content.lower(),
            'asks_for_discounts': 'discount' in interaction.content.lower(),
            'compares_prices': 'compare' in interaction.content.lower() and 'price' in interaction.content.lower(),
            'negotiates_aggressively': interaction.content.lower().count('negotiate') > 1,
            'makes_quick_decisions': interaction.response_time_hours < 24,
            'responds_to_urgency': 'urgent' in interaction.content.lower() or 'immediate' in interaction.content.lower(),
            'short_decision_cycle': interaction.response_time_hours < 48,
            'requests_data': 'data' in interaction.content.lower() or 'analysis' in interaction.content.lower(),
            'asks_for_comparisons': 'compare' in interaction.content.lower(),
            'wants_detailed_reports': 'report' in interaction.content.lower() and 'detail' in interaction.content.lower(),
            'values_relationships': 'relationship' in interaction.content.lower() or 'partnership' in interaction.content.lower(),
            'prefers_personal_contact': interaction.interaction_type in ['call', 'meeting'],
            'builds_rapport': interaction.engagement_level > 0.7,
            'price_focused': 'price' in interaction.content.lower() or 'cost' in interaction.content.lower(),
            'budget_conscious': 'budget' in interaction.content.lower(),
            'cost_optimization': 'optimize' in interaction.content.lower() and 'cost' in interaction.content.lower(),
            'fast_decisions': interaction.response_time_hours < 12,
            'minimal_analysis': len(interaction.content) < 100,
            'trusts_intuition': 'feel' in interaction.content.lower() or 'intuition' in interaction.content.lower(),
            'negotiates_terms': 'terms' in interaction.content.lower(),
            'asks_for_concessions': 'concession' in interaction.content.lower(),
            'bargains_aggressively': interaction.content.lower().count('bargain') > 0,
            'values_status': 'premium' in interaction.content.lower() or 'exclusive' in interaction.content.lower(),
            'prefers_premium': 'premium' in interaction.content.lower(),
            'brand_conscious': 'brand' in interaction.content.lower(),
            'evaluates_value': 'value' in interaction.content.lower(),
            'considers_benefits': 'benefit' in interaction.content.lower(),
            'roi_focused': 'roi' in interaction.content.lower() or 'return' in interaction.content.lower()
        }
        
        return pattern_mapping.get(pattern, False)
    
    def _determine_buying_style(self, trait_scores: Dict[PsychologyTrait, int]) -> BuyingStyle:
        """Determine buying style based on trait scores"""
        if trait_scores.get(PsychologyTrait.QUICK_DECISION_MAKER, 0) > 5:
            return BuyingStyle.QUICK_DECISION
        elif trait_scores.get(PsychologyTrait.NEGOTIATOR, 0) > 5:
            return BuyingStyle.NEGOTIATION_HEAVY
        elif trait_scores.get(PsychologyTrait.RELATIONSHIP_FOCUSED, 0) > 5:
            return BuyingStyle.RELATIONSHIP_FIRST
        elif trait_scores.get(PsychologyTrait.ANALYTICAL, 0) > 5:
            return BuyingStyle.RESEARCH_INTENSIVE
        elif trait_scores.get(PsychologyTrait.PRICE_SENSITIVE, 0) > 5:
            return BuyingStyle.PRICE_FOCUSED
        else:
            return BuyingStyle.FEATURE_FOCUSED

class PredictiveModels:
    """Machine learning models for predicting customer behavior"""
    
    def __init__(self):
        self.model_weights = {
            'conversion': {
                'engagement_score': 0.3,
                'sentiment_score': 0.25,
                'response_rate': 0.2,
                'deal_value': 0.15,
                'interaction_frequency': 0.1
            },
            'churn': {
                'last_interaction_days': 0.4,
                'sentiment_trend': 0.3,
                'engagement_decline': 0.2,
                'response_rate': 0.1
            },
            'engagement': {
                'interaction_frequency': 0.4,
                'response_time': 0.3,
                'content_quality': 0.2,
                'sentiment_score': 0.1
            }
        }
    
    def predict_conversion_probability(self, customer_profile: CustomerProfile, 
                                     interaction_history: List[InteractionHistory]) -> float:
        """Predict probability of conversion (0-1)"""
        if not interaction_history:
            return 0.1
        
        # Calculate features
        engagement_score = self._calculate_engagement_score(interaction_history)
        sentiment_score = customer_profile.sentiment_score
        response_rate = customer_profile.response_rate
        avg_deal_value = customer_profile.avg_deal_value
        interaction_frequency = len(interaction_history) / max(1, (datetime.now() - customer_profile.last_interaction).days)
        
        # Normalize features
        engagement_score = min(1.0, engagement_score)
        sentiment_score = (sentiment_score + 1) / 2  # Convert from -1,1 to 0,1
        response_rate = min(1.0, response_rate)
        deal_value_score = min(1.0, avg_deal_value / 100000)  # Normalize to 100k max
        interaction_frequency = min(1.0, interaction_frequency)
        
        # Calculate weighted score
        weights = self.model_weights['conversion']
        conversion_prob = (
            engagement_score * weights['engagement_score'] +
            sentiment_score * weights['sentiment_score'] +
            response_rate * weights['response_rate'] +
            deal_value_score * weights['deal_value'] +
            interaction_frequency * weights['interaction_frequency']
        )
        
        return min(1.0, max(0.0, conversion_prob))
    
    def predict_churn_risk(self, customer_profile: CustomerProfile, 
                          interaction_history: List[InteractionHistory]) -> float:
        """Predict churn risk (0-1)"""
        if not interaction_history:
            return 0.8  # High risk if no interactions
        
        # Calculate features
        days_since_last = (datetime.now() - customer_profile.last_interaction).days
        sentiment_trend = self._calculate_sentiment_trend(interaction_history)
        engagement_decline = self._calculate_engagement_decline(interaction_history)
        response_rate = customer_profile.response_rate
        
        # Normalize features
        last_interaction_score = min(1.0, days_since_last / 90)  # 90 days = max risk
        sentiment_trend = (sentiment_trend + 1) / 2  # Convert to 0-1
        engagement_decline = min(1.0, max(0.0, engagement_decline))
        
        # Calculate weighted score
        weights = self.model_weights['churn']
        churn_risk = (
            last_interaction_score * weights['last_interaction_days'] +
            sentiment_trend * weights['sentiment_trend'] +
            engagement_decline * weights['engagement_decline'] +
            (1 - response_rate) * weights['response_rate']
        )
        
        return min(1.0, max(0.0, churn_risk))
    
    def predict_engagement_score(self, customer_profile: CustomerProfile, 
                               interaction_history: List[InteractionHistory]) -> float:
        """Predict engagement score (0-1)"""
        if not interaction_history:
            return 0.1
        
        # Calculate features
        interaction_frequency = len(interaction_history) / max(1, (datetime.now() - customer_profile.last_interaction).days)
        avg_response_time = np.mean([i.response_time_hours for i in interaction_history])
        content_quality = np.mean([i.engagement_level for i in interaction_history])
        sentiment_score = customer_profile.sentiment_score
        
        # Normalize features
        interaction_frequency = min(1.0, interaction_frequency)
        response_time_score = max(0.0, 1.0 - (avg_response_time / 168))  # 1 week = 0 score
        content_quality = min(1.0, content_quality)
        sentiment_score = (sentiment_score + 1) / 2
        
        # Calculate weighted score
        weights = self.model_weights['engagement']
        engagement_score = (
            interaction_frequency * weights['interaction_frequency'] +
            response_time_score * weights['response_time'] +
            content_quality * weights['content_quality'] +
            sentiment_score * weights['sentiment_score']
        )
        
        return min(1.0, max(0.0, engagement_score))
    
    def _calculate_engagement_score(self, interactions: List[InteractionHistory]) -> float:
        """Calculate engagement score from interactions"""
        if not interactions:
            return 0.0
        
        # Weight recent interactions more heavily
        now = datetime.now()
        weighted_scores = []
        
        for interaction in interactions:
            days_ago = (now - interaction.timestamp).days
            weight = max(0.1, 1.0 - (days_ago / 90))  # 90 days = 0 weight
            weighted_scores.append(interaction.engagement_level * weight)
        
        return np.mean(weighted_scores) if weighted_scores else 0.0
    
    def _calculate_sentiment_trend(self, interactions: List[InteractionHistory]) -> float:
        """Calculate sentiment trend over time"""
        if len(interactions) < 2:
            return 0.0
        
        # Sort by timestamp
        sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
        
        # Calculate trend
        early_sentiment = np.mean([i.sentiment for i in sorted_interactions[:len(sorted_interactions)//2]])
        recent_sentiment = np.mean([i.sentiment for i in sorted_interactions[len(sorted_interactions)//2:]])
        
        return recent_sentiment - early_sentiment
    
    def _calculate_engagement_decline(self, interactions: List[InteractionHistory]) -> float:
        """Calculate engagement decline over time"""
        if len(interactions) < 2:
            return 0.0
        
        # Sort by timestamp
        sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
        
        # Calculate trend
        early_engagement = np.mean([i.engagement_level for i in sorted_interactions[:len(sorted_interactions)//2]])
        recent_engagement = np.mean([i.engagement_level for i in sorted_interactions[len(sorted_interactions)//2:]])
        
        return early_engagement - recent_engagement

class PrescriptiveEngine:
    """Prescriptive AI engine that recommends specific actions"""
    
    def __init__(self):
        self.action_playbooks = {
            PsychologyTrait.PRICE_SENSITIVE: {
                'conversion': 'Offer a 5-10% discount with clear value proposition',
                'retention': 'Provide cost-saving analysis and budget optimization',
                'engagement': 'Send price comparison charts and savings calculator'
            },
            PsychologyTrait.ANALYTICAL: {
                'conversion': 'Send detailed ROI analysis and comparison reports',
                'retention': 'Provide comprehensive data dashboard and metrics',
                'engagement': 'Share case studies and technical documentation'
            },
            PsychologyTrait.IMPULSIVE_BUYER: {
                'conversion': 'Create urgency with limited-time offer and immediate benefits',
                'retention': 'Offer quick wins and immediate value delivery',
                'engagement': 'Send time-sensitive updates and quick action items'
            },
            PsychologyTrait.RELATIONSHIP_FOCUSED: {
                'conversion': 'Schedule personal call and build relationship first',
                'retention': 'Arrange executive meeting and strengthen partnership',
                'engagement': 'Send personalized content and exclusive invitations'
            },
            PsychologyTrait.RISK_AVERSE: {
                'conversion': 'Provide guarantees, references, and risk mitigation',
                'retention': 'Offer extended support and success assurance',
                'engagement': 'Share testimonials and security certifications'
            },
            PsychologyTrait.NEGOTIATOR: {
                'conversion': 'Present multiple options and flexible terms',
                'retention': 'Offer contract flexibility and negotiation room',
                'engagement': 'Schedule negotiation session and discuss terms'
            }
        }
        
        self.timeframe_guidelines = {
            RiskLevel.CRITICAL: "within 24 hours",
            RiskLevel.HIGH: "within 3 days",
            RiskLevel.MEDIUM: "within 1 week",
            RiskLevel.LOW: "within 2 weeks"
        }
    
    def generate_recommendation(self, customer_profile: CustomerProfile, 
                              interaction_history: List[InteractionHistory],
                              conversion_prob: float, churn_risk: float) -> PrescriptiveRecommendation:
        """Generate prescriptive recommendation based on predictions and psychology"""
        
        # Determine action type and priority
        action_type, priority = self._determine_action_type_and_priority(conversion_prob, churn_risk)
        
        # Get psychology-based recommendation
        psychology_traits = customer_profile.psychology_traits
        recommended_action = self._get_psychology_based_action(psychology_traits, action_type)
        
        # Determine timeframe
        risk_level = self._determine_risk_level(churn_risk, conversion_prob)
        timeframe = self.timeframe_guidelines[risk_level]
        
        # Generate justification
        justification = self._generate_justification(customer_profile, psychology_traits, 
                                                   action_type, conversion_prob, churn_risk)
        
        # Generate expected outcome
        expected_outcome = self._generate_expected_outcome(action_type, conversion_prob, churn_risk)
        
        # Generate alternative actions
        alternative_actions = self._generate_alternative_actions(psychology_traits, action_type)
        
        # Determine required resources
        required_resources = self._determine_required_resources(action_type, psychology_traits)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(customer_profile, interaction_history)
        
        # Generate success metrics
        success_metrics = self._generate_success_metrics(action_type, conversion_prob, churn_risk)
        
        return PrescriptiveRecommendation(
            customer_id=customer_profile.customer_id,
            conversion_probability=conversion_prob,
            churn_risk=churn_risk,
            recommended_action=recommended_action,
            action_type=action_type,
            suggested_timeframe=timeframe,
            justification=justification,
            confidence_score=confidence_score,
            expected_outcome=expected_outcome,
            priority=priority,
            required_resources=required_resources,
            alternative_actions=alternative_actions,
            success_metrics=success_metrics
        )
    
    def _determine_action_type_and_priority(self, conversion_prob: float, churn_risk: float) -> Tuple[ActionType, str]:
        """Determine action type and priority based on predictions"""
        if churn_risk > 0.7:
            return ActionType.RETENTION, "critical"
        elif churn_risk > 0.5:
            return ActionType.RETENTION, "high"
        elif conversion_prob > 0.6:
            return ActionType.CONVERSION, "high"
        elif conversion_prob > 0.4:
            return ActionType.NEGOTIATION, "medium"
        elif conversion_prob > 0.2:
            return ActionType.ENGAGEMENT, "medium"
        else:
            return ActionType.RELATIONSHIP_BUILDING, "low"
    
    def _get_psychology_based_action(self, psychology_traits: List[PsychologyTrait], 
                                   action_type: ActionType) -> str:
        """Get psychology-based action recommendation"""
        if not psychology_traits:
            return self._get_default_action(action_type)
        
        # Use the most prominent trait
        primary_trait = psychology_traits[0]
        action_key = action_type.value
        
        if primary_trait in self.action_playbooks and action_key in self.action_playbooks[primary_trait]:
            return self.action_playbooks[primary_trait][action_key]
        
        return self._get_default_action(action_type)
    
    def _get_default_action(self, action_type: ActionType) -> str:
        """Get default action if no psychology match"""
        default_actions = {
            ActionType.RETENTION: "Schedule a personal call to understand concerns and offer solutions",
            ActionType.CONVERSION: "Present a compelling value proposition with clear next steps",
            ActionType.NEGOTIATION: "Offer flexible terms and multiple options to choose from",
            ActionType.ENGAGEMENT: "Send relevant content and schedule a follow-up meeting",
            ActionType.RELATIONSHIP_BUILDING: "Arrange a personal meeting to build rapport and trust",
            ActionType.URGENCY_CREATION: "Create time-sensitive offer with limited availability"
        }
        return default_actions.get(action_type, "Schedule a follow-up meeting")
    
    def _determine_risk_level(self, churn_risk: float, conversion_prob: float) -> RiskLevel:
        """Determine risk level based on predictions"""
        if churn_risk > 0.8 or conversion_prob < 0.1:
            return RiskLevel.CRITICAL
        elif churn_risk > 0.6 or conversion_prob < 0.3:
            return RiskLevel.HIGH
        elif churn_risk > 0.4 or conversion_prob < 0.5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_justification(self, customer_profile: CustomerProfile, 
                              psychology_traits: List[PsychologyTrait],
                              action_type: ActionType, conversion_prob: float, 
                              churn_risk: float) -> str:
        """Generate explainable AI justification"""
        justification_parts = []
        
        # Psychology-based justification
        if psychology_traits:
            trait_names = [trait.value.replace('_', ' ') for trait in psychology_traits]
            justification_parts.append(f"Customer shows {', '.join(trait_names)} traits")
        
        # Prediction-based justification
        if churn_risk > 0.6:
            justification_parts.append(f"High churn risk ({churn_risk:.1%}) requires immediate retention action")
        elif conversion_prob > 0.6:
            justification_parts.append(f"High conversion probability ({conversion_prob:.1%}) suggests closing opportunity")
        elif conversion_prob < 0.3:
            justification_parts.append(f"Low conversion probability ({conversion_prob:.1%}) indicates need for engagement")
        
        # Action-specific justification
        if action_type == ActionType.RETENTION:
            justification_parts.append("Retention action recommended to prevent customer loss")
        elif action_type == ActionType.CONVERSION:
            justification_parts.append("Conversion action recommended to close the deal")
        elif action_type == ActionType.NEGOTIATION:
            justification_parts.append("Negotiation action recommended to address concerns")
        
        return ". ".join(justification_parts) + "."
    
    def _generate_expected_outcome(self, action_type: ActionType, 
                                 conversion_prob: float, churn_risk: float) -> str:
        """Generate expected outcome description"""
        if action_type == ActionType.RETENTION:
            return f"Expected to reduce churn risk from {churn_risk:.1%} to {churn_risk * 0.7:.1%}"
        elif action_type == ActionType.CONVERSION:
            return f"Expected to increase conversion probability from {conversion_prob:.1%} to {min(0.9, conversion_prob * 1.3):.1%}"
        elif action_type == ActionType.NEGOTIATION:
            return f"Expected to improve negotiation position and address key concerns"
        else:
            return f"Expected to improve engagement and relationship strength"
    
    def _generate_alternative_actions(self, psychology_traits: List[PsychologyTrait], 
                                    action_type: ActionType) -> List[str]:
        """Generate alternative action recommendations"""
        alternatives = []
        
        if action_type == ActionType.RETENTION:
            alternatives.extend([
                "Offer extended trial period",
                "Provide additional support resources",
                "Schedule executive escalation call"
            ])
        elif action_type == ActionType.CONVERSION:
            alternatives.extend([
                "Present case study with similar customer",
                "Offer pilot program with success metrics",
                "Schedule product demonstration"
            ])
        elif action_type == ActionType.NEGOTIATION:
            alternatives.extend([
                "Present multiple pricing options",
                "Offer flexible payment terms",
                "Schedule negotiation session"
            ])
        
        return alternatives[:3]  # Limit to 3 alternatives
    
    def _determine_required_resources(self, action_type: ActionType, 
                                    psychology_traits: List[PsychologyTrait]) -> List[str]:
        """Determine required resources for action"""
        resources = []
        
        if action_type == ActionType.RETENTION:
            resources.extend(["Account manager", "Support team", "Executive sponsor"])
        elif action_type == ActionType.CONVERSION:
            resources.extend(["Sales representative", "Technical expert", "Legal team"])
        elif action_type == ActionType.NEGOTIATION:
            resources.extend(["Sales manager", "Pricing team", "Legal team"])
        elif action_type == ActionType.RELATIONSHIP_BUILDING:
            resources.extend(["Account manager", "Executive team", "Marketing team"])
        
        return resources
    
    def _calculate_confidence_score(self, customer_profile: CustomerProfile, 
                                  interaction_history: List[InteractionHistory]) -> float:
        """Calculate confidence score for recommendation"""
        base_confidence = 0.5
        
        # Increase confidence with more data
        if len(interaction_history) > 10:
            base_confidence += 0.2
        elif len(interaction_history) > 5:
            base_confidence += 0.1
        
        # Increase confidence with clear psychology traits
        if len(customer_profile.psychology_traits) > 2:
            base_confidence += 0.2
        elif len(customer_profile.psychology_traits) > 0:
            base_confidence += 0.1
        
        # Increase confidence with recent interactions
        if customer_profile.last_interaction and (datetime.now() - customer_profile.last_interaction).days < 7:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _generate_success_metrics(self, action_type: ActionType, 
                                conversion_prob: float, churn_risk: float) -> Dict[str, Any]:
        """Generate success metrics for the action"""
        metrics = {
            "primary_metric": "",
            "target_value": 0.0,
            "measurement_period": "30 days",
            "secondary_metrics": []
        }
        
        if action_type == ActionType.RETENTION:
            metrics["primary_metric"] = "Churn risk reduction"
            metrics["target_value"] = churn_risk * 0.7
            metrics["secondary_metrics"] = ["Engagement increase", "Response rate improvement"]
        elif action_type == ActionType.CONVERSION:
            metrics["primary_metric"] = "Conversion probability increase"
            metrics["target_value"] = min(0.9, conversion_prob * 1.3)
            metrics["secondary_metrics"] = ["Deal value increase", "Sales cycle reduction"]
        elif action_type == ActionType.NEGOTIATION:
            metrics["primary_metric"] = "Negotiation success rate"
            metrics["target_value"] = 0.8
            metrics["secondary_metrics"] = ["Deal closure rate", "Customer satisfaction"]
        else:
            metrics["primary_metric"] = "Engagement score improvement"
            metrics["target_value"] = 0.8
            metrics["secondary_metrics"] = ["Response rate", "Meeting attendance"]
        
        return metrics

class PrescriptiveAI:
    """Main Prescriptive AI orchestrator"""
    
    def __init__(self):
        self.psychology_profiler = PsychologyProfiler()
        self.predictive_models = PredictiveModels()
        self.prescriptive_engine = PrescriptiveEngine()
    
    def analyze_customer(self, customer_profile: CustomerProfile, 
                        interaction_history: List[InteractionHistory]) -> PrescriptiveRecommendation:
        """Main analysis function that generates prescriptive recommendations"""
        
        # Step 1: Analyze psychology traits
        psychology_traits = self.psychology_profiler.analyze_psychology(customer_profile, interaction_history)
        customer_profile.psychology_traits = psychology_traits
        
        # Step 2: Generate predictions
        conversion_prob = self.predictive_models.predict_conversion_probability(customer_profile, interaction_history)
        churn_risk = self.predictive_models.predict_churn_risk(customer_profile, interaction_history)
        engagement_score = self.predictive_models.predict_engagement_score(customer_profile, interaction_history)
        
        # Update customer profile
        customer_profile.conversion_probability = conversion_prob
        customer_profile.churn_risk = churn_risk
        customer_profile.engagement_score = engagement_score
        
        # Step 3: Generate prescriptive recommendation
        recommendation = self.prescriptive_engine.generate_recommendation(
            customer_profile, interaction_history, conversion_prob, churn_risk
        )
        
        return recommendation

# Example usage and testing
def create_sample_customer_profile() -> CustomerProfile:
    """Create a sample customer profile for testing"""
    return CustomerProfile(
        customer_id="12345",
        demographics={"age": 35, "industry": "technology", "company_size": "mid"},
        preferences={"communication": "email", "meeting_time": "morning"},
        sentiment_score=0.3,
        lead_score=0.7,
        last_interaction=datetime.now() - timedelta(days=5),
        total_interactions=15,
        successful_deals=3,
        avg_deal_value=50000,
        response_rate=0.8,
        preferred_communication="email",
        timezone="EST"
    )

def create_sample_interaction_history() -> List[InteractionHistory]:
    """Create sample interaction history for testing"""
    now = datetime.now()
    return [
        InteractionHistory(
            interaction_id="int1",
            customer_id="12345",
            interaction_type="email",
            timestamp=now - timedelta(days=5),
            content="I need to see the ROI data and cost-benefit analysis before making any decisions.",
            sentiment=0.2,
            outcome="positive",
            response_time_hours=12,
            engagement_level=0.7,
            offer_presented="Standard package",
            discount_offered=0.05
        ),
        InteractionHistory(
            interaction_id="int2",
            customer_id="12345",
            interaction_type="call",
            timestamp=now - timedelta(days=3),
            content="Can you provide a detailed comparison with our current solution?",
            sentiment=0.1,
            outcome="positive",
            response_time_hours=4,
            engagement_level=0.8,
            offer_presented="Premium package"
        ),
        InteractionHistory(
            interaction_id="int3",
            customer_id="12345",
            interaction_type="meeting",
            timestamp=now - timedelta(days=1),
            content="The pricing seems high. What kind of discount can you offer?",
            sentiment=-0.2,
            outcome="neutral",
            response_time_hours=2,
            engagement_level=0.6,
            offer_presented="Negotiated package",
            discount_offered=0.1
        )
    ]

if __name__ == "__main__":
    # Test the prescriptive AI
    prescriptive_ai = PrescriptiveAI()
    
    # Create sample data
    customer_profile = create_sample_customer_profile()
    interaction_history = create_sample_interaction_history()
    
    # Generate recommendation
    recommendation = prescriptive_ai.analyze_customer(customer_profile, interaction_history)
    
    # Print results
    print("=== Prescriptive AI Recommendation ===")
    print(f"Customer ID: {recommendation.customer_id}")
    print(f"Conversion Probability: {recommendation.conversion_probability:.1%}")
    print(f"Churn Risk: {recommendation.churn_risk:.1%}")
    print(f"Recommended Action: {recommendation.recommended_action}")
    print(f"Action Type: {recommendation.action_type.value}")
    print(f"Timeframe: {recommendation.suggested_timeframe}")
    print(f"Priority: {recommendation.priority}")
    print(f"Justification: {recommendation.justification}")
    print(f"Expected Outcome: {recommendation.expected_outcome}")
    print(f"Confidence Score: {recommendation.confidence_score:.1%}")
    print(f"Required Resources: {', '.join(recommendation.required_resources)}")
    print(f"Alternative Actions: {recommendation.alternative_actions}")
    print(f"Success Metrics: {recommendation.success_metrics}")





