"""
CRM API Endpoints for NLP CRM System
Handles CRM logging and interaction management.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from crm_models import (
    InteractionLog, InteractionStatus, InteractionType, 
    IntentType, SentimentType, CRMStorage
)
from llm_integration import StructuredResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router for CRM endpoints
router = APIRouter(prefix="/api/v1", tags=["CRM"])

# Global CRM storage instance
crm_storage = None

def get_crm_storage():
    """Get CRM storage instance."""
    global crm_storage
    if crm_storage is None:
        crm_storage = CRMStorage()
    return crm_storage

# Pydantic models for API requests/responses
class InteractionLogRequest(BaseModel):
    """Request model for logging interactions."""
    customer_query: str
    assistant_response: str
    response_type: str
    confidence: str
    session_id: str
    user_id: Optional[str] = None
    customer_id: Optional[str] = None
    agent_id: Optional[str] = None
    department: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=5)
    tags: List[str] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_context: Optional[str] = None
    suggested_actions: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    escalation_required: bool = False
    escalation_reason: Optional[str] = None
    user_satisfaction_prediction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    response_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    processing_time_ms: Optional[int] = None
    query_metadata: Optional[Dict[str, Any]] = None
    retrieved_docs_count: Optional[int] = None
    confidence_score: Optional[float] = None
    source_channel: str = "web_api"

class InteractionLogResponse(BaseModel):
    """Response model for interaction logging."""
    success: bool
    interaction_id: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None

class InteractionUpdateRequest(BaseModel):
    """Request model for updating interactions."""
    status: InteractionStatus
    agent_id: Optional[str] = None
    notes: Optional[str] = None

class InteractionResponse(BaseModel):
    """Response model for interaction data."""
    success: bool
    interaction: Optional[InteractionLog] = None
    error: Optional[str] = None

class InteractionsListResponse(BaseModel):
    """Response model for interactions list."""
    success: bool
    interactions: List[InteractionLog] = Field(default_factory=list)
    total_count: int = 0
    error: Optional[str] = None

class CRMStatsResponse(BaseModel):
    """Response model for CRM statistics."""
    success: bool
    stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Helper functions
def _determine_intent_from_response(response_type: str, customer_query: str) -> IntentType:
    """Determine intent from response type and query."""
    query_lower = customer_query.lower()
    
    if "refund" in query_lower or "return" in query_lower:
        return IntentType.REFUND
    elif "shipping" in query_lower or "delivery" in query_lower:
        return IntentType.SHIPPING
    elif "warranty" in query_lower or "guarantee" in query_lower:
        return IntentType.WARRANTY
    elif "password" in query_lower or "login" in query_lower:
        return IntentType.PASSWORD
    elif "payment" in query_lower or "billing" in query_lower:
        return IntentType.PAYMENT
    elif "support" in query_lower or "help" in query_lower:
        return IntentType.SUPPORT
    elif "hello" in query_lower or "hi" in query_lower:
        return IntentType.GREETING
    elif "bye" in query_lower or "goodbye" in query_lower:
        return IntentType.GOODBYE
    elif "complaint" in query_lower or "problem" in query_lower:
        return IntentType.COMPLAINT
    else:
        return IntentType.GENERAL

def _determine_sentiment_from_query(customer_query: str) -> SentimentType:
    """Determine sentiment from customer query."""
    query_lower = customer_query.lower()
    
    positive_words = ["thank", "thanks", "great", "good", "excellent", "love", "happy", "satisfied"]
    negative_words = ["angry", "frustrated", "disappointed", "terrible", "awful", "hate", "complaint", "problem"]
    
    positive_count = sum(1 for word in positive_words if word in query_lower)
    negative_count = sum(1 for word in negative_words if word in query_lower)
    
    if positive_count > negative_count:
        return SentimentType.POSITIVE
    elif negative_count > positive_count:
        return SentimentType.NEGATIVE
    elif positive_count > 0 and negative_count > 0:
        return SentimentType.MIXED
    else:
        return SentimentType.NEUTRAL

def _determine_interaction_type(intent: IntentType, escalation_required: bool) -> InteractionType:
    """Determine interaction type from intent and escalation status."""
    if escalation_required:
        return InteractionType.ESCALATION
    elif intent == IntentType.COMPLAINT:
        return InteractionType.COMPLAINT
    elif intent in [IntentType.SUPPORT, IntentType.PASSWORD, IntentType.WARRANTY]:
        return InteractionType.SUPPORT
    elif intent in [IntentType.PAYMENT, IntentType.REFUND]:
        return InteractionType.SALES
    else:
        return InteractionType.QUERY

# API Endpoints
@router.post("/interactions/log", response_model=InteractionLogResponse)
async def log_interaction(request: InteractionLogRequest):
    """
    Log a customer interaction to the CRM system.
    
    This endpoint receives the final, rich JSON payload from the NLP system
    and stores it in the Interactions collection.
    """
    try:
        storage = get_crm_storage()
        
        # Determine intent, sentiment, and interaction type
        intent = _determine_intent_from_response(request.response_type, request.customer_query)
        sentiment = _determine_sentiment_from_query(request.customer_query)
        interaction_type = _determine_interaction_type(intent, request.escalation_required)
        
        # Determine status based on escalation requirement
        status = InteractionStatus.ESCALATED if request.escalation_required else InteractionStatus.PENDING
        
        # Create interaction log
        interaction = InteractionLog(
            status=status,
            interaction_type=interaction_type,
            intent=intent,
            sentiment=sentiment,
            customer_query=request.customer_query,
            assistant_response=request.assistant_response,
            response_type=request.response_type,
            confidence=request.confidence,
            metadata={
                "session_id": request.session_id,
                "user_id": request.user_id,
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": request.processing_time_ms,
                "query_metadata": request.query_metadata,
                "retrieved_docs_count": request.retrieved_docs_count,
                "confidence_score": request.confidence_score,
                "escalation_reason": request.escalation_reason,
                "source_channel": request.source_channel
            },
            sources=request.sources,
            conversation_context=request.conversation_context,
            customer_id=request.customer_id,
            agent_id=request.agent_id,
            department=request.department,
            priority=request.priority,
            tags=request.tags,
            suggested_actions=request.suggested_actions,
            follow_up_questions=request.follow_up_questions,
            escalation_required=request.escalation_required,
            user_satisfaction_prediction=request.user_satisfaction_prediction,
            response_quality_score=request.response_quality_score
        )
        
        # Log the interaction
        interaction_id = storage.log_interaction(interaction)
        
        logger.info(f"Interaction logged successfully: {interaction_id}")
        
        return InteractionLogResponse(
            success=True,
            interaction_id=interaction_id,
            message="Interaction logged successfully"
        )
        
    except Exception as e:
        logger.error(f"Error logging interaction: {str(e)}")
        return InteractionLogResponse(
            success=False,
            error=str(e)
        )

@router.get("/interactions/{interaction_id}", response_model=InteractionResponse)
async def get_interaction(interaction_id: str):
    """Get a specific interaction by ID."""
    try:
        storage = get_crm_storage()
        interaction = storage.get_interaction(interaction_id)
        
        if not interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")
        
        return InteractionResponse(
            success=True,
            interaction=interaction
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting interaction {interaction_id}: {str(e)}")
        return InteractionResponse(
            success=False,
            error=str(e)
        )

@router.get("/interactions", response_model=InteractionsListResponse)
async def get_interactions(
    status: Optional[InteractionStatus] = Query(None, description="Filter by status"),
    customer_id: Optional[str] = Query(None, description="Filter by customer ID"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of interactions to return")
):
    """Get interactions with optional filtering."""
    try:
        storage = get_crm_storage()
        
        if customer_id:
            interactions = storage.get_interactions_by_customer(customer_id, limit)
        elif status:
            interactions = storage.get_interactions_by_status(status, limit)
        else:
            # Get all interactions (limited)
            interactions = []
            interactions_dir = storage.storage_path / "interactions"
            for interaction_file in interactions_dir.glob("*.json"):
                try:
                    interaction = storage.get_interaction(interaction_file.stem)
                    if interaction:
                        interactions.append(interaction)
                        if len(interactions) >= limit:
                            break
                except Exception as e:
                    logger.warning(f"Error reading interaction file {interaction_file}: {str(e)}")
                    continue
            
            # Sort by created_at descending
            interactions.sort(key=lambda x: x.created_at, reverse=True)
        
        return InteractionsListResponse(
            success=True,
            interactions=interactions,
            total_count=len(interactions)
        )
        
    except Exception as e:
        logger.error(f"Error getting interactions: {str(e)}")
        return InteractionsListResponse(
            success=False,
            error=str(e)
        )

@router.put("/interactions/{interaction_id}/status", response_model=InteractionLogResponse)
async def update_interaction_status(
    interaction_id: str, 
    request: InteractionUpdateRequest
):
    """Update interaction status."""
    try:
        storage = get_crm_storage()
        
        success = storage.update_interaction_status(
            interaction_id=interaction_id,
            status=request.status,
            agent_id=request.agent_id,
            notes=request.notes
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Interaction not found")
        
        return InteractionLogResponse(
            success=True,
            interaction_id=interaction_id,
            message=f"Interaction status updated to {request.status}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating interaction status: {str(e)}")
        return InteractionLogResponse(
            success=False,
            error=str(e)
        )

@router.get("/stats", response_model=CRMStatsResponse)
async def get_crm_stats():
    """Get CRM statistics and analytics."""
    try:
        storage = get_crm_storage()
        stats = storage.get_interaction_stats()
        
        return CRMStatsResponse(
            success=True,
            stats=stats
        )
    except Exception as e:
        logger.error(f"Error getting CRM stats: {str(e)}")
        return CRMStatsResponse(
            success=False,
            error=str(e)
        )

@router.delete("/interactions/{interaction_id}", response_model=InteractionLogResponse)
async def delete_interaction(interaction_id: str):
    """Delete a specific interaction by ID."""
    try:
        storage = get_crm_storage()
        success = storage.delete_interaction(interaction_id)
        if not success:
            raise HTTPException(status_code=404, detail="Interaction not found")
        return InteractionLogResponse(success=True, interaction_id=interaction_id, message="Interaction deleted")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting interaction {interaction_id}: {str(e)}")
        return InteractionLogResponse(success=False, error=str(e))

@router.get("/triage", response_model=InteractionsListResponse)
async def get_triage_view(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of interactions to return")
):
    """Get triage view - interactions that need attention."""
    try:
        storage = get_crm_storage()
        
        # Get escalated and pending interactions
        escalated = storage.get_interactions_by_status(InteractionStatus.ESCALATED, limit // 2)
        pending = storage.get_interactions_by_status(InteractionStatus.PENDING, limit // 2)
        
        # Combine and sort by priority and timestamp
        triage_interactions = escalated + pending
        triage_interactions.sort(
            key=lambda x: (x.priority, x.created_at), 
            reverse=True
        )
        
        return InteractionsListResponse(
            success=True,
            interactions=triage_interactions[:limit],
            total_count=len(triage_interactions)
        )
        
    except Exception as e:
        logger.error(f"Error getting triage view: {str(e)}")
        return InteractionsListResponse(
            success=False,
            error=str(e)
        )
