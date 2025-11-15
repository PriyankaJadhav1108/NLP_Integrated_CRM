"""
CRM Database Models for NLP CRM System
Handles CRM data storage and interaction logging.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractionStatus(str, Enum):
    """Status of customer interactions."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    CLOSED = "closed"

class InteractionType(str, Enum):
    """Types of customer interactions."""
    QUERY = "query"
    COMPLAINT = "complaint"
    SUPPORT = "support"
    SALES = "sales"
    FEEDBACK = "feedback"
    ESCALATION = "escalation"

class SentimentType(str, Enum):
    """Sentiment analysis results."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"

class IntentType(str, Enum):
    """Customer intent types."""
    REFUND = "refund"
    SHIPPING = "shipping"
    WARRANTY = "warranty"
    PASSWORD = "password"
    PAYMENT = "payment"
    SUPPORT = "support"
    GREETING = "greeting"
    GOODBYE = "goodbye"
    COMPLAINT = "complaint"
    GENERAL = "general"

class InteractionMetadata(BaseModel):
    """Metadata for customer interactions."""
    session_id: str
    user_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    processing_time_ms: Optional[int] = None
    query_metadata: Optional[Dict[str, Any]] = None
    retrieved_docs_count: Optional[int] = None
    confidence_score: Optional[float] = None
    escalation_reason: Optional[str] = None
    source_channel: str = "web_api"  # web_api, mobile, phone, email, etc.

class InteractionLog(BaseModel):
    """Complete interaction log entry."""
    interaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: InteractionStatus = InteractionStatus.PENDING
    interaction_type: InteractionType = InteractionType.QUERY
    intent: IntentType = IntentType.GENERAL
    sentiment: SentimentType = SentimentType.NEUTRAL
    
    # Core interaction data
    customer_query: str
    assistant_response: str
    response_type: str
    confidence: str
    
    # Metadata
    metadata: InteractionMetadata
    
    # Sources and context
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_context: Optional[str] = None
    
    # CRM specific fields
    customer_id: Optional[str] = None
    agent_id: Optional[str] = None
    department: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=5)  # 1=low, 5=high
    tags: List[str] = Field(default_factory=list)
    
    # Follow-up and actions
    suggested_actions: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    escalation_required: bool = False
    
    # Analytics
    user_satisfaction_prediction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    response_quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    # Timestamps
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    resolved_at: Optional[str] = None

class CRMStorage:
    """Handles CRM data storage and retrieval."""
    
    def __init__(self, storage_path: str = "./crm_data"):
        """
        Initialize CRM storage.
        
        Args:
            storage_path: Directory to store CRM data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.storage_path / "interactions").mkdir(exist_ok=True)
        (self.storage_path / "customers").mkdir(exist_ok=True)
        (self.storage_path / "agents").mkdir(exist_ok=True)
        
        logger.info(f"CRM storage initialized at: {storage_path}")
    
    def log_interaction(self, interaction: InteractionLog) -> str:
        """
        Log a customer interaction.
        
        Args:
            interaction: InteractionLog object
            
        Returns:
            Interaction ID
        """
        try:
            # Save interaction to file
            interaction_file = self.storage_path / "interactions" / f"{interaction.interaction_id}.json"
            
            with open(interaction_file, 'w', encoding='utf-8') as f:
                json.dump(interaction.dict(), f, indent=2, ensure_ascii=False)
            
            # Update customer record if customer_id exists
            if interaction.customer_id:
                self._update_customer_record(interaction)
            
            logger.info(f"Interaction logged: {interaction.interaction_id}")
            return interaction.interaction_id
            
        except Exception as e:
            logger.error(f"Error logging interaction: {str(e)}")
            raise
    
    def get_interaction(self, interaction_id: str) -> Optional[InteractionLog]:
        """
        Get an interaction by ID.
        
        Args:
            interaction_id: Interaction identifier
            
        Returns:
            InteractionLog if found, None otherwise
        """
        try:
            interaction_file = self.storage_path / "interactions" / f"{interaction_id}.json"
            
            if not interaction_file.exists():
                return None
            
            with open(interaction_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return InteractionLog(**data)
            
        except Exception as e:
            logger.error(f"Error getting interaction {interaction_id}: {str(e)}")
            return None
    
    def get_interactions_by_customer(self, customer_id: str, limit: int = 50) -> List[InteractionLog]:
        """
        Get interactions for a specific customer.
        
        Args:
            customer_id: Customer identifier
            limit: Maximum number of interactions to return
            
        Returns:
            List of InteractionLog objects
        """
        try:
            interactions = []
            interactions_dir = self.storage_path / "interactions"
            
            for interaction_file in interactions_dir.glob("*.json"):
                try:
                    with open(interaction_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if data.get("customer_id") == customer_id:
                        interactions.append(InteractionLog(**data))
                        
                        if len(interactions) >= limit:
                            break
                            
                except Exception as e:
                    logger.warning(f"Error reading interaction file {interaction_file}: {str(e)}")
                    continue
            
            # Sort by created_at descending
            interactions.sort(key=lambda x: x.created_at, reverse=True)
            return interactions
            
        except Exception as e:
            logger.error(f"Error getting interactions for customer {customer_id}: {str(e)}")
            return []
    
    def get_interactions_by_status(self, status: InteractionStatus, limit: int = 100) -> List[InteractionLog]:
        """
        Get interactions by status.
        
        Args:
            status: Interaction status
            limit: Maximum number of interactions to return
            
        Returns:
            List of InteractionLog objects
        """
        try:
            interactions = []
            interactions_dir = self.storage_path / "interactions"
            
            for interaction_file in interactions_dir.glob("*.json"):
                try:
                    with open(interaction_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if data.get("status") == status.value:
                        interactions.append(InteractionLog(**data))
                        
                        if len(interactions) >= limit:
                            break
                            
                except Exception as e:
                    logger.warning(f"Error reading interaction file {interaction_file}: {str(e)}")
                    continue
            
            # Sort by created_at descending
            interactions.sort(key=lambda x: x.created_at, reverse=True)
            return interactions
            
        except Exception as e:
            logger.error(f"Error getting interactions by status {status}: {str(e)}")
            return []
    
    def get_interaction_stats(self) -> Dict[str, Any]:
        """
        Get interaction statistics.
        
        Returns:
            Dictionary with interaction statistics
        """
        try:
            interactions_dir = self.storage_path / "interactions"
            interaction_files = list(interactions_dir.glob("*.json"))
            
            stats = {
                "total_interactions": len(interaction_files),
                "by_status": {},
                "by_intent": {},
                "by_sentiment": {},
                "by_type": {},
                "escalation_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_processing_time": 0.0
            }
            
            total_confidence = 0.0
            total_processing_time = 0.0
            escalated_count = 0
            
            for interaction_file in interaction_files:
                try:
                    with open(interaction_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Count by status
                    status = data.get("status", "unknown")
                    stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
                    
                    # Count by intent
                    intent = data.get("intent", "unknown")
                    stats["by_intent"][intent] = stats["by_intent"].get(intent, 0) + 1
                    
                    # Count by sentiment
                    sentiment = data.get("sentiment", "unknown")
                    stats["by_sentiment"][sentiment] = stats["by_sentiment"].get(sentiment, 0) + 1
                    
                    # Count by type
                    interaction_type = data.get("interaction_type", "unknown")
                    stats["by_type"][interaction_type] = stats["by_type"].get(interaction_type, 0) + 1
                    
                    # Escalation tracking
                    if data.get("escalation_required", False):
                        escalated_count += 1
                    
                    # Confidence tracking
                    confidence = data.get("metadata", {}).get("confidence_score")
                    if confidence is not None:
                        total_confidence += confidence
                    
                    # Processing time tracking
                    processing_time = data.get("metadata", {}).get("processing_time_ms")
                    if processing_time is not None:
                        total_processing_time += processing_time
                        
                except Exception as e:
                    logger.warning(f"Error reading interaction file {interaction_file}: {str(e)}")
                    continue
            
            # Calculate averages
            if len(interaction_files) > 0:
                stats["escalation_rate"] = escalated_count / len(interaction_files)
                stats["avg_confidence"] = total_confidence / len(interaction_files) if total_confidence > 0 else 0.0
                stats["avg_processing_time"] = total_processing_time / len(interaction_files) if total_processing_time > 0 else 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting interaction stats: {str(e)}")
            return {"error": str(e)}
    
    def _update_customer_record(self, interaction: InteractionLog):
        """Update customer record with interaction data."""
        try:
            customer_file = self.storage_path / "customers" / f"{interaction.customer_id}.json"
            
            # Load existing customer data or create new
            if customer_file.exists():
                with open(customer_file, 'r', encoding='utf-8') as f:
                    customer_data = json.load(f)
            else:
                customer_data = {
                    "customer_id": interaction.customer_id,
                    "created_at": datetime.now().isoformat(),
                    "interactions": [],
                    "total_interactions": 0,
                    "last_interaction": None
                }
            
            # Update customer record
            customer_data["interactions"].append(interaction.interaction_id)
            customer_data["total_interactions"] = len(customer_data["interactions"])
            customer_data["last_interaction"] = interaction.created_at
            customer_data["updated_at"] = datetime.now().isoformat()
            
            # Save updated customer data
            with open(customer_file, 'w', encoding='utf-8') as f:
                json.dump(customer_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error updating customer record: {str(e)}")
    
    def update_interaction_status(self, interaction_id: str, status: InteractionStatus, 
                                agent_id: str = None, notes: str = None) -> bool:
        """
        Update interaction status.
        
        Args:
            interaction_id: Interaction identifier
            status: New status
            agent_id: Agent handling the interaction
            notes: Additional notes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            interaction = self.get_interaction(interaction_id)
            if not interaction:
                return False
            
            # Update interaction
            interaction.status = status
            interaction.updated_at = datetime.now().isoformat()
            
            if agent_id:
                interaction.agent_id = agent_id
            
            if status == InteractionStatus.RESOLVED:
                interaction.resolved_at = datetime.now().isoformat()
            
            # Save updated interaction
            interaction_file = self.storage_path / "interactions" / f"{interaction_id}.json"
            with open(interaction_file, 'w', encoding='utf-8') as f:
                json.dump(interaction.dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Updated interaction {interaction_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating interaction status: {str(e)}")
            return False

    def delete_interaction(self, interaction_id: str) -> bool:
        """Delete an interaction by ID from storage."""
        try:
            interaction_file = self.storage_path / "interactions" / f"{interaction_id}.json"
            if not interaction_file.exists():
                return False
            # Best-effort load to update customer record references in future (skipped for now)
            try:
                interaction = None
                with open(interaction_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    interaction = data
            except Exception:
                interaction = None
            interaction_file.unlink()
            logger.info(f"Deleted interaction {interaction_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting interaction {interaction_id}: {str(e)}")
            return False