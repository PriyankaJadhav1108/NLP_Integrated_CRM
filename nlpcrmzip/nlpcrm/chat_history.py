"""
Chat History Management Module for NLP CRM System
Handles conversation history storage and retrieval.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatMessage:
    """Represents a single chat message."""
    
    def __init__(self, role: str, content: str, timestamp: str = None, message_id: str = None):
        """
        Initialize a chat message.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            timestamp: ISO timestamp (auto-generated if None)
            message_id: Unique message ID (auto-generated if None)
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now().isoformat()
        self.message_id = message_id or str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create message from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp"),
            message_id=data.get("message_id")
        )

class ChatSession:
    """Represents a chat session with multiple messages."""
    
    def __init__(self, session_id: str = None, user_id: str = None, created_at: str = None):
        """
        Initialize a chat session.
        
        Args:
            session_id: Unique session ID (auto-generated if None)
            user_id: User identifier
            created_at: ISO timestamp (auto-generated if None)
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id
        self.created_at = created_at or datetime.now().isoformat()
        self.messages: List[ChatMessage] = []
        self.last_updated = self.created_at
    
    def add_message(self, role: str, content: str) -> ChatMessage:
        """
        Add a message to the session.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            
        Returns:
            The created ChatMessage
        """
        message = ChatMessage(role=role, content=content)
        self.messages.append(message)
        self.last_updated = datetime.now().isoformat()
        return message
    
    def get_messages(self, limit: int = None) -> List[ChatMessage]:
        """
        Get messages from the session.
        
        Args:
            limit: Maximum number of messages to return (None for all)
            
        Returns:
            List of ChatMessage objects
        """
        if limit is None:
            return self.messages.copy()
        return self.messages[-limit:] if limit > 0 else []
    
    def get_conversation_context(self, max_messages: int = 10) -> str:
        """
        Get conversation context as formatted string.
        
        Args:
            max_messages: Maximum number of recent messages to include
            
        Returns:
            Formatted conversation context
        """
        recent_messages = self.get_messages(limit=max_messages)
        context_parts = []
        
        for msg in recent_messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{role_label}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "messages": [msg.to_dict() for msg in self.messages]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Create session from dictionary."""
        session = cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            created_at=data.get("created_at")
        )
        session.last_updated = data.get("last_updated", session.created_at)
        session.messages = [ChatMessage.from_dict(msg_data) for msg_data in data.get("messages", [])]
        return session

class ChatHistoryManager:
    """Manages chat history storage and retrieval."""
    
    def __init__(self, storage_path: str = "./chat_history"):
        """
        Initialize the chat history manager.
        
        Args:
            storage_path: Directory to store chat history files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.active_sessions: Dict[str, ChatSession] = {}
        
        logger.info(f"Chat history manager initialized with storage path: {storage_path}")
    
    def create_session(self, user_id: str = None) -> ChatSession:
        """
        Create a new chat session.
        
        Args:
            user_id: User identifier
            
        Returns:
            New ChatSession
        """
        session = ChatSession(user_id=user_id)
        self.active_sessions[session.session_id] = session
        self._save_session(session)
        
        logger.info(f"Created new chat session: {session.session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get a chat session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ChatSession if found, None otherwise
        """
        # Check active sessions first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Try to load from storage
        session = self._load_session(session_id)
        if session:
            self.active_sessions[session_id] = session
        
        return session
    
    def add_message(self, session_id: str, role: str, content: str) -> Optional[ChatMessage]:
        """
        Add a message to a session.
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
            
        Returns:
            ChatMessage if successful, None otherwise
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None
        
        message = session.add_message(role=role, content=content)
        self._save_session(session)
        
        logger.info(f"Added {role} message to session {session_id}")
        return message
    
    def get_conversation_history(self, session_id: str, max_messages: int = 10) -> Optional[str]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            max_messages: Maximum number of messages to include
            
        Returns:
            Formatted conversation history or None if session not found
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        return session.get_conversation_context(max_messages=max_messages)
    
    def get_recent_messages(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent messages from a session.
        
        Args:
            session_id: Session identifier
            limit: Number of recent messages to return
            
        Returns:
            List of message dictionaries
        """
        session = self.get_session(session_id)
        if not session:
            return []
        
        recent_messages = session.get_messages(limit=limit)
        return [msg.to_dict() for msg in recent_messages]
    
    def _save_session(self, session: ChatSession):
        """Save session to storage."""
        try:
            file_path = self.storage_path / f"{session.session_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {str(e)}")
    
    def _load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load session from storage."""
        try:
            file_path = self.storage_path / f"{session_id}.json"
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return ChatSession.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {str(e)}")
            return None
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """
        Clean up old session files.
        
        Args:
            days_old: Number of days after which to delete sessions
        """
        try:
            cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            for file_path in self.storage_path.glob("*.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    logger.info(f"Deleted old session file: {file_path.name}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about stored sessions."""
        try:
            session_files = list(self.storage_path.glob("*.json"))
            total_sessions = len(session_files)
            active_sessions = len(self.active_sessions)
            
            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "storage_path": str(self.storage_path)
            }
        except Exception as e:
            logger.error(f"Error getting session stats: {str(e)}")
            return {"error": str(e)}
