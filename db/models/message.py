from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict
import streamlit as st
from .message_source import MessageSource

@dataclass
class Message:
    conversation_id: int
    role: str
    content: str
    id: Optional[int] = None
    model_version: Optional[str] = None
    confidence: Optional[float] = None
    message_order: Optional[int] = None
    created_at: Optional[datetime] = None
    sources: List['MessageSource'] = field(default_factory=list)

    @classmethod
    def get_conversation_messages(cls, conversation_id: int) -> List['Message']:
        """Get all messages in a conversation with their sources"""
        with st.session_state.database.get_cursor() as cursor:
            cursor.execute('''
                SELECT 
                    id, conversation_id, role, content,
                    model_version, confidence, created_at,
                    message_order
                FROM messages
                WHERE conversation_id = ?
                ORDER BY message_order
            ''', (conversation_id,))
            
            messages = []
            for row in cursor.fetchall():
                message = cls(
                    id=row[0],
                    conversation_id=row[1],
                    role=row[2],
                    content=row[3],
                    model_version=row[4],
                    confidence=row[5],
                    created_at=row[6],
                    message_order=row[7]
                )
                # Get sources for this message
                message.sources = MessageSource.get_for_message(message.id)
                messages.append(message)
            return messages

    def has_feedback(self) -> bool:
        """Check if feedback exists for this message"""
        if not self.id:
            return False
            
        with st.session_state.database.get_cursor() as cursor:
            cursor.execute('''
                SELECT id FROM message_feedback 
                WHERE message_id = ?
            ''', (self.id,))
            return cursor.fetchone() is not None

    def get_feedback(self) -> Optional[Dict]:
        """Get existing feedback for this message"""
        if not self.id:
            return None
            
        with st.session_state.database.get_cursor() as cursor:
            cursor.execute('''
                SELECT answer_relevance, answer_accuracy, feedback_text
                FROM message_feedback 
                WHERE message_id = ?
            ''', (self.id,))
            row = cursor.fetchone()
            if row:
                return {
                    'relevance': row[0],
                    'accuracy': row[1],
                    'feedback_text': row[2] or ''
                }
            return None

    def get_source_feedback(self) -> Dict[int, int]:
        """Get existing source feedback for this message's sources"""
        if not self.id:
            return {}
            
        source_ratings = {}
        with st.session_state.database.get_cursor() as cursor:
            cursor.execute('''
                SELECT ms.id, sf.rating
                FROM message_sources ms
                LEFT JOIN source_feedback sf ON ms.id = sf.message_source_id
                WHERE ms.message_id = ?
            ''', (self.id,))
            for row in cursor.fetchall():
                if row[1] is not None:  # if there's a rating
                    source_ratings[row[0]] = row[1]
        return source_ratings