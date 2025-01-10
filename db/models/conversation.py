from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict
import json
import streamlit as st

@dataclass
class Conversation:
    id: Optional[int] = None
    title: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict] = None
    preview: Optional[str] = None  # Added for latest message preview
    message_count: Optional[int] = None  # Added for message count

    @classmethod
    def get_all(cls) -> List['Conversation']:
        """Get all conversations with their latest messages"""
        with st.session_state.database.get_cursor() as cursor:
            cursor.execute('''
                SELECT 
                    c.id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    c.metadata,
                    m.content as latest_message,
                    COUNT(messages.id) as message_count
                FROM conversations c
                LEFT JOIN messages ON c.id = messages.conversation_id
                LEFT JOIN (
                    SELECT conversation_id, content
                    FROM messages m2
                    WHERE (conversation_id, created_at) IN (
                        SELECT conversation_id, MAX(created_at)
                        FROM messages
                        GROUP BY conversation_id
                    )
                ) m ON c.id = m.conversation_id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
            ''')
            
            conversations = []
            for row in cursor.fetchall():
                preview = row[5][:50] + "..." if row[5] and len(row[5]) > 50 else (row[5] or '')
                conversations.append(cls(
                    id=row[0],
                    title=row[1] or 'New Conversation',
                    created_at=row[2],
                    updated_at=row[3],
                    metadata=json.loads(row[4]) if row[4] else {},
                    preview=preview,
                    message_count=row[6]
                ))
            return conversations