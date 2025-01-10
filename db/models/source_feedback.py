from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import streamlit as st

@dataclass
class SourceFeedback:
    message_source_id: int
    rating: int  # 1-5
    id: Optional[int] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if not (1 <= self.rating <= 5):
            raise ValueError("rating must be between 1 and 5")

    def save(self) -> None:
        """Save source feedback to database"""
        with st.session_state.database.get_cursor() as cursor:
            if self.id:
                cursor.execute('''
                    UPDATE source_feedback SET 
                        rating = ?
                    WHERE id = ?
                ''', (self.rating, self.id))
            else:
                cursor.execute('''
                    INSERT INTO source_feedback (
                        message_source_id, rating, created_at
                    ) VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (self.message_source_id, self.rating))
                self.id = cursor.lastrowid