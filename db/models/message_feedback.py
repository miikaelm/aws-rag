from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import streamlit as st

@dataclass
class MessageFeedback:
   message_id: int
   answer_relevance: int  # 1-5
   answer_accuracy: int   # 1-5
   id: Optional[int] = None
   feedback_text: Optional[str] = None
   created_at: Optional[datetime] = None

   def __post_init__(self):
       if not (1 <= self.answer_relevance <= 5):
           raise ValueError("answer_relevance must be between 1 and 5")
       if not (1 <= self.answer_accuracy <= 5):
           raise ValueError("answer_accuracy must be between 1 and 5")

   def save(self) -> None:
       """Save feedback to database"""
       with st.session_state.database.get_cursor() as cursor:
           if self.id:
               cursor.execute('''
                   UPDATE message_feedback SET 
                       answer_relevance = ?, answer_accuracy = ?,
                       feedback_text = ?
                   WHERE id = ?
               ''', (
                   self.answer_relevance, self.answer_accuracy,
                   self.feedback_text, self.id
               ))
           else:
               cursor.execute('''
                   INSERT INTO message_feedback (
                       message_id, answer_relevance, answer_accuracy,
                       feedback_text, created_at
                   ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
               ''', (
                   self.message_id, self.answer_relevance, 
                   self.answer_accuracy, self.feedback_text
               ))
               self.id = cursor.lastrowid