from dataclasses import dataclass
from typing import Optional, List
import streamlit as st

@dataclass
class MessageSource:
   message_id: int
   title: str
   content: str
   url: Optional[str] = None
   relevance_score: Optional[float] = None
   id: Optional[int] = None
   
   @classmethod
   def get_for_message(cls, message_id: int) -> List['MessageSource']:
       """Get all sources for a message"""
       with st.session_state.database.get_cursor() as cursor:
           cursor.execute('''
               SELECT id, message_id, title, url, content, relevance_score
               FROM message_sources
               WHERE message_id = ?
           ''', (message_id,))
           
           return [
               cls(
                   id=row[0],
                   message_id=row[1],
                   title=row[2],
                   url=row[3],
                   content=row[4],
                   relevance_score=row[5]
               )
               for row in cursor.fetchall()
           ]

   def save(self) -> None:
       """Save source to database"""
       with st.session_state.database.get_cursor() as cursor:
           if self.id:
               cursor.execute('''
                   UPDATE message_sources 
                   SET title = ?, url = ?, content = ?, relevance_score = ?
                   WHERE id = ?
               ''', (
                   self.title, self.url, self.content, 
                   self.relevance_score, self.id
               ))
           else:
               cursor.execute('''
                   INSERT INTO message_sources (
                       message_id, title, url, content, relevance_score
                   ) VALUES (?, ?, ?, ?, ?)
               ''', (
                   self.message_id, self.title, self.url, 
                   self.content, self.relevance_score
               ))
               self.id = cursor.lastrowid

   def delete(self) -> None:
       """Delete source from database"""
       if not self.id:
           raise ValueError("Cannot delete source without ID")
           
       with st.session_state.database.get_cursor() as cursor:
           cursor.execute('DELETE FROM message_sources WHERE id = ?', (self.id,))