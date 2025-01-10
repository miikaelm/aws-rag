from dataclasses import dataclass
from typing import Optional, List
import streamlit as st

@dataclass
class Section:
   title: str
   content: str
   level: int
   url_fragment: str
   id: Optional[int] = None
   url_id: Optional[int] = None
   parent_id: Optional[int] = None
   section_order: Optional[int] = None
   depth: Optional[int] = 0
   path: Optional[str] = None

   @classmethod
   def get_by_url(cls, url_id: int) -> List['Section']:
       """Get all sections for a URL in hierarchical order"""
       with st.session_state.database.get_cursor() as cursor:
           cursor.execute('''
               WITH RECURSIVE section_tree AS (
                   SELECT 
                       id, url_id, parent_id, title, content, level,
                       url_fragment, section_order, 0 as depth,
                       title as path
                   FROM sections 
                   WHERE url_id = ? AND parent_id IS NULL
                   
                   UNION ALL
                   
                   SELECT 
                       s.id, s.url_id, s.parent_id, s.title, s.content, s.level,
                       s.url_fragment, s.section_order,
                       st.depth + 1,
                       st.path || ' > ' || s.title
                   FROM sections s
                   JOIN section_tree st ON s.parent_id = st.id
               )
               SELECT *
               FROM section_tree
               ORDER BY section_order, depth;
           ''', (url_id,))
           
           return [
               cls(
                   id=row[0],
                   url_id=row[1],
                   parent_id=row[2],
                   title=row[3],
                   content=row[4],
                   level=row[5],
                   url_fragment=row[6],
                   section_order=row[7],
                   depth=row[8],
                   path=row[9]
               )
               for row in cursor.fetchall()
           ]

   def save(self) -> None:
       """Save section to database"""
       if not self.url_id:
           raise ValueError("Cannot save section without url_id")

       with st.session_state.database.get_cursor() as cursor:
           if self.id:
               cursor.execute('''
                   UPDATE sections SET 
                       url_id = ?, parent_id = ?, title = ?, content = ?,
                       level = ?, url_fragment = ?, section_order = ?
                   WHERE id = ?
               ''', (
                   self.url_id, self.parent_id, self.title, self.content,
                   self.level, self.url_fragment, self.section_order, self.id
               ))
           else:
               cursor.execute('''
                   INSERT INTO sections (
                       url_id, parent_id, title, content, 
                       level, url_fragment, section_order
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)
               ''', (
                   self.url_id, self.parent_id, self.title, self.content,
                   self.level, self.url_fragment, self.section_order
               ))
               self.id = cursor.lastrowid

   @classmethod
   def delete_by_url(cls, url_id: int) -> None:
       """Delete all sections for a URL"""
       with st.session_state.database.get_cursor() as cursor:
           cursor.execute('DELETE FROM sections WHERE url_id = ?', (url_id,))

   def delete(self) -> None:
       """Delete section and its subsections"""
       if not self.id:
           raise ValueError("Cannot delete section without ID")
           
       with st.session_state.database.get_cursor() as cursor:
           cursor.execute('DELETE FROM sections WHERE id = ?', (self.id,))