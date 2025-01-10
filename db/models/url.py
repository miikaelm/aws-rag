from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import streamlit as st
import sqlite3

@dataclass
class URL():
    url: str
    description: str
    added_date: datetime
    id: Optional[int] = None
    last_scraped: Optional[datetime] = None

    @classmethod
    def add(cls, url: str, description: str) -> Optional[URL]:
        """Add a new URL to the database"""
        try:
            with st.session_state.database.get_cursor() as cursor:
                cursor.execute(
                    'INSERT INTO urls (url, description, added_date) VALUES (?, ?, ?)',
                    (url, description, datetime.now())
                )
                return cls(url=url, description=description, added_date=datetime.now(), id=cursor.lastrowid)
        except sqlite3.IntegrityError:
            st.error("This URL already exists in the database!")
            return None

    @classmethod
    def get_all(cls) -> List[URL]:
        """Retrieve all URLs from the database"""
        with st.session_state.database.get_cursor() as cursor:
            cursor.execute('''
                SELECT id, url, description, added_date, last_scraped 
                FROM urls 
                ORDER BY added_date DESC
            ''')
            return [
                cls(
                    id=row[0],
                    url=row[1],
                    description=row[2],
                    added_date=row[3],
                    last_scraped=row[4]
                )
                for row in cursor.fetchall()
            ]

    def delete(self) -> None:
        """Delete URL from database"""
        if not self.id:
            raise ValueError("Cannot delete URL without ID")
        with st.session_state.database.get_cursor() as cursor:
            cursor.execute('DELETE FROM urls WHERE id = ?', (self.id,))

    def update_last_scraped(self) -> None:
        """Update last_scraped timestamp"""
        if not self.id:
            raise ValueError("Cannot update URL without ID")
        with st.session_state.database.get_cursor() as cursor:
            cursor.execute(
                'UPDATE urls SET last_scraped = ? WHERE id = ?',
                (datetime.now(), self.id)
            )