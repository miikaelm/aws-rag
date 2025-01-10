from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from typing import Generator
import streamlit as st
from .schema import CREATE_TABLES_SQL
 
class DatabaseConnection:
    def __init__(self, db_path: str = 'rag_settings.db'):
        self.db_path = db_path
        self.init_db()  # Initialize schema on connection creation

    def init_db(self) -> None:
        """Initialize database with schema"""
        with self.get_cursor() as cursor:
            for table_sql in CREATE_TABLES_SQL.values():
                cursor.execute(table_sql)
                
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    @contextmanager
    def get_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database cursors"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e

def get_db() -> DatabaseConnection:
    """Get or create Database instance from session state"""
    if 'database' not in st.session_state:
        st.session_state.database = DatabaseConnection()
    return st.session_state.database