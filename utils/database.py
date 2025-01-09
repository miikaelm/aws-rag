from __future__ import annotations
import sqlite3
from datetime import datetime
from contextlib import contextmanager
import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, List, Generator, Any, TypeVar, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin

# Type variables for generic database operations
T = TypeVar('T')

# Models
@dataclass
class Section:
    title: str
    content: str
    level: int
    url_fragment: str
    subsections: List['Section']
    id: Optional[int] = None
    parent_id: Optional[int] = None
    section_order: Optional[int] = None
    depth: Optional[int] = 0
    path: Optional[str] = None

@dataclass
class URL:
    url: str
    description: str
    added_date: datetime
    id: Optional[int] = None
    last_scraped: Optional[datetime] = None

@dataclass
class ContentWarning:
    level: str
    message: str
    title: str
    content_length: int

class DatabaseConnection:
    def __init__(self, db_path: str = 'rag_settings.db'):
        self.db_path = db_path
    
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

class Schema:
    """Database schema management"""
    
    CREATE_TABLES_SQL = {
        'urls': '''
            CREATE TABLE IF NOT EXISTS urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL UNIQUE,
                description TEXT,
                added_date TIMESTAMP,
                last_scraped TIMESTAMP NULL
            )
        ''',
        'content': '''
            CREATE TABLE IF NOT EXISTS content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER,
                title TEXT,
                content TEXT,
                scraped_date TIMESTAMP,
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE
            )
        ''',
        'sections': '''
            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER,
                parent_id INTEGER NULL,
                title TEXT,
                content TEXT,
                level INTEGER,
                url_fragment TEXT,
                section_order INTEGER,
                FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE,
                FOREIGN KEY (parent_id) REFERENCES sections (id) ON DELETE CASCADE
            )
        '''
    }
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
    
    def init_db(self) -> None:
        """Initialize database schema"""
        with self.db.get_cursor() as cursor:
            for table_sql in self.CREATE_TABLES_SQL.values():
                cursor.execute(table_sql)

class ContentProcessor:
    """Handles content processing and warnings"""
    
    WARNING_THRESHOLD = 1700  # Current max
    MEDIUM_THRESHOLD = 2500   # Getting large
    LARGE_THRESHOLD = 4000    # Needs attention
    
    @staticmethod
    def check_content_length(content: str, title: str) -> Optional[ContentWarning]:
        """Check content length and return warning if needed"""
        content_length = len(content)
        
        if content_length > ContentProcessor.LARGE_THRESHOLD:
            return ContentWarning(
                level='high',
                message=f"🚨 Section '{title}' is very large ({content_length} chars) and may need chunking",
                title=title,
                content_length=content_length
            )
        elif content_length > ContentProcessor.MEDIUM_THRESHOLD:
            return ContentWarning(
                level='medium',
                message=f"⚠️ Section '{title}' is large ({content_length} chars) and may need chunking",
                title=title,
                content_length=content_length
            )
        elif content_length > ContentProcessor.WARNING_THRESHOLD:
            return ContentWarning(
                level='low',
                message=f"ℹ️ Section '{title}' is approaching size limit ({content_length} chars)",
                title=title,
                content_length=content_length
            )
        return None

class DocumentScraper:
    """Handles document scraping and section extraction"""
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
    
    def scrape_url(self, url: str) -> Optional[Tuple[str, List[Section]]]:
        """Scrape content from a URL, maintaining document hierarchy"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()

            # Extract title
            title = soup.title.string if soup.title else url
            
            # Extract hierarchical sections
            sections = self._extract_sections(soup, url)
            
            return title, sections
            
        except Exception as e:
            st.error(f"Error scraping {url}: {str(e)}")
            return None
    
    def _extract_sections(self, soup: BeautifulSoup, base_url: str) -> List[Section]:
        """Extract hierarchical sections from AWS documentation"""
        main_content = soup.find('div', {'id': 'main-content'}) or soup.find('main')
        if not main_content:
            return []

        sections = []
        section_stack = []
        
        for header in main_content.find_all(['h1', 'h2', 'h3']):
            section = self._process_header(header, base_url)
            
            # Handle hierarchy
            while section_stack and section_stack[-1].level >= section.level:
                section_stack.pop()

            if section_stack:
                section_stack[-1].subsections.append(section)
            else:
                sections.append(section)

            section_stack.append(section)

        return sections
    
    def _process_header(self, header: BeautifulSoup, base_url: str) -> Section:
        """Process a single header element and its content"""
        level = int(header.name[1])
        url_fragment = self._get_header_id(header)
        content = self._extract_content(header)
        title = header.get_text(strip=True)
        
        # Check content length and add warning if needed
        warning = ContentProcessor.check_content_length(content, title)
        if warning:
            if 'content_warnings' not in st.session_state:
                st.session_state.content_warnings = []
            st.session_state.content_warnings.append(warning)
        
        return Section(
            title=title,
            content=content,
            level=level,
            url_fragment=url_fragment,
            subsections=[]
        )
    
    def _get_header_id(self, header: BeautifulSoup) -> str:
        """Extract header ID or find nearby ID"""
        url_fragment = header.get('id', '')
        if not url_fragment and header.get('class'):
            nearby_id = header.find_parent(class_='awsdocs-section')
            if nearby_id:
                url_fragment = nearby_id.get('id', '')
        return url_fragment
    
    def _extract_content(self, header: BeautifulSoup) -> str:
        """Extract content following a header until the next header"""
        content_elements = []
        current_element = header.find_next_sibling()
        
        while current_element and current_element.name not in ['h1', 'h2', 'h3']:
            if content := self._process_element(current_element):
                content_elements.append(content)
            current_element = current_element.find_next_sibling()
        
        return '\n'.join(content_elements)
    
    def _process_element(self, element: BeautifulSoup) -> Optional[str]:
        """Process a single content element"""
        if element.name == 'p':
            text = ' '.join(element.get_text().split())
            return text if text else None
            
        elif element.name in ['ul', 'ol']:
            list_items = []
            for li in element.find_all('li'):
                text = ' '.join(li.get_text().split())
                if text:
                    list_items.append(f"• {text}")
            return '\n'.join(list_items) if list_items else None
            
        elif element.name == 'pre':
            code = element.get_text(strip=True)
            return f"```\n{code}\n```" if code else None
            
        elif element.name == 'code':
            code = element.get_text(strip=True)
            return f"`{code}`" if code else None
            
        return None

class Database:
    """Main database interface"""
    
    def __init__(self):
        self.db = DatabaseConnection()
        self.schema = Schema(self.db)
        self.scraper = DocumentScraper(self.db)
    
    def initialize(self) -> None:
        """Initialize database"""
        self.schema.init_db()
    
    def add_url(self, url: str, description: str) -> bool:
        """Add a new URL to the database"""
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute(
                    'INSERT INTO urls (url, description, added_date) VALUES (?, ?, ?)',
                    (url, description, datetime.now())
                )
            return True
        except sqlite3.IntegrityError:
            st.error("This URL already exists in the database!")
            return False
    
    def get_urls(self) -> List[URL]:
        """Retrieve all URLs from the database"""
        with self.db.get_cursor() as cursor:
            cursor.execute('''
                SELECT id, url, description, added_date, last_scraped 
                FROM urls 
                ORDER BY added_date DESC
            ''')
            return [
                URL(
                    id=row[0],
                    url=row[1],
                    description=row[2],
                    added_date=row[3],
                    last_scraped=row[4]
                )
                for row in cursor.fetchall()
            ]
    
    def delete_url(self, url_id: int) -> None:
        """Delete a URL from the database"""
        with self.db.get_cursor() as cursor:
            cursor.execute('DELETE FROM urls WHERE id = ?', (url_id,))
    
    def save_sections(self, url_id: int, title: str, sections: List[Section]) -> bool:
        """Save scraped sections to database"""
        try:
            with self.db.get_cursor() as cursor:
                # Update last_scraped timestamp
                cursor.execute(
                    'UPDATE urls SET last_scraped = ? WHERE id = ?',
                    (datetime.now(), url_id)
                )
                
                # Clear old sections
                cursor.execute('DELETE FROM sections WHERE url_id = ?', (url_id,))
                
                # Save new sections
                for i, section in enumerate(sections):
                    self._save_section_recursive(cursor, url_id, section, None, i)
                
            return True
            
        except Exception as e:
            st.error(f"Error saving content: {str(e)}")
            return False
    
    def _save_section_recursive(
        self, 
        cursor: sqlite3.Cursor, 
        url_id: int, 
        section: Section, 
        parent_id: Optional[int] = None, 
        order: int = 0
    ) -> None:
        """Recursively save a section and its subsections"""
        cursor.execute('''
            INSERT INTO sections (
                url_id, parent_id, title, content, 
                level, url_fragment, section_order
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            url_id, parent_id, section.title, section.content,
            section.level, section.url_fragment, order
        ))
        
        section_id = cursor.lastrowid
        
        for i, subsection in enumerate(section.subsections):
            self._save_section_recursive(cursor, url_id, subsection, section_id, i)
    
    def get_sections(self, url_id: int) -> List[Section]:
        """Retrieve hierarchical sections for a URL"""
        with self.db.get_cursor() as cursor:
            cursor.execute('''
                WITH RECURSIVE section_tree AS (
                    SELECT 
                        id, parent_id, title, content, level, url_fragment,
                        section_order, 0 as depth,
                        title as path
                    FROM sections 
                    WHERE url_id = ? AND parent_id IS NULL
                    
                    UNION ALL
                    
                    SELECT 
                        s.id, s.parent_id, s.title, s.content, s.level,
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
            
            columns = [description[0] for description in cursor.description]
            return [
                Section(
                    id=row[columns.index('id')],
                    parent_id=row[columns.index('parent_id')],
                    title=row[columns.index('title')],
                    content=row[columns.index('content')],
                    level=row[columns.index('level')],
                    url_fragment=row[columns.index('url_fragment')],
                    section_order=row[columns.index('section_order')],
                    depth=row[columns.index('depth')],
                    path=row[columns.index('path')],
                    subsections=[]
                )
                for row in cursor.fetchall()
            ]

# Initialize database helper
def get_database() -> Database:
    """Get or create Database instance from session state"""
    if 'database' not in st.session_state:
        st.session_state.database = Database()
        st.session_state.database.initialize()
    return st.session_state.database