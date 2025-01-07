import sqlite3
from datetime import datetime
import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, List
from dataclasses import dataclass
from urllib.parse import urljoin

@dataclass
class Section:
    title: str
    content: str
    level: int
    url_fragment: str
    subsections: List['Section']

def init_db():
    """Initialize SQLite database with both urls and content tables"""
    conn = sqlite3.connect('rag_settings.db')
    c = conn.cursor()
    
    # URLs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS urls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL UNIQUE,
            description TEXT,
            added_date TIMESTAMP,
            last_scraped TIMESTAMP NULL
        )
    ''')
    
    # Content table
    c.execute('''
        CREATE TABLE IF NOT EXISTS content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url_id INTEGER,
            title TEXT,
            content TEXT,
            scraped_date TIMESTAMP,
            FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE
        )
    ''')

    # Sections table for hierarchical content
    c.execute('''
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
    ''')
    
    conn.commit()
    conn.close()

def extract_sections(soup, base_url: str) -> List[Section]:
    """Extract hierarchical sections from AWS documentation"""
    sections = []
    main_content = soup.find('div', {'id': 'main-content'}) or soup.find('main')
    
    if not main_content:
        return []

    # AWS docs typically use h1, h2, h3 for hierarchy
    headers = main_content.find_all(['h1', 'h2', 'h3'])
    current_section = None
    section_stack = []

    for header in headers:
        # Get the header level (1, 2, or 3)
        level = int(header.name[1])
        
        # Get the header ID or create one from text
        url_fragment = header.get('id', '')
        if not url_fragment and header.get('class'):
            # AWS often puts IDs in nearby elements
            nearby_id = header.find_parent(class_='awsdocs-section')
            if nearby_id:
                url_fragment = nearby_id.get('id', '')
        
        # Create section content by getting all text until next header
        content_elements = []
        current_element = header.find_next_sibling()
        while current_element and current_element.name not in ['h1', 'h2', 'h3']:
            if current_element.name == 'p':
                # Clean and normalize whitespace in paragraphs
                text = ' '.join(current_element.get_text().split())
                if text:  # Only add non-empty paragraphs
                    content_elements.append(text)
            elif current_element.name in ['ul', 'ol']:
                list_items = []
                for li in current_element.find_all('li'):
                    # Clean whitespace in list items
                    text = ' '.join(li.get_text().split())
                    if text:
                        list_items.append(f"• {text}")
                if list_items:  # Only add non-empty lists
                    content_elements.extend(list_items)
            elif current_element.name == 'pre':
                # Preserve whitespace in code blocks
                code = current_element.get_text(strip=True)
                if code:
                    content_elements.append(f"```\n{code}\n```")
            elif current_element.name == 'code':
                # Inline code
                code = current_element.get_text(strip=True)
                if code:
                    content_elements.append(f"`{code}`")

            current_element = current_element.find_next_sibling()
        
        content = '\n'.join(content_elements)
        
        handle_content_length_warning(content)
        # Create new section
        new_section = Section(
            title=header.get_text(strip=True),
            content=content,
            level=level,
            url_fragment=url_fragment,
            subsections=[]
        )

        # Handle hierarchy
        while section_stack and section_stack[-1].level >= level:
            section_stack.pop()

        if section_stack:
            section_stack[-1].subsections.append(new_section)
        else:
            sections.append(new_section)

        section_stack.append(new_section)

    return sections

def handle_content_length_warning(content: str):
    
    WARNING_THRESHOLD = 1700  # Current max
    MEDIUM_THRESHOLD = 2500   # Getting large
    LARGE_THRESHOLD = 4000    # Needs attention

    if len(content) > LARGE_THRESHOLD:
        st.warning("This content is very large and may not be processed correctly. Please consider breaking it down into smaller chunks.")
    elif len(content) > MEDIUM_THRESHOLD:
        st.warning("This content is large and may not be processed correctly. Please consider breaking it down into smaller chunks.")
    elif len(content) > WARNING_THRESHOLD:
        st.warning("This content is getting close to the limit and may not be processed correctly. Please consider breaking it down into smaller chunks.")

def save_section(cursor, url_id: int, section: Section, parent_id: Optional[int] = None, order: int = 0):
    """Recursively save a section and its subsections to the database"""
    cursor.execute('''
        INSERT INTO sections (url_id, parent_id, title, content, level, url_fragment, section_order)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (url_id, parent_id, section.title, section.content, section.level, section.url_fragment, order))
    
    section_id = cursor.lastrowid
    
    for i, subsection in enumerate(section.subsections):
        save_section(cursor, url_id, subsection, section_id, i)

def scrape_url(url: str) -> Optional[tuple[str, List[Section]]]:
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
        sections = extract_sections(soup, url)
        
        return title, sections
        
    except Exception as e:
        st.error(f"Error scraping {url}: {str(e)}")
        return None

def save_scraped_content(url_id: int, title: str, sections: List[Section]):
    """Save scraped hierarchical content to database"""
    conn = sqlite3.connect('rag_settings.db')
    c = conn.cursor()
    
    try:
        # Update last_scraped timestamp
        c.execute(
            'UPDATE urls SET last_scraped = ? WHERE id = ?',
            (datetime.now(), url_id)
        )
        
        # Clear old sections
        c.execute('DELETE FROM sections WHERE url_id = ?', (url_id,))
        
        # Save new sections
        for i, section in enumerate(sections):
            save_section(c, url_id, section, None, i)
        
        conn.commit()
        return True
        
    except Exception as e:
        st.error(f"Error saving content: {str(e)}")
        return False
        
    finally:
        conn.close()

def get_sections(url_id: int) -> List[Dict]:
    """Retrieve hierarchical content for a URL"""
    conn = sqlite3.connect('rag_settings.db')
    c = conn.cursor()
    try:
        # Get all sections for the URL
        c.execute('''
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
            SELECT 
                id,
                parent_id,
                title,
                content,
                level,
                url_fragment,
                section_order,
                depth,
                path
            FROM section_tree
            ORDER BY section_order, depth;
        ''', (url_id,))
        
        # Get column names from cursor description
        columns = [description[0] for description in c.description]
        
        # Convert each row to a dictionary with column names
        results = []
        for row in c.fetchall():
            results.append(dict(zip(columns, row)))
            
        return results
    finally:
        conn.close()
         
def add_url(url: str, description: str):
    """Add a new URL to the database"""
    conn = sqlite3.connect('rag_settings.db')
    c = conn.cursor()
    try:
        c.execute(
            'INSERT INTO urls (url, description, added_date) VALUES (?, ?, ?)',
            (url, description, datetime.now())
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        st.error("This URL already exists in the database!")
        return False
    finally:
        conn.close()

def delete_url(url_id: int):
    """Delete a URL from the database"""
    conn = sqlite3.connect('rag_settings.db')
    c = conn.cursor()
    c.execute('DELETE FROM urls WHERE id = ?', (url_id,))
    conn.commit()
    conn.close()

def get_urls():
    """Retrieve all URLs from the database"""
    conn = sqlite3.connect('rag_settings.db')
    c = conn.cursor()
    try:
        c.execute('''
            SELECT id, url, description, added_date, last_scraped 
            FROM urls 
            ORDER BY added_date DESC
        ''')
        urls = c.fetchall()
        return urls
    except Exception as e:
        st.error(f"Error retrieving URLs: {str(e)}")
        return []
    finally:
        conn.close()

def get_content(url_id: int):
    """Retrieve content for a specific URL"""
    conn = sqlite3.connect('rag_settings.db')
    c = conn.cursor()
    try:
        c.execute('''
            SELECT title, content, scraped_date 
            FROM content 
            WHERE url_id = ?
            ORDER BY scraped_date DESC 
            LIMIT 1
        ''', (url_id,))
        return c.fetchone()
    except Exception as e:
        st.error(f"Error retrieving content: {str(e)}")
        return None
    finally:
        conn.close()