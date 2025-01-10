import requests
from bs4 import BeautifulSoup
from typing import Optional, List, Tuple
import streamlit as st
from db.models.section import Section
from utils.content_processor import ContentProcessor, ContentWarning

class DocumentScraper:
    """Handles document scraping and section extraction"""
    
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
        section_order = 0
        
        for header in main_content.find_all(['h1', 'h2', 'h3']):
            section = self._process_header(header, base_url, section_order)
            section_order += 1
            
            # Handle hierarchy
            while section_stack and section_stack[-1].level >= section.level:
                section_stack.pop()

            if section_stack:
                section.parent_id = section_stack[-1].id
                
            section_stack.append(section)
            sections.append(section)

        return sections
    
    def _process_header(self, header: BeautifulSoup, base_url: str, order: int) -> Section:
        """Process a single header element and its content"""
        level = int(header.name[1])
        url_fragment = self._get_header_id(header)
        content = self._extract_content(header)
        title = header.get_text(strip=True)
        
        # Create section instance
        section = Section(
            title=title,
            content=content,
            level=level,
            url_fragment=url_fragment,
            section_order=order
        )
        
        # Check content length and add warning if needed
        warning = ContentProcessor.check_content_length(content, title)
        if warning:
            if 'content_warnings' not in st.session_state:
                st.session_state.content_warnings = []
            st.session_state.content_warnings.append(warning)
        
        return section
    
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