from typing import List, Dict
import re
from utils.vector_store import Document

def chunk_section_content(
    content: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[str]:
    """
    Split section content into overlapping chunks
    
    Args:
        content: Section content to split
        chunk_size: Target chunk size in characters
        overlap: Number of characters to overlap
    """
    # Clean and normalize whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Return as single chunk if content is small enough
    if len(content) <= chunk_size:
        return [content]
    
    chunks = []
    start = 0
    
    while start < len(content):
        # Calculate end position
        end = start + chunk_size
        
        if end >= len(content):
            # Add final chunk
            chunks.append(content[start:])
            break
        
        # Try to find sentence boundary
        next_period = content.find('.', end - 30, end + 30)
        if next_period != -1:
            end = next_period + 1
        else:
            # Try to find word boundary
            while end > start and content[end] != ' ':
                end -= 1
            if end == start:
                # No good boundary found, use chunk_size
                end = start + chunk_size
        
        # Add chunk and move start position
        chunks.append(content[start:end].strip())
        start = end - overlap
    
    return chunks

def process_section_content(
    section: Dict,
    url: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[Document]:
    """
    Process section content into vector store documents
    
    Args:
        section: Section dictionary from database
        url: Base URL for the document
        chunk_size: Target chunk size
        overlap: Chunk overlap size
    """
    # Skip empty content
    if not section['content']:
        return []
        
    # Split content into chunks
    content_chunks = chunk_section_content(
        section['content'],
        chunk_size=chunk_size,
        overlap=overlap
    )
    
    # Create document objects
    documents = []
    for i, chunk in enumerate(content_chunks):
        # Prepare metadata
        metadata = {
            'title': section['title'],
            'section_id': section['id'],
            'level': section['level'],
            'path': section['path'],
            'url': f"{url}#{section['url_fragment']}",
            'chunk_index': i,
            'total_chunks': len(content_chunks)
        }
        
        # Create document
        document = Document(
            content=chunk,
            metadata=metadata
        )
        
        documents.append(document)
    
    return documents

def prepare_sections_for_indexing(
    sections: List[Dict],
    url: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[Document]:
    """
    Process all sections into documents for vector store
    
    Args:
        sections: List of sections from database
        url: Base URL for the document
        chunk_size: Target chunk size
        overlap: Chunk overlap size
    """
    documents = []
    
    for section in sections:
        section_docs = process_section_content(
            section,
            url,
            chunk_size=chunk_size,
            overlap=overlap
        )
        documents.extend(section_docs)
    
    return documents