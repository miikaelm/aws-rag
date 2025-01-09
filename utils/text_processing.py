from typing import List, Dict
import re
from utils.vector_store import Document
import tiktoken

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string"""
    # Use cl100k_base encoder (used by latest GPT models)
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

def chunk_section_content(
    content: str,
    chunk_size: int = 512,  # Now represents tokens, not characters
    overlap: int = 50  # Now represents tokens, not characters
) -> List[str]:
    """
    Split section content into overlapping chunks based on token count
    
    Args:
        content: Section content to split
        chunk_size: Target chunk size in tokens
        overlap: Number of tokens to overlap
    """
    # Clean and normalize whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Estimate characters per token (typically 4-5 characters per token)
    chars_per_token = 4
    char_chunk_size = chunk_size * chars_per_token
    char_overlap = overlap * chars_per_token
    
    # Return as single chunk if content is small enough
    if estimate_tokens(content) <= chunk_size:
        return [content]
    
    chunks = []
    start = 0
    
    while start < len(content):
        # Calculate initial end position
        end = start + char_chunk_size
        
        if end >= len(content):
            # Add final chunk
            final_chunk = content[start:].strip()
            if estimate_tokens(final_chunk) > 0:  # Only add if not empty
                chunks.append(final_chunk)
            break
        
        # Look for sentence boundary within a reasonable window
        next_period = content.find('.', end - 100, end + 100)
        if next_period != -1:
            end = next_period + 1
        else:
            # Find word boundary
            while end > start and not content[end-1].isspace():
                end -= 1
            if end == start:
                # No good boundary found, use character-based chunk size
                end = start + char_chunk_size
                # Try to break at word boundary
                while end < len(content) and not content[end].isspace():
                    end += 1
        
        chunk = content[start:end].strip()
        if estimate_tokens(chunk) > 0:  # Only add if not empty
            chunks.append(chunk)
        
        # Move start position back by overlap tokens
        start = end - char_overlap
        
        # Verify chunk sizes and adjust if needed
        if len(chunks) > 0 and estimate_tokens(chunks[-1]) > chunk_size * 1.5:
            # If chunk is too large, split it further
            large_chunk = chunks.pop()
            subchunks = chunk_section_content(
                large_chunk,
                chunk_size=chunk_size,
                overlap=overlap
            )
            chunks.extend(subchunks)
    
    return chunks

def process_section_content(
    section: Dict,
    url: str,
    chunk_size: int = 512,  # Now in tokens
    overlap: int = 50  # Now in tokens
) -> List[Document]:
    """
    Process section content into vector store documents
    
    Args:
        section: Section dictionary from database
        url: Base URL for the document
        chunk_size: Target chunk size in tokens
        overlap: Number of tokens to overlap
    """
    # Skip empty content
    if not section.get('content'):
        return []
    
    # Include title and path in content for better context
    full_content = f"{section['title']}\n\n{section['content']}"
    if section.get('path'):
        full_content = f"Path: {section['path']}\n\n{full_content}"
    
    # Split content into chunks
    content_chunks = chunk_section_content(
        full_content,
        chunk_size=chunk_size,
        overlap=overlap
    )
    
    # Create document objects
    documents = []
    for i, chunk in enumerate(content_chunks):
        # Prepare metadata
        metadata = {
            'title': str(section['title']),
            'section_id': str(section['id']),
            'level': str(section['level']),
            'path': str(section['path']),
            'url': str(f"{url}#{section.get('url_fragment', '')}"),
            'chunk_index': str(i),
            'total_chunks': str(len(content_chunks)),
            'token_count': str(estimate_tokens(chunk))
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
    chunk_size: int = 512,  # Now in tokens
    overlap: int = 50  # Now in tokens
) -> List[Document]:
    """
    Process all sections into documents for vector store
    
    Args:
        sections: List of sections from database
        url: Base URL for the document
        chunk_size: Target chunk size in tokens
        overlap: Number of tokens to overlap
    """
    documents = []
    
    # Process sections in hierarchical order
    sorted_sections = sorted(sections, key=lambda x: x.get('path', ''))
    
    for section in sorted_sections:
        section_docs = process_section_content(
            section,
            url,
            chunk_size=chunk_size,
            overlap=overlap
        )
        documents.extend(section_docs)
    
    return documents