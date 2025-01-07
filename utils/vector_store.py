from typing import List, Dict, Optional
import os
import numpy as np
from dataclasses import dataclass
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st
from datetime import datetime

@dataclass
class Document:
    """Represents a document chunk with metadata"""
    content: str
    metadata: Dict

class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db"):
        """
        Initialize ChromaDB vector store
        
        Args:
            persist_dir: Directory for storing vector data
        """
        # Ensure vector store directory exists
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Create or get collection with sentence transformer embeddings
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="aws_docs",
            embedding_function=self.embedding_func
        )
    
    def add_section_chunks(self, chunks: List[Document], url_id: int) -> bool:
        """
        Add chunked section content to vector store
        
        Args:
            chunks: List of Document objects containing section chunks
            url_id: ID of the source URL
        """
        try:
            if not chunks:
                return True
                
            # Generate unique IDs for chunks
            chunk_ids = [
                f"{url_id}_{i}" 
                for i in range(len(chunks))
            ]
            
            # Extract content and metadata
            texts = [chunk.content for chunk in chunks]
            metadatas = [
                {
                    **chunk.metadata,
                    "url_id": str(url_id),  # ChromaDB requires metadata values to be strings
                    "indexed_at": datetime.now().isoformat()
                }
                for chunk in chunks
            ]
            
            # Add chunks to collection
            self.collection.upsert(
                documents=texts,
                ids=chunk_ids,
                metadatas=metadatas
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error adding chunks to vector store: {str(e)}")
            return False

    def search_similar(
        self,
        query: str,
        url_id: Optional[int] = None,
        n_results: int = 5,
        min_relevance: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar content chunks
        
        Args:
            query: Search query
            url_id: Optional URL ID filter
            n_results: Number of results to return
            min_relevance: Minimum relevance score (0-1)
        """
        try:
            # Prepare where clause if url_id provided
            where = {"url_id": str(url_id)} if url_id is not None else None
            
            # Query collection
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            documents = []
            if results['ids'] and results['ids'][0]:  # Check if we have results
                for i in range(len(results['ids'][0])):
                    # Calculate relevance score (1 - normalized distance)
                    distance = results['distances'][0][i] if 'distances' in results else 0
                    relevance = 1 - (distance / 2)  # Convert distance to 0-1 score
                    
                    # Skip if below minimum relevance
                    if relevance < min_relevance:
                        continue
                    
                    documents.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'relevance': relevance
                    })
            
            return documents
            
        except Exception as e:
            st.error(f"Error searching vector store: {str(e)}")
            return []

    def delete_url_content(self, url_id: int) -> bool:
        """Delete all content chunks for a URL"""
        try:
            # Delete chunks by URL ID
            self.collection.delete(
                where={"url_id": str(url_id)}
            )
            return True
        except Exception as e:
            st.error(f"Error deleting chunks from vector store: {str(e)}")
            return False

    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        try:
            return {
                "total_chunks": self.collection.count(),
                "embedding_dims": 384  # all-MiniLM-L6-v2 produces 384-dim embeddings
            }
        except Exception as e:
            st.error(f"Error getting vector store stats: {str(e)}")
            return {"total_chunks": 0, "embedding_dims": 0}