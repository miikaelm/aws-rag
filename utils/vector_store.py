from typing import List, Dict, Optional
import os
import numpy as np
from dataclasses import dataclass
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st
from datetime import datetime
from utils.logger import Logger

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
            
            logger = Logger()
            logger.info(f"Adding {len(chunks)} chunks to vector store for URL ID {url_id}")
            
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
        n_results: int = 10,  # Increased from 5
        min_relevance: float = 0.6  # Increased from 0.0
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
            
            # Query collection with increased n_results to account for filtering
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results * 2,  # Get more results initially
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results with better relevance scoring
            documents = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    # Calculate relevance score with improved scaling
                    distance = results['distances'][0][i]
                    # Sigmoid-like transformation for better relevance distribution
                    relevance = 1 / (1 + np.exp(distance * 2 - 1))
                    
                    # Skip if below minimum relevance
                    if relevance < min_relevance:
                        continue
                        
                    # Get token count if available
                    token_count = int(results['metadatas'][0][i].get('token_count', '0'))
                    
                    # Boost relevance for larger chunks (more context)
                    if token_count > 0:
                        size_boost = min(token_count / 512, 1.2)  # Max 20% boost
                        relevance *= size_boost
                    
                    documents.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'relevance': relevance
                    })
            
            # Sort by relevance and limit to n_results
            documents.sort(key=lambda x: x['relevance'], reverse=True)
            return documents[:n_results]
            
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