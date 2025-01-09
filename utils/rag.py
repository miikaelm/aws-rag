from typing import List, Dict, Optional, Tuple, Set
import streamlit as st
from datetime import datetime
from dataclasses import dataclass
from utils.vector_store import VectorStore, Document
from utils.logger import Logger
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
import hashlib
import re

@dataclass
class RAGResponse:
    """Represents a RAG-generated response with metadata"""
    answer: str
    sources: List[Dict]  # List of source chunks with metadata
    confidence: float    # Overall confidence score (0-1)
    
@dataclass
class Message:
    """Message format for chat interface"""
    role: str  # "system", "user", or "assistant"
    content: str
    sources: Optional[str] = None
    confidence: Optional[float] = None
    chunk_hashes: Optional[Set[str]] = None  # Store hashes of chunks used
    
class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.logger = Logger()
        
        # Initialize LLM
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable must be set")
            
        self._llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=api_key,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
        
        # Initialize messages and context tracking in session state
        if 'messages' not in st.session_state:
            st.session_state.messages = [
                Message(
                    role="system",
                    content="""You are an AWS documentation assistant. Your role is to provide accurate, 
                    helpful answers about AWS services and features based on the provided documentation context.
                    Only answer based on the provided context. If the context doesn't contain enough information,
                    say so. 
                    
                    When referring to documentation, use the exact format [[Title](URL)] for citations, where Title
                    and URL match those provided in the context. Always include at least one citation per statement.""",
                    chunk_hashes=set()
                )
            ]
        if 'used_chunks' not in st.session_state:
            st.session_state.used_chunks = set()

    def _create_source_map(self, chunks: List[Dict]) -> Dict[str, str]:
        """Create a mapping of titles to URLs for source linking"""
        source_map = {}
        for chunk in chunks:
            metadata = chunk['metadata']
            title = metadata.get('title', 'Unknown Section')
            url = metadata.get('url', '')
            if url and title:
                source_map[title] = url
        return source_map
    
    def _hash_chunk(self, chunk: Dict) -> str:
        """Create a unique hash for a chunk based on its content and metadata"""
        content = chunk['content']
        metadata = str(sorted(chunk['metadata'].items()))  # Convert metadata to stable string
        return hashlib.md5(f"{content}{metadata}".encode()).hexdigest()

    def _process_response(self, response: str, source_map: Dict[str, str]) -> str:
        """Process the response to ensure proper markdown formatting of references"""
        # Replace any existing markdown-style links that might be malformed
        response = re.sub(r'\[([^\]]+)\]\((?:[^\)]+)?\)', r'[\1]', response)
        
        # Replace references with proper markdown links
        for title, url in source_map.items():
            # Look for titles in square brackets without links
            response = re.sub(
                f'\\[{re.escape(title)}\\](?!\\()',
                f'[{title}]({url})',
                response
            )
        
        return response
    
    def _filter_new_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Filter out chunks that have been used in previous context"""
        new_chunks = []
        for chunk in chunks:
            chunk_hash = self._hash_chunk(chunk)
            if chunk_hash not in st.session_state.used_chunks:
                new_chunks.append(chunk)
                # Add to tracking sets
                st.session_state.used_chunks.add(chunk_hash)
        return new_chunks

    def _convert_to_langchain_messages(self, messages: List[Message], include_system: bool = True):
        """Convert our Message objects to LangChain message format"""
        langchain_messages = []
        for msg in messages:
            if msg.role == "system" and include_system:
                langchain_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
        return langchain_messages

    def format_context(self, chunks: List[Dict], new_chunks_only: bool = True) -> str:
        """Format context chunks into a string, optionally marking new chunks"""
        context_parts = []
        for chunk in chunks:
            title = chunk['metadata'].get('title', 'Unknown Section')
            prefix = "[NEW] " if new_chunks_only and chunk in self._filter_new_chunks(chunks) else ""
            context_parts.append(f"{prefix}[{title}]\n{chunk['content']}")
        return "\n\n".join(context_parts)
    
    async def get_answer(
        self,
        question: str,
        url_id: Optional[int] = None,
        min_relevance: float = 0,
        max_chunks: int = 5
    ) -> RAGResponse:
        """Generate an answer using RAG pipeline"""
        self.logger.info(f"Question: {question}")
        
        try:
            # Retrieve relevant chunks
            all_chunks = self.vector_store.search_similar(
                query=question,
                url_id=url_id,
                n_results=max_chunks,
                min_relevance=min_relevance
            )
            
            # Filter for new chunks
            new_chunks = self._filter_new_chunks(all_chunks)
            
            if not new_chunks and not all_chunks:
                no_context_response = RAGResponse(
                    answer="I don't have enough context in my knowledge base to answer this question.",
                    sources=[],
                    confidence=0.0
                )
                
                st.session_state.messages.extend([
                    Message(role="user", content=question),
                    Message(
                        role="assistant",
                        content=no_context_response.answer,
                        confidence=0.0,
                        chunk_hashes=set()
                    )
                ])
                return no_context_response
            
            # Create source map for link formatting
            source_map = self._create_source_map(all_chunks)
            
            # Format context, including only new chunks
            context = self.format_context(new_chunks) if new_chunks else "Using previously provided context."

            # Create the contextual question
            contextual_question = f"""Context information is below:
            ---------------
            {context}
            ---------------
            Question: {question}
            Remember to use the exact format [[Title](URL)] for all references, where Title and URL match those in the context.
            Note: Context shown is additional to previously provided information."""
            
            # Convert history to LangChain format
            history_messages = self._convert_to_langchain_messages(st.session_state.messages)
            
            # Add the contextual question
            llm_messages = history_messages + [HumanMessage(content=contextual_question)]
            
            self.logger.info(f"LLM messages: {llm_messages}")
            # Get response from LLM
            response = await self._llm.ainvoke(llm_messages)
            
            # Process response to ensure proper link formatting
            processed_response = self._process_response(response.content, source_map)
            
            # Calculate confidence based on all relevant chunks
            avg_relevance = sum(chunk.get('relevance', 0) for chunk in all_chunks) / len(all_chunks)
            
            # Format sources (show all relevant sources, both new and previously used)
            sources_text = self.format_sources(all_chunks)
            
            # Create chunk hashes set for this message
            current_chunk_hashes = {self._hash_chunk(chunk) for chunk in all_chunks}
            
            # Create RAG response
            rag_response = RAGResponse(
                answer=processed_response,
                sources=all_chunks,
                confidence=avg_relevance
            )
            
            # Add visible messages to history
            st.session_state.messages.extend([
                Message(role="user", content=question),
                Message(
                    role="assistant",
                    content=processed_response,
                    sources=sources_text,
                    confidence=avg_relevance,
                    chunk_hashes=current_chunk_hashes
                )
            ])
            
            # Keep only last N turns plus system message
            max_turns = 10  # 5 turns (question/answer pairs)
            if len(st.session_state.messages) > (max_turns + 1):
                st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-(max_turns):]
            
            return rag_response
            
        except Exception as e:
            error_msg = f"Error in RAG pipeline: {str(e)}"
            self.logger.error(error_msg)
            
            st.session_state.messages.extend([
                Message(role="user", content=question),
                Message(
                    role="assistant",
                    content=error_msg,
                    confidence=0.0,
                    chunk_hashes=set()
                )
            ])
            
            return RAGResponse(
                answer=error_msg,
                sources=[],
                confidence=0.0
            )

    def format_sources(self, sources: List[Dict]) -> str:
        """Format source references for display"""
        formatted_sources = []
        for source in sources:
            metadata = source['metadata']
            title = metadata.get('title', 'Unknown Section')
            url = metadata.get('url', '')
            self.logger.info(f"Metadata: {metadata}")
            relevance = round(source.get('relevance', 0), 2)
            path = metadata.get('path', '')
            is_new = self._hash_chunk(source) in self._filter_new_chunks([source])
            prefix = "[NEW] " if is_new else ""
            if url:
                formatted_sources.append(f"- {prefix}[{title}]({url}) *{path}* ({relevance})")
            else:
                formatted_sources.append(f"- {prefix}{title} *{path}* ({relevance})")
        return "\n".join(formatted_sources)
    
    def clear_history(self):
        """Clear conversation history and context tracking"""
        st.session_state.messages = [st.session_state.messages[0]]  # Keep only system message
        st.session_state.used_chunks = set()  # Reset chunk tracking