from typing import List, Dict, Optional, Any
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
from dataclasses import dataclass, field

@dataclass
class RAGResponse:
    """Represents a RAG-generated response with metadata"""
    answer: str
    sources: List[Dict]  # List of source chunks with metadata
    confidence: float    # Overall confidence score (0-1)
    
@dataclass
class LLMHistoryItem:
    """Represents a full context + message pair for LLM history"""
    role: str  # "system", "user", or "assistant" 
    content: str
    context: Optional[str] = None  # The context provided for this message
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class UIMessage:
    """Message format for chat interface"""
    role: str  # "user" or "assistant"
    content: str
    sources: Optional[str] = None
    confidence: Optional[float] = None
    
class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        # Track used chunks to avoid repetition
        if 'used_chunks' not in st.session_state:
            st.session_state.used_chunks = set()  # Set of chunk hashes
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
        
        # Initialize separate histories
        if 'llm_history' not in st.session_state:
            st.session_state.llm_history = [
                LLMHistoryItem(
                    role="system",
                    content="""You are an AWS documentation assistant. Your role is to provide accurate, 
                    helpful answers about AWS services and features based on the provided documentation context.
                    Only answer based on the provided context. If the context doesn't contain enough information,
                    say so. 
                    
                    When referring to documentation, use the exact format [[Title](URL)] for citations, where Title
                    and URL match those provided in the context. Always include at least one citation per statement."""
                )
            ]
            
        if 'ui_messages' not in st.session_state:
            st.session_state.ui_messages = []

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

    def _filter_new_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Filter out chunks that have been used in previous context"""
        new_chunks = []
        for chunk in chunks:
            chunk_hash = self._hash_chunk(chunk)
            if chunk_hash not in st.session_state.used_chunks:
                new_chunks.append(chunk)
                st.session_state.used_chunks.add(chunk_hash)
        return new_chunks

    def _process_response(self, response: str, source_map: Dict[str, str]) -> str:
        """Process the response to ensure proper markdown formatting of references"""
        # Replace any existing markdown-style links that might be malformed
        response = re.sub(r'\[([^\]]+)\]\((?:[^\)]+)?\)', r'[\1]', response)
        
        # Replace references with proper markdown links
        for title, url in source_map.items():
            response = re.sub(
                f'\\[{re.escape(title)}\\](?!\\()',
                f'[{title}]({url})',
                response
            )
        
        return response

    def format_context(self, chunks: List[Dict], new_chunks_only: bool = True) -> str:
        """Format context chunks into a string, optionally marking new chunks"""
        context_parts = []
        for chunk in chunks:
            title = chunk['metadata'].get('title', 'Unknown Section')
            prefix = "[NEW] " if new_chunks_only and chunk in self._filter_new_chunks(chunks) else ""
            context_parts.append(f"{prefix}[{title}]\n{chunk['content']}")
        return "\n\n".join(context_parts)

    def format_sources(self, sources: List[Dict]) -> str:
        """Format source references for display"""
        formatted_sources = []
        for source in sources:
            metadata = source['metadata']
            title = metadata.get('title', 'Unknown Section')
            url = metadata.get('url', '')
            relevance = round(source.get('relevance', 0), 2)
            path = metadata.get('path', '')
            is_new = self._hash_chunk(source) in self._filter_new_chunks([source])
            prefix = "[NEW] " if is_new else ""
            
            if url:
                formatted_sources.append(f"- {prefix}[{title}]({url}) *{path}* ({relevance})")
            else:
                formatted_sources.append(f"- {prefix}{title} *{path}* ({relevance})")
        return "\n".join(formatted_sources)

    def _convert_to_langchain_messages(self, history: List[LLMHistoryItem]) -> List[Any]:
        """Convert our LLMHistoryItem objects to LangChain message format"""
        langchain_messages = []
        for item in history:
            content = item.content
            if item.context:  # Add context if it exists
                content = f"Context:\n{item.context}\n\nMessage:\n{content}"
                
            if item.role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif item.role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif item.role == "assistant":
                langchain_messages.append(AIMessage(content=content))
        return langchain_messages

    async def get_answer(
        self,
        question: str,
        url_id: Optional[int] = None,
        min_relevance: float = 0,
        max_chunks: int = 5
    ) -> RAGResponse:
        """Generate an answer using RAG pipeline"""
        self.logger.info(f"Generating answer for question: {question}")
        try:
            # Retrieve and process chunks
            chunks = self.vector_store.search_similar(
                query=question,
                url_id=url_id,
                n_results=max_chunks,
                min_relevance=min_relevance
            )
            
            if not chunks:
                return self._handle_no_context(question)
            
            # Format context
            context = self.format_context(chunks)
            self.logger.info(f"Context: {context}")
            source_map = self._create_source_map(chunks)
            
            # Add question with context to LLM history
            st.session_state.llm_history.append(
                LLMHistoryItem(
                    role="user",
                    content=question,
                    context=context
                )
            )
            
            # Convert full history to LangChain format
            llm_messages = self._convert_to_langchain_messages(st.session_state.llm_history)
            self.logger.info(f"LLM messages: {llm_messages}")
            
            # Get response from LLM
            response = await self._llm.ainvoke(llm_messages)
            self.logger.info(f"LLM response: {response.content}")
            processed_response = self._process_response(response.content, source_map)
            
            # Calculate confidence
            avg_relevance = sum(chunk.get('relevance', 0) for chunk in chunks) / len(chunks)
            sources_text = self.format_sources(chunks)
            
            # Add response to LLM history
            st.session_state.llm_history.append(
                LLMHistoryItem(
                    role="assistant",
                    content=processed_response,
                    context=context
                )
            )
            
            # Add to UI messages (clean version)
            st.session_state.ui_messages.extend([
                UIMessage(role="user", content=question),
                UIMessage(
                    role="assistant",
                    content=processed_response,
                    sources=sources_text,
                    confidence=avg_relevance
                )
            ])
            
            # Prune histories if too long
            self._prune_histories()
            
            return RAGResponse(
                answer=processed_response,
                sources=chunks,
                confidence=avg_relevance
            )
            
        except Exception as e:
            return self._handle_error(question, str(e))

    def _handle_no_context(self, question: str) -> RAGResponse:
        """Handle case where no relevant context is found"""
        error_msg = "I don't have enough context in my knowledge base to answer this question."
        
        # Add to histories
        st.session_state.llm_history.append(
            LLMHistoryItem(role="user", content=question)
        )
        st.session_state.llm_history.append(
            LLMHistoryItem(role="assistant", content=error_msg)
        )
        
        st.session_state.ui_messages.extend([
            UIMessage(role="user", content=question),
            UIMessage(role="assistant", content=error_msg, confidence=0.0)
        ])
        
        return RAGResponse(answer=error_msg, sources=[], confidence=0.0)

    def _handle_error(self, question: str, error_msg: str) -> RAGResponse:
        """Handle errors in the RAG pipeline"""
        error_response = f"Error in RAG pipeline: {error_msg}"
        
        # Add to histories
        st.session_state.llm_history.append(
            LLMHistoryItem(role="user", content=question)
        )
        st.session_state.llm_history.append(
            LLMHistoryItem(role="assistant", content=error_response)
        )
        
        st.session_state.ui_messages.extend([
            UIMessage(role="user", content=question),
            UIMessage(role="assistant", content=error_response, confidence=0.0)
        ])
        
        return RAGResponse(answer=error_response, sources=[], confidence=0.0)

    def _prune_histories(self, max_turns: int = 10):
        """Prune both histories while maintaining consistency"""
        # Keep system message plus last max_turns * 2 messages (each turn is Q&A)
        if len(st.session_state.llm_history) > (max_turns * 2 + 1):
            st.session_state.llm_history = (
                [st.session_state.llm_history[0]] +  # Keep system message
                st.session_state.llm_history[-(max_turns * 2):]  # Keep last N turns
            )
        
        # Keep last max_turns * 2 UI messages
        if len(st.session_state.ui_messages) > (max_turns * 2):
            st.session_state.ui_messages = st.session_state.ui_messages[-(max_turns * 2):]

    def clear_history(self):
        """Clear both conversation histories"""
        st.session_state.llm_history = [st.session_state.llm_history[0]]  # Keep only system message
        st.session_state.ui_messages = []  # Clear UI messages