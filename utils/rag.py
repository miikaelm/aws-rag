from typing import List, Dict, Optional, Tuple
import streamlit as st
from datetime import datetime
from dataclasses import dataclass
from utils.vector_store import VectorStore, Document
from utils.logger import Logger
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os

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
        
        # Initialize messages in session state if not already present
        if 'messages' not in st.session_state:
            st.session_state.messages = [
                Message(
                    role="system",
                    content="""You are an AWS documentation assistant. Your role is to provide accurate, 
                    helpful answers about AWS services and features based on the provided documentation context.
                    Only answer based on the provided context. If the context doesn't contain enough information,
                    say so. Include code examples when available and cite sources using [Title] notation."""
                )
            ]

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

    def format_context(self, chunks: List[Dict]) -> str:
        """Format context chunks into a string"""
        context_parts = []
        for chunk in chunks:
            title = chunk['metadata'].get('title', 'Unknown Section')
            context_parts.append(f"[{title}]\n{chunk['content']}")
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
            chunks = self.vector_store.search_similar(
                query=question,
                url_id=url_id,
                n_results=max_chunks,
                min_relevance=min_relevance
            )
            
            if not chunks:
                no_context_response = RAGResponse(
                    answer="I don't have enough context in my knowledge base to answer this question.",
                    sources=[],
                    confidence=0.0
                )
                
                # Add messages to history
                st.session_state.messages.extend([
                    Message(role="user", content=question),
                    Message(
                        role="assistant",
                        content=no_context_response.answer,
                        confidence=0.0
                    )
                ])
                return no_context_response
            
            # Format context
            context = self.format_context(chunks)
            
            # Create the contextual question but don't add to visible history
            contextual_question = f"""Context information is below:
            ---------------
            {context}
            ---------------
            Question: {question}"""
            
            # Convert history to LangChain format
            history_messages = self._convert_to_langchain_messages(st.session_state.messages)
            
            # Add the contextual question
            llm_messages = history_messages + [HumanMessage(content=contextual_question)]
            
            # Get response from LLM
            response = await self._llm.ainvoke(llm_messages)
            
            # Calculate confidence
            avg_relevance = sum(chunk.get('relevance', 0) for chunk in chunks) / len(chunks)
            
            # Format sources
            sources_text = self.format_sources(chunks)
            
            # Create RAG response
            rag_response = RAGResponse(
                answer=response.content,
                sources=chunks,
                confidence=avg_relevance
            )
            
            # Add visible messages to history (original question, not contextual)
            st.session_state.messages.extend([
                Message(role="user", content=question),
                Message(
                    role="assistant",
                    content=response.content,
                    sources=sources_text,
                    confidence=avg_relevance
                )
            ])
            
            # Keep only last N turns plus system message
            max_turns = 10  # 5 turns (question/answer pairs)
            if len(st.session_state.messages) > (max_turns + 1):  # +1 for system message
                st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-(max_turns):]
            
            return rag_response
            
        except Exception as e:
            error_msg = f"Error in RAG pipeline: {str(e)}"
            self.logger.error(error_msg)
            
            # Add error message to history
            st.session_state.messages.extend([
                Message(role="user", content=question),
                Message(
                    role="assistant",
                    content=error_msg,
                    confidence=0.0
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
            if url:
                formatted_sources.append(f"- [{title}]({url}) *{path}* ({relevance})")
            else:
                formatted_sources.append(f"- {title} *{path}* ({relevance})")
        return "\n".join(formatted_sources)
    
    def clear_history(self):
        """Clear conversation history but keep system prompt"""
        st.session_state.messages = [st.session_state.messages[0]]  # Keep only system message