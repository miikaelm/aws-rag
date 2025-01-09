from typing import List, Dict, Optional
import streamlit as st
from datetime import datetime
from dataclasses import dataclass
from utils.vector_store import VectorStore, Document
from utils.logger import Logger
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.messages import SystemMessage, HumanMessage
import os

@dataclass
class RAGResponse:
    """Represents a RAG-generated response with metadata"""
    answer: str
    sources: List[Dict]  # List of source chunks with metadata
    confidence: float    # Overall confidence score (0-1)

def init_gemini_llm(api_key: Optional[str] = None) -> ChatGoogleGenerativeAI:
    """
    Initialize Google Gemini LLM with LangChain
    
    Args:
        api_key: Optional API key, will use environment variable if not provided
        
    Returns:
        Configured ChatGoogleGenerativeAI instance
    """
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    logger = Logger()
    logger.info(f"GOOGLE_API_KEY: {api_key}")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY must be provided or set in environment")
        
    return ChatGoogleGenerativeAI(
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


class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        """
        Initialize RAG pipeline
        
        Args:
            vector_store: Initialized VectorStore instance
        """
        self.vector_store = vector_store
        self.logger = Logger()
        
        # Default system prompt template
        self.system_prompt = """You are an AWS documentation assistant. Your role is to provide accurate, 
        helpful answers about AWS services and features based on the provided documentation context. 
        
        Guidelines:
        - Only answer based on the provided context
        - If the context doesn't contain enough information, say so
        - Include relevant code examples when available
        - Cite sources using metadata from chunks
        - Maintain a professional, technical tone
        """
        
        # Default user prompt template
        self.user_prompt_template = """Context information is below:
        ---------------
        {context}
        ---------------
        Given the context information, answer the following question:
        {question}
        
        When referencing information, cite the source section using [Title] notation.
        If you cannot answer the question based on the context, say so clearly."""

    def generate_prompt(self, question: str, context_chunks: List[Dict]) -> str:
        """
        Generate the complete prompt from templates and chunks
        
        Args:
            question: User's question
            context_chunks: Retrieved relevant chunks with metadata
        """
        # Format context from chunks
        context_parts = []
        for chunk in context_chunks:
            # Add section title as reference
            title = chunk['metadata'].get('title', 'Unknown Section')
            
            # Format chunk content with title
            context_parts.append(f"[{title}]\n{chunk['content']}")
            
        context_text = "\n\n".join(context_parts)
        
        # Fill user prompt template
        user_prompt = self.user_prompt_template.format(
            context=context_text,
            question=question
        )
        
        return user_prompt

    async def get_answer(
        self,
        question: str,
        url_id: Optional[int] = None,
        min_relevance: float = 0,
        max_chunks: int = 5
    ) -> RAGResponse:
        """
        Generate an answer using RAG pipeline
        
        Args:
            question: User's question
            url_id: Optional URL ID to limit search scope
            min_relevance: Minimum relevance score for chunks
            max_chunks: Maximum number of chunks to include
        """
        self.logger.info(f"Question: {question}")
        try:
            # 1. Retrieve relevant chunks
            chunks = self.vector_store.search_similar(
                query=question,
                url_id=url_id,
                n_results=max_chunks,
                min_relevance=min_relevance
            )
            

            self.logger.info(f"Chunks: {chunks}")
            
            if not chunks:
                return RAGResponse(
                    answer="I don't have enough context in my knowledge base to answer this question.",
                    sources=[],
                    confidence=0.0
                )
            
            # 2. Generate prompt with context
            prompt = self.generate_prompt(question, chunks)
            
            # 3. Get response from LLM
            response = await self._get_llm_response(prompt)
            
            # 4. Calculate confidence based on chunk relevance
            avg_relevance = sum(chunk['relevance'] for chunk in chunks) / len(chunks)
            
            return RAGResponse(
                answer=response,
                sources=chunks,
                confidence=avg_relevance
            )
            
        except Exception as e:
            self.logger.error(f"Error in RAG pipeline: {str(e)}")
            st.error(f"Error in RAG pipeline: {str(e)}")
            return RAGResponse(
                answer="Sorry, I encountered an error while processing your question.",
                sources=[],
                confidence=0.0
            )

    async def _get_llm_response(self, prompt: str) -> str:
        """
        Get response from Gemini LLM using LangChain
        
        Args:
            prompt: Complete prompt for LLM
            
        Returns:
            Generated response text
        """
        try:
            # Initialize LLM if not already done
            if not hasattr(self, '_llm'):
                self._llm = init_gemini_llm()
            
            # Create message list with system and user prompts
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            # Get response from LLM
            response = await self._llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def format_sources(self, sources: List[Dict]) -> str:
        """
        Format source references for display
        
        Args:
            sources: List of source chunks with metadata
        """
        logger = Logger()
        formatted_sources = []
        for source in sources:
            metadata = source['metadata']
            title = metadata.get('title', 'Unknown Section')
            url = metadata.get('url', '')
            logger.info(f"Metadata: {metadata}")
            # round relevance to 2 decimal places
            relevance = round(source.get('relevance', 0), 2)
            path = metadata.get('path', '')
            if url:
                source_url = f"{url}"
                formatted_sources.append(f"- [{title}]({source_url}) *{path}* ({relevance} %)")
            else:
                formatted_sources.append(f"- {title} *{path}* ({relevance} %)")
                
        return "\n".join(formatted_sources)

    def update_system_prompt(self, new_prompt: str):
        """Update the system prompt template"""
        self.system_prompt = new_prompt

    def update_user_prompt_template(self, new_template: str):
        """Update the user prompt template"""
        self.user_prompt_template = new_template