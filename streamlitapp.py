import streamlit as st
from typing import Dict, List, Optional
from dataclasses import dataclass
import asyncio
from contextlib import asynccontextmanager
from utils.vector_store import VectorStore
from utils.rag import RAGPipeline
from utils.logger import Logger

# Type definitions for better type safety
@dataclass
class Message:
    role: str
    content: str
    sources: Optional[str] = None
    confidence: Optional[float] = None

class ChatInterface:
    def __init__(self):
        self.logger = Logger()
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize all session state variables"""
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = VectorStore()
        
        if 'rag_pipeline' not in st.session_state:
            st.session_state.rag_pipeline = RAGPipeline(st.session_state.vector_store)
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history: List[Message] = []
        
        if 'processing' not in st.session_state:
            st.session_state.processing = False

    @asynccontextmanager
    async def _processing_state(self):
        """Context manager to handle processing state"""
        st.session_state.processing = True
        try:
            yield
        finally:
            st.session_state.processing = False

    def _display_message(self, message: Message):
        """Display a single chat message with proper formatting"""
        if message.role == "user":
            st.write("You:", message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)
                if message.sources:
                    with st.expander("View Sources"):
                        st.markdown(message.sources)
                        if message.confidence is not None:
                            confidence = round(message.confidence, 2)
                            st.progress(confidence)
                            st.caption(f"Confidence Score: {confidence}")

    def display_chat_history(self):
        """Display the entire chat history"""
        self.logger.info(f"Displaying chat history with {len(st.session_state.chat_history)} messages")
        for message in st.session_state.chat_history:
            self._display_message(message)

    async def process_question(self, question: str):
        """Process user question through RAG pipeline"""
        async with self._processing_state():
            try:
                # Get response from RAG pipeline
                response = await st.session_state.rag_pipeline.get_answer(question)
                
                # Format source references
                sources_text = st.session_state.rag_pipeline.format_sources(response.sources)
                
                # Add messages to chat history
                st.session_state.chat_history.extend([
                    Message(role="user", content=question),
                    Message(
                        role="assistant",
                        content=response.answer,
                        sources=sources_text,
                        confidence=response.confidence
                    )
                ])
                
            except Exception as e:
                self.logger.error(f"Error processing question: {str(e)}")
                st.error("I encountered an error while processing your question. Please try again.")
                
                # Add error message to chat history
                st.session_state.chat_history.extend([
                    Message(role="user", content=question),
                    Message(
                        role="assistant",
                        content="I apologize, but I encountered an error while processing your question. Please try again."
                    )
                ])

def main():
    st.set_page_config(
        page_title="AWS Documentation RAG",
        page_icon="📚",
        layout="wide"
    )
    
    chat = ChatInterface()
    
    st.title("AWS Documentation Assistant")
    
    st.write("""
    Welcome to the AWS Documentation Assistant! This tool helps you search and understand AWS documentation.
    
    Use the sidebar to:
    - Chat with the documentation
    - Manage documentation sources in Settings
    """)
    
    # Main chat interface
    with st.container():
        st.subheader("Ask about AWS")
        
        # Display chat history
        chat.display_chat_history()
        st.divider()
        
        # Callback to handle input submission
        def handle_input():
            if st.session_state.user_input:
                if st.session_state.vector_store.get_stats()["total_chunks"] == 0:
                    st.warning("Please add and scrape some AWS documentation in the Settings page first!")
                else:
                    # Store the question and clear input
                    question = st.session_state.user_input
                    st.session_state.user_input = ""
                    
                    # Process the question
                    with st.spinner("Searching documentation..."):
                        asyncio.run(chat.process_question(question))
        
        # User input - disabled during processing
        st.text_input(
            "Your question:",
            key="user_input",
            disabled=st.session_state.processing,
            on_change=handle_input
        )
        
        # Add clear chat button
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.session_state.chat_history and st.button("Clear Chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()