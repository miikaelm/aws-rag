import streamlit as st
from typing import Dict, List, Optional
from dataclasses import dataclass
import asyncio
from contextlib import asynccontextmanager
from utils.vector_store import VectorStore
from utils.rag import RAGPipeline, UIMessage
from utils.logger import Logger
from utils.database import get_database

def initialize_components():
    """Initialize all necessary components"""
    # Get database instance
    get_database()
    
    # Initialize other components
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline(st.session_state.vector_store)
        
    if 'processing' not in st.session_state:
        st.session_state.processing = False

def get_or_create_eventloop():
    """Get the current event loop or create a new one"""
    if 'event_loop' not in st.session_state:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        st.session_state.event_loop = loop
    return st.session_state.event_loop

class ChatInterface:
    def __init__(self):
        self.logger = Logger()

    @asynccontextmanager
    async def _processing_state(self):
        """Context manager to handle processing state"""
        st.session_state.processing = True
        try:
            yield
        finally:
            st.session_state.processing = False

    def _display_message(self, message: UIMessage):
        """Display a single chat message with proper formatting"""
        # Display user messages
        if message.role == "user":
            with st.chat_message("user"):
                st.write(message.content)
                
        # Display assistant messages with sources and confidence
        elif message.role == "assistant":
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
        self.logger.info(f"Displaying chat history with {len(st.session_state.ui_messages)} messages")
        # Display all UI messages
        for message in st.session_state.ui_messages:
            self._display_message(message)

    async def process_question(self, question: str):
        """Process user question through RAG pipeline"""
        async with self._processing_state():
            try:
                # Get response from RAG pipeline
                # Note: RAG pipeline now handles adding messages to both histories
                response = await st.session_state.rag_pipeline.get_answer(question)
                
            except Exception as e:
                self.logger.error(f"Error processing question: {str(e)}")
                st.error("I encountered an error while processing your question. Please try again.")

def main():
    st.set_page_config(
        page_title="AWS Documentation RAG",
        page_icon="📚",
        layout="wide"
    )
    initialize_components()
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
                        loop = get_or_create_eventloop()
                        loop.run_until_complete(chat.process_question(question))
        
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
            if len(st.session_state.ui_messages) > 0 and st.button("Clear Chat", type="secondary"):
                # Clear both histories through RAG pipeline
                st.session_state.rag_pipeline.clear_history()
                st.rerun()

if __name__ == "__main__":
    main()