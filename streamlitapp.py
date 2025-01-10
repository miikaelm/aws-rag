import streamlit as st
from typing import Dict, List, Optional
from dataclasses import dataclass
import asyncio
from contextlib import asynccontextmanager
from utils.vector_store import VectorStore
from utils.rag import RAGPipeline, UIMessage
from utils.logger import Logger
from utils.database import get_database
from datetime import datetime

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
        self.db = get_database()
        
        # Initialize conversation state
        if 'current_conversation_id' not in st.session_state:
            self._initialize_active_conversation()
            
        # Load available conversations
        self.conversations = self.db.get_conversations()

    def _initialize_active_conversation(self):
            """Set the active conversation"""
            latest_conv_id = self.db.get_latest_conversation()
            if latest_conv_id:
                st.session_state.current_conversation_id = latest_conv_id
            else:
                # Create new conversation if none exists
                st.session_state.current_conversation_id = self.db.create_conversation()

    def _create_new_conversation(self) -> str:
        """Create a new conversation and return its ID"""
        return f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def conversation_selector(self):
        """Display conversation selector in the sidebar"""
        with st.sidebar:
            st.subheader("Conversations")
            
            # New conversation button
            if st.button("New Conversation", type="primary"):
                st.session_state.current_conversation_id = self.db.create_conversation()
                st.rerun()
            
            st.divider()
            
            # List existing conversations
            for conv in self.conversations:
                col1, col2 = st.columns([4, 1])
                with col1:
                    button_text = (
                        f"{conv['title']}\n"
                        f"{datetime.fromisoformat(conv['updated_at']).strftime('%Y-%m-%d %H:%M')}"
                    )
                    
                    if st.button(
                        button_text,
                        key=f"conv_{conv['id']}",
                        type="primary" if conv['id'] == st.session_state.current_conversation_id else "secondary",
                        use_container_width=True
                    ):
                        st.session_state.current_conversation_id = conv['id']
                        st.rerun()
                
                # Delete conversation button
                with col2:
                    if st.button("🗑️", key=f"del_{conv['id']}", type="secondary"):
                        self.db.delete_conversation(conv['id'])
                        if conv['id'] == st.session_state.current_conversation_id:
                            self._initialize_active_conversation()
                        st.rerun()

    def display_chat_history(self):
        """Display current conversation from DB"""
        messages = self.db.get_conversation_messages(st.session_state.current_conversation_id)
        
        # Display conversation title if it exists
        current_conv = next(
            (c for c in self.conversations if c['id'] == st.session_state.current_conversation_id), 
            None
        )
        if current_conv and current_conv.get('title'):
            st.caption(f"Current conversation: {current_conv['title']}")
        
        for msg in messages:
            if msg['message']['role'] == "user":
                with st.chat_message("user"):
                    st.write(msg['message']['content'])
            
            elif msg['message']['role'] == "assistant":
                with st.chat_message("assistant"):
                    st.write(msg['message']['content'])
                    if msg['sources']:
                        with st.expander("View Sources"):
                            st.markdown(self._format_sources(msg['sources']))
                            confidence = msg['message']['confidence']
                            if confidence is not None:
                                st.progress(confidence)
                                st.caption(f"Confidence Score: {confidence}")


    async def process_question(self, question: str):
        """Process user question and save directly to DB"""
        try:
            # Save user question to DB
            msg_order = self.db.get_next_message_order(st.session_state.current_conversation_id)
            user_msg_id = self.db.save_message(
                conversation_id=st.session_state.current_conversation_id,
                role="user",
                content=question,
                message_order=msg_order
            )

            # Get response from RAG pipeline
            response = await st.session_state.rag_pipeline.get_answer(question)
            
            # Save assistant response to DB
            assistant_msg_id = self.db.save_message(
                conversation_id=st.session_state.current_conversation_id,
                role="assistant",
                content=response.answer,
                confidence=response.confidence,
                message_order=msg_order + 1
            )
            
            # Save sources
            if hasattr(response, 'sources'):
                self.db.save_message_sources(assistant_msg_id, response.sources)

        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}")
            st.error("I encountered an error while processing your question. Please try again.")

    def _format_sources(self, sources: List[Dict]) -> str:
        """Format source references for display in markdown"""
        formatted_sources = []
        for source in sources:
            title = source.get('title', 'Unknown Section')
            url = source.get('url', '')
            relevance = round(source.get('relevance_score', 0), 2)
            
            if url:
                formatted_sources.append(f"- [{title}]({url}) (Relevance: {relevance})")
            else:
                formatted_sources.append(f"- {title} (Relevance: {relevance})")
                
        return "\n".join(formatted_sources) if formatted_sources else "No sources available"

def main():
    st.set_page_config(
        page_title="AWS Documentation RAG",
        page_icon="📚",
        layout="wide"
    )
    initialize_components()
    chat = ChatInterface()
    
    st.title("AWS Documentation Assistant")
    
    # Add conversation selector to sidebar
    chat.conversation_selector()
    
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
        if st.button("Clear Chat", type="secondary"):
            st.session_state.current_conversation_id = chat.db.create_conversation()
            st.rerun()

if __name__ == "__main__":
    main()