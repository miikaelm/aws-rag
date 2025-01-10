import streamlit as st
from typing import Dict, List, Optional
import asyncio
from datetime import datetime
from utils.vector_store import VectorStore
from utils.rag import RAGPipeline
from utils.logger import Logger
from db.models.conversation import Conversation
from db.models.message import Message
from db.models.message_source import MessageSource
from db.models.message_feedback import MessageFeedback
from db.models.source_feedback import SourceFeedback
from db.connection import get_db
from utils.scraper import DocumentScraper

def initialize_components():
    """Initialize all necessary components"""
    # Initialize database instance
    get_db()
    
    # Initialize other components
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline(st.session_state.vector_store)
        
    if 'processing' not in st.session_state:
        st.session_state.processing = False
        
    if 'scraper' not in st.session_state:
        st.session_state.scraper = DocumentScraper()

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
        
        # Load available conversations
        self.conversations = Conversation.get_all()
        
        # Initialize conversation state
        if 'current_conversation_id' not in st.session_state:
            self._initialize_active_conversation()
            
        # Initialize feedback states
        if 'feedback_states' not in st.session_state:
            st.session_state.feedback_states = {}

    def _initialize_active_conversation(self):
        """Set the active conversation"""
        if self.conversations:
            st.session_state.current_conversation_id = self.conversations[0].id
        else:
            # Create new conversation if none exists
            new_conv = Conversation.create()
            st.session_state.current_conversation_id = new_conv.id

    def conversation_selector(self):
        """Display conversation selector in the sidebar"""
        with st.sidebar:
            st.subheader("Conversations")
            
            # New conversation button
            if st.button("New Conversation", type="secondary"):
                new_conv = Conversation.create()
                st.session_state.current_conversation_id = new_conv.id
                st.rerun()
            
            st.divider()
            
            # List existing conversations
            for conv in self.conversations:
                col1, col2 = st.columns([4, 1])
                with col1:
                    preview = getattr(conv, 'preview', '') or 'New Conversation'
                    button_text = (
                        f"{conv.title or 'New Conversation'}\n"
                        f"{datetime.fromisoformat(str(conv.updated_at)).strftime('%Y-%m-%d %H:%M')}\n"
                        f"{preview}"
                    )
                    
                    if st.button(
                        button_text,
                        key=f"conv_{conv.id}",
                        type="secondary" if conv.id == st.session_state.current_conversation_id else "tertiary",
                        use_container_width=True
                    ):
                        st.session_state.current_conversation_id = conv.id
                        st.rerun()
                
                # Delete conversation button
                with col2:
                    if st.button("🗑️", key=f"del_{conv.id}", type="secondary"):
                        conv_obj = Conversation(id=conv.id)
                        conv_obj.delete()
                        if conv.id == st.session_state.current_conversation_id:
                            self._initialize_active_conversation()
                        st.rerun()

    def display_chat_history(self):
        """Display current conversation messages"""
        messages = Message.get_conversation_messages(st.session_state.current_conversation_id)
        
        # Display conversation title if it exists
        current_conv = next(
            (c for c in self.conversations if c.id == st.session_state.current_conversation_id), 
            None
        )
        if current_conv and current_conv.title:
            st.caption(f"Current conversation: {current_conv.title}")
        
        for msg in messages:
            if msg.role == "user":
                with st.chat_message("user"):
                    st.write(msg.content)
            
            elif msg.role == "assistant":
                with st.chat_message("assistant"):
                    st.write(msg.content)
                    
                    # Initialize feedback state for this message
                    if msg.id not in st.session_state.feedback_states:
                        st.session_state.feedback_states[msg.id] = {
                            'show_feedback': False,
                            'relevance': 3,
                            'accuracy': 3,
                            'feedback_text': '',
                            'source_ratings': {}
                        }
                    
                    # Feedback button
                    col1, col2 = st.columns([6, 1])
                    with col2:
                        if st.button("Rate Answer", key=f"fb_{msg.id}"):
                            st.session_state.feedback_states[msg.id]['show_feedback'] = True
                    
                    # Show feedback form if button was clicked
                    if st.session_state.feedback_states[msg.id]['show_feedback']:
                        with st.expander("Provide Feedback", expanded=True):
                            # Message feedback
                            st.write("Rate the answer:")
                            col1, col2 = st.columns(2)
                            with col1:
                                relevance = st.slider(
                                    "Relevance",
                                    1, 5, 
                                    st.session_state.feedback_states[msg.id]['relevance'],
                                    key=f"rel_{msg.id}"
                                )
                            with col2:
                                accuracy = st.slider(
                                    "Accuracy",
                                    1, 5,
                                    st.session_state.feedback_states[msg.id]['accuracy'],
                                    key=f"acc_{msg.id}"
                                )
                            
                            feedback_text = st.text_area(
                                "Additional feedback (optional):",
                                value=st.session_state.feedback_states[msg.id]['feedback_text'],
                                key=f"txt_{msg.id}"
                            )
                            
                            if st.button("Submit Feedback", key=f"submit_{msg.id}"):
                                try:
                                    # Save message feedback
                                    message_feedback = MessageFeedback(
                                        message_id=msg.id,
                                        answer_relevance=relevance,
                                        answer_accuracy=accuracy,
                                        feedback_text=feedback_text
                                    )
                                    message_feedback.save()
                                    
                                    # Save source feedback
                                    for source_id, rating in st.session_state.feedback_states[msg.id]['source_ratings'].items():
                                        source_feedback = SourceFeedback(
                                            message_source_id=source_id,
                                            rating=rating
                                        )
                                        source_feedback.save()
                                    
                                    st.success("Thank you for your feedback!")
                                    st.session_state.feedback_states[msg.id]['show_feedback'] = False
                                    
                                except Exception as e:
                                    self.logger.error(f"Error saving feedback: {str(e)}")
                                    st.error("Error saving feedback. Please try again.")
                    
                    # Display sources with feedback options
                    if msg.sources:
                        with st.expander("View Sources"):
                            st.markdown(self._format_sources(msg.sources))
                            
                            # Source feedback if feedback form is shown
                            if st.session_state.feedback_states[msg.id]['show_feedback']:
                                st.write("Rate the relevance of each source:")
                                for source in msg.sources:
                                    if source.id not in st.session_state.feedback_states[msg.id]['source_ratings']:
                                        st.session_state.feedback_states[msg.id]['source_ratings'][source.id] = 3
                                    
                                    source_rating = st.slider(
                                        f"Source: {source.title}",
                                        1, 5,
                                        st.session_state.feedback_states[msg.id]['source_ratings'][source.id],
                                        key=f"src_{msg.id}_{source.id}"
                                    )
                                    st.session_state.feedback_states[msg.id]['source_ratings'][source.id] = source_rating
                            
                            if msg.confidence is not None:
                                st.progress(msg.confidence)
                                st.caption(f"Confidence Score: {msg.confidence}")

    async def process_question(self, question: str):
        """Process user question and save using Message model"""
        try:
            # Create and save user message
            user_msg = Message(
                conversation_id=st.session_state.current_conversation_id,
                role="user",
                content=question
            )
            user_msg.save()

            # Get response from RAG pipeline
            response = await st.session_state.rag_pipeline.get_answer(question)
            
            # Create and save assistant message
            assistant_msg = Message(
                conversation_id=st.session_state.current_conversation_id,
                role="assistant",
                content=response.answer,
                confidence=response.confidence
            )
            assistant_msg.save()
            
            # Save sources if available
            if hasattr(response, 'sources') and assistant_msg.id:
                for source in response.sources:
                    source['message_id'] = assistant_msg.id
                    MessageSource(**source).save()

        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}")
            st.error("I encountered an error while processing your question. Please try again.")

    def _format_sources(self, sources: List['MessageSource']) -> str:
        """Format source references for display in markdown"""
        formatted_sources = []
        for source in sources:
            title = getattr(source, 'title', 'Unknown Section')
            url = getattr(source, 'url', '')
            relevance = round(getattr(source, 'relevance_score', 0), 2)
            
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
            new_conv = Conversation.create()
            st.session_state.current_conversation_id = new_conv.id
            st.rerun()

if __name__ == "__main__":
    main()