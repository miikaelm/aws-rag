import streamlit as st
import asyncio
from utils.vector_store import VectorStore
from utils.rag import RAGPipeline
from utils.logger import Logger

st.set_page_config(
    page_title="AWS Documentation RAG",
    page_icon="📚",
    layout="wide"
)

def initialize_rag():
    """Initialize Vector Store and RAG Pipeline"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline(st.session_state.vector_store)
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def display_chat_history():
    """Display the chat history with sources"""
    logger = Logger()
    logger.info(f"Chat History: {st.session_state.chat_history}")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write("You: " + message["content"])
        else:
            with st.container():
                st.write("Assistant: " + message["content"])
                if message.get("sources"):
                    with st.expander("View Sources"):
                        st.markdown(message["sources"], help="Sources used to generate this response")
                        if message.get("confidence"):
                            confidence = round(message['confidence'], 2)
                            st.write(f"Confidence in retrieved sources: {confidence}")

async def process_question(question: str):
    """Process user question through RAG pipeline"""
    try:
        # Get response from RAG pipeline
        response = await st.session_state.rag_pipeline.get_answer(question)
        
        # Format source references
        sources_text = st.session_state.rag_pipeline.format_sources(response.sources)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response.answer,
            "sources": sources_text,
            "confidence": response.confidence
        })
        
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    # Initialize RAG components
    initialize_rag()
    
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
        
        # User input
        user_question = st.text_input("Your question:", key="user_input")
        
        # Process question when submitted
        if user_question:
            if st.session_state.vector_store.get_stats()["total_chunks"] == 0:
                st.warning("Please add and scrape some AWS documentation in the Settings page first!")
            else:
                with st.spinner("Searching documentation..."):
                    # Run async process_question
                    asyncio.run(process_question(user_question))
        
        # Display chat history
        st.divider()
        display_chat_history()
        
        # Add clear chat button
        if st.session_state.chat_history and st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()