# Home.py
import streamlit as st

st.set_page_config(
    page_title="AWS Documentation RAG",
    page_icon="📚",
    layout="wide"
)

def main():
    st.title("AWS Documentation Assistant")
    
    st.write("""
    Welcome to the AWS Documentation Assistant! This tool helps you search and understand AWS documentation.
    
    Use the sidebar to:
    - Chat with the documentation
    - Manage documentation sources in Settings
    """)
    
    # Main chat interface here
    with st.container():
        st.subheader("Ask about AWS")
        user_question = st.text_input("Your question:")
        if user_question:
            st.write("Your answer will appear here...")

if __name__ == "__main__":
    main()