import streamlit as st

st.set_page_config(
    page_title="Settings - AWS Documentation RAG",
    page_icon="⚙️",
    layout="wide"
)

from utils.database import get_database
from utils.vector_store import VectorStore
from utils.text_processing import prepare_sections_for_indexing

def display_sections(db, url_id: int, base_url: str):
    """Display hierarchical content with expandable sections"""
    sections = db.get_sections(url_id)
    
    if not sections:
        st.write("No content available")
        return

    # Create a more readable display format
    section_display = []
    for section in sections:
        section_display.append({
            'Title': section.title,
            'Level': section.level,
            'URL': f"{base_url}#{section.url_fragment}" if section.url_fragment else base_url,
            'Content Length': len(section.content) if section.content else 0,
            'Path': section.path or section.title
        })
    
    # Display as a table
    st.table(section_display)

def settings_page():
    # Get database instance
    db = get_database()
    
    st.title("RAG Settings")

    if 'content_warnings' in st.session_state and st.session_state.content_warnings:
        with st.expander("⚠️ Content Size Warnings", expanded=True):
            for warning in st.session_state.content_warnings:
                # Handle both dictionary and dataclass formats for backward compatibility
                if isinstance(warning, dict):
                    level = warning['level']
                    message = warning['message']
                else:
                    level = warning.level
                    message = warning.message
                    
                if level == 'high':
                    st.warning(message)
                elif level == 'medium':
                    st.warning(message)
                elif level == 'low':
                    st.info(message)
            
            if st.button("Clear Warnings"):
                st.session_state.content_warnings = []
                st.rerun()

    # Add new URL section
    st.header("Add New Documentation URL")
    with st.form("add_url_form"):
        new_url = st.text_input("URL", placeholder="https://docs.aws.amazon.com/...")
        description = st.text_input("Description", placeholder="e.g., AWS Lambda Documentation")
        submitted = st.form_submit_button("Add URL")
        
        if submitted and new_url:
            if not new_url.startswith(('http://', 'https://')):
                st.error("Please enter a valid URL starting with http:// or https://")
            else:
                success = db.add_url(new_url, description)
                if success:
                    st.success("URL added successfully!")
                    st.rerun()

    # Display URLs section
    st.header("Managed URLs")
    urls = db.get_urls()
    
    if not urls:
        st.info("No URLs have been added yet.")
    else:
        # Initialize vector store
        vector_store = VectorStore()
        
        for url in urls:
            with st.expander(f"{url.description or url.url}"):
                st.write(f"**URL:** {url.url}")
                st.write(f"**Added:** {url.added_date}")
                
                # Display hierarchical content
                if url.last_scraped:
                    st.write(f"**Last scraped:** {url.last_scraped}")
                    st.write("**Document Structure:**")
                    display_sections(db, url.id, url.url)
                else:
                    st.write("**Status:** Not yet scraped")
                
                # Scrape and Delete buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Scrape", key=f"scrape_{url.id}"):
                        with st.spinner(f"Scraping {url.url}..."):
                            try:    
                                result = db.scraper.scrape_url(url.url)
                                if result is None:
                                    st.error("Error during scraping process")
                                    return

                                title, sections = result
                                if db.save_sections(url.id, title, sections):
                                    # Get saved sections
                                    saved_sections = db.get_sections(url.id)
                                    
                                    # Convert sections to dictionary format for vector store
                                    section_dicts = [{
                                        'id': section.id,
                                        'title': section.title,
                                        'content': section.content,
                                        'level': section.level,
                                        'url_fragment': section.url_fragment,
                                        'path': section.path
                                    } for section in saved_sections]
                                    
                                    # Process sections for vector store
                                    documents = prepare_sections_for_indexing(section_dicts, url.url)
                                    
                                    # Add to vector store
                                    if vector_store.add_section_chunks(documents, url.id):
                                        st.success("Content scraped and indexed successfully!")
                                        st.rerun()
                                    else:
                                        st.error("Error adding content to vector store")
                            except Exception as e:
                                st.error(f"Error during scraping process: {str(e)}")
                
                with col2:
                    if st.button(f"Delete", key=f"del_{url.id}"):
                        db.delete_url(url.id)
                        vector_store.delete_url_content(url.id)
                        st.rerun()

if __name__ == "__main__":
    settings_page()