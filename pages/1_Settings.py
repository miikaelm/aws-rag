import streamlit as st

st.set_page_config(
    page_title="Settings - AWS Documentation RAG",
    page_icon="⚙️",
    layout="wide"
)

from utils.database import init_db, scrape_url, save_scraped_content, get_urls, get_content, add_url, delete_url, get_sections

def display_sections(url_id: int, base_url: str):
    """Display hierarchical content with expandable sections"""
    sections = get_sections(url_id)
    
    if not sections:
        st.write("No content available")
        return

    for section in sections:
        section['full_url'] = f"{base_url}#{section['url_fragment']}"
    
    st.write(sections)

# Modify the settings_page function to include scraping functionality
def settings_page():
    st.title("RAG Settings")
    init_db()

    if 'content_warnings' in st.session_state and st.session_state.content_warnings:
        with st.expander("⚠️ Content Size Warnings", expanded=True):
            for warning in st.session_state.content_warnings:
                if warning['level'] == 'high':
                    st.warning(warning['message'])
                elif warning['level'] == 'medium':
                    st.warning(warning['message'])
                elif warning['level'] == 'low':
                    st.info(warning['message'])
            
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
                success = add_url(new_url, description)
                if success:
                    st.success("URL added successfully!")

    # Modify the URL display section to include scraping
    st.header("Managed URLs")
    urls = get_urls()
    
    if not urls:
        st.info("No URLs have been added yet.")
    else:
        for url_id, url, description, added_date, last_scraped in urls:
            with st.expander(f"{description or url}"):
                st.write(f"**URL:** {url}")
                st.write(f"**Added:** {added_date}")
                
                # Display hierarchical content
                if last_scraped:
                    st.write(f"**Last scraped:** {last_scraped}")
                    st.write("**Document Structure:**")
                    display_sections(url_id, url)
                else:
                    st.write("**Status:** Not yet scraped")
                
                # Scrape and Delete buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Scrape", key=f"scrape_{url_id}"):
                        with st.spinner(f"Scraping {url}..."):
                            try:    
                                result = scrape_url(url)

                                if result is None:
                                    st.error(f"Error during scraping process {url}: {str(e)}")
                                    return

                                title, sections = result
                                if save_scraped_content(url_id, title, sections):
                                    st.success("Content scraped and saved successfully!")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error during scraping process: {str(e)}")
                
                with col2:
                    if st.button(f"Delete", key=f"del_{url_id}"):
                        delete_url(url_id)
                        st.rerun()

if __name__ == "__main__":
    settings_page()
