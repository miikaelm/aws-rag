import streamlit as st
import pandas as pd
from utils.logger import Logger

st.set_page_config(
    page_title="Logs - AWS Documentation RAG",
    page_icon="📋",
    layout="wide"
)

def display_log_stats(logs):
    """Display statistics about logs"""
    if not logs:
        return
        
    # Create columns for stats
    col1, col2, col3, col4 = st.columns(4)
    
    # Total logs
    with col1:
        st.metric("Total Logs", len(logs))
    
    # Count by level
    level_counts = pd.DataFrame(logs)['level'].value_counts()
    
    with col2:
        st.metric("Errors", level_counts.get('ERROR', 0))
    with col3:
        st.metric("Warnings", level_counts.get('WARNING', 0))
    with col4:
        st.metric("Info", level_counts.get('INFO', 0))

def logs_page():
    st.title("📋 Application Logs")
    
    logger = Logger()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Log Controls")
        
        # Number of lines to display
        n_lines = st.number_input(
            "Number of lines to display",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )
        
        # Level filter
        level_filter = st.selectbox(
            "Filter by level",
            ["All", "ERROR", "WARNING", "INFO", "DEBUG"]
        )
        
        # Clear logs button
        if st.button("Clear All Logs"):
            if logger.clear_logfile():
                st.success("Log file cleared!")
                st.rerun()
    
    # Get logs with filters
    logs = logger.get_log_contents(
        n_lines=n_lines,
        level_filter=None if level_filter == "All" else level_filter
    )
    
    # Display stats
    display_log_stats(logs)
    
    # Display logs in a dataframe
    if logs:
        st.header("Log Entries")
        
        # Convert to dataframe for display
        df = pd.DataFrame(logs)
        
        # Apply custom styling
        def style_log_level(val):
            color_map = {
                'ERROR': '#ffcdd2',
                'WARNING': '#fff9c4',
                'INFO': '#c8e6c9',
                'DEBUG': '#f5f5f5'
            }
            return f'background-color: {color_map.get(val, "#ffffff")}'
        
        # Display with styling
        st.dataframe(
            df,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Timestamp"),
                "level": st.column_config.TextColumn("Level"),
                "message": st.column_config.TextColumn("Message", width="large"),
            },
            hide_index=True
        )
        
        # Export option
        if st.button("Export Logs to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "logs.csv",
                "text/csv",
                key='download-csv'
            )
    else:
        st.info("No logs found matching the current filters.")

if __name__ == "__main__":
    logs_page()