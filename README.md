# AWS Documentation RAG Application

A Streamlit application that implements RAG (Retrieval-Augmented Generation) for AWS documentation. The application scrapes AWS documentation, processes it into searchable sections, and enables natural language querying with accurate source citations.

A proof-of-concept/prototype. Something I hacked together in two days so definitely not the cleanest implementation. Implementation of a basic RAG is simple, but the idea was to create a "framework" around a production-level RAG, where the users can give feedback to output of the system. This feedback can then be used to first of all track the performance AND the improvement of the RAG system over time, but also to act as a test data to measure the expected performance change after system changes. For example, changing a system prompt can make the results a lot worse and finding it out during test time is necessary.

The implementation so far does not have the most interesting part, the performance measurement/tracking yet, but it is a WIP.

## Features

### Phase 1 (Done)
- Document scraping with hierarchy preservation
- Basic text chunking and vector storage
- Simple similarity search
- Chat interface with conversation history
- Source references for answers

### Phase 2 (Done)
- Enhanced section extraction
- Sliding window chunking with overlap
- Metadata extraction (service names, API references)

### Phase 3 (Done)
  - Conversation memory and context management
  - User feedback system

### Phase 4-6 (Future)
- Answer quality improvements
- Testing framework
- Advanced RAG capabilities
- Hybrid search implementation
- Document preview and source highlighting


## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```
Run the application:
```bash
streamlit run streamlitapp.py
```

## Database Schema
The application uses SQLite with the following main tables:  

urls: Tracks AWS documentation URLs  
sections: Stores hierarchical document sections  
conversations: Manages chat conversations  
messages: Stores chat messages  
message_sources: Links responses to source documents  
message_feedback: Captures user feedback on answers  
source_feedback: Tracks source relevance ratings  

## Usage

1. Add AWS documentation URLs in the Settings page  
2. The scraper will process documents maintaining their hierarchy  
3. Start a conversation and ask questions  
4. View source references and provide feedback  
5. View old conversations or review feedback  
