CREATE_TABLES_SQL = {
    'urls': '''
        CREATE TABLE IF NOT EXISTS urls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL UNIQUE,
            description TEXT,
            added_date TIMESTAMP,
            last_scraped TIMESTAMP NULL
        )
    ''',
    'content': '''
        CREATE TABLE IF NOT EXISTS content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url_id INTEGER,
            title TEXT,
            content TEXT,
            scraped_date TIMESTAMP,
            FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE
        )
    ''',
    'sections': '''
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url_id INTEGER,
            parent_id INTEGER NULL,
            title TEXT,
            content TEXT,
            level INTEGER,
            url_fragment TEXT,
            section_order INTEGER,
            FOREIGN KEY (url_id) REFERENCES urls (id) ON DELETE CASCADE,
            FOREIGN KEY (parent_id) REFERENCES sections (id) ON DELETE CASCADE
        )
    ''',
    'conversations': '''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
    ''',
    'messages': '''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_version TEXT,
            confidence FLOAT,
            message_order INTEGER NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    ''',
    'message_sources': '''
        CREATE TABLE IF NOT EXISTS message_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            url TEXT,
            content TEXT NOT NULL,
            relevance_score FLOAT,
            FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
        )
    ''',
    'message_feedback': '''
        CREATE TABLE IF NOT EXISTS message_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL,
            answer_relevance INTEGER CHECK (answer_relevance BETWEEN 1 AND 5),
            answer_accuracy INTEGER CHECK (answer_accuracy BETWEEN 1 AND 5),
            feedback_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
        )
    ''',
    'source_feedback': '''
        CREATE TABLE IF NOT EXISTS source_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_source_id INTEGER NOT NULL,
            rating INTEGER CHECK (rating BETWEEN 1 AND 5),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (message_source_id) REFERENCES message_sources(id) ON DELETE CASCADE
        )
    '''
}