from dataclasses import dataclass
from typing import Optional

@dataclass
class ContentWarning:
    level: str
    message: str
    title: str
    content_length: int

class ContentProcessor:
    """Handles content processing and warnings"""
    
    WARNING_THRESHOLD = 1700  # Current max
    MEDIUM_THRESHOLD = 2500   # Getting large
    LARGE_THRESHOLD = 4000    # Needs attention
    
    @staticmethod
    def check_content_length(content: str, title: str) -> Optional[ContentWarning]:
        """Check content length and return warning if needed"""
        content_length = len(content)
        
        if content_length > ContentProcessor.LARGE_THRESHOLD:
            return ContentWarning(
                level='high',
                message=f"🚨 Section '{title}' is very large ({content_length} chars) and may need chunking",
                title=title,
                content_length=content_length
            )
        elif content_length > ContentProcessor.MEDIUM_THRESHOLD:
            return ContentWarning(
                level='medium',
                message=f"⚠️ Section '{title}' is large ({content_length} chars) and may need chunking",
                title=title,
                content_length=content_length
            )
        elif content_length > ContentProcessor.WARNING_THRESHOLD:
            return ContentWarning(
                level='low',
                message=f"ℹ️ Section '{title}' is approaching size limit ({content_length} chars)",
                title=title,
                content_length=content_length
            )
        return None