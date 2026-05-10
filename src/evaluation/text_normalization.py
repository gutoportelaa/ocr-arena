import re
import unicodedata

def normalize_text(text: str) -> str:
    """Basic text normalization for comparison."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove accents
    text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    
    # Replace multiple whitespaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters except alphanumeric and basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?;:-]', '', text)
    
    return text.strip()

def to_markdown_comparable(text: str) -> str:
    """Normalize markdown for structural comparison."""
    # Simplified markdown normalization
    if not text:
        return ""
    
    # Ensure consistent line endings
    text = text.replace('\r\n', '\n')
    
    # Remove excessive empty lines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()
