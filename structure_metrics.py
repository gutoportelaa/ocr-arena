import difflib
from typing import Dict

def calculate_edit_distance_ratio(text1: str, text2: str) -> float:
    """Calculate normalized Levenshtein distance ratio."""
    return difflib.SequenceMatcher(None, text1, text2).ratio()

def calculate_structural_score(md1: str, md2: str) -> Dict[str, float]:
    """Calculate structural similarity metrics between two markdown strings."""
    # Count structural elements
    def count_elements(md: str) -> Dict[str, int]:
        return {
            "headers": len(re.findall(r'^#+', md, re.MULTILINE)),
            "lists": len(re.findall(r'^[*-] ', md, re.MULTILINE)),
            "tables": len(re.findall(r'\|', md)),
            "bold": len(re.findall(r'\*\*.*?\*\*', md))
        }
    
    import re
    counts1 = count_elements(md1)
    counts2 = count_elements(md2)
    
    scores = {}
    for key in counts1:
        max_val = max(counts1[key], counts2[key])
        if max_val == 0:
            scores[f"{key}_similarity"] = 1.0
        else:
            scores[f"{key}_similarity"] = 1.0 - abs(counts1[key] - counts2[key]) / max_val
            
    return scores
