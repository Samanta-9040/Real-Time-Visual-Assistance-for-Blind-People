from difflib import SequenceMatcher

def string_similarity(str1, str2):
    """
    Returns a similarity score between 0.0 and 1.0 for two strings.
    """
    if not str1 or not str2:
        return 0.0
    # Clean string comparison (lowercase and strip whitespace)
    s1 = str1.lower().strip()
    s2 = str2.lower().strip()
    return SequenceMatcher(None, s1, s2).ratio()

def is_duplicate_text(new_text, previous_texts, threshold=0.85):
    """
    Checks if new_text is a duplicate of any texts in the previous list.
    """
    for prev in previous_texts:
        if string_similarity(new_text, prev) >= threshold:
            return True
    return False
