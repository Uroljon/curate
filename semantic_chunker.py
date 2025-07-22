import re
from typing import List, Tuple, Optional

# Indicator patterns for German municipal documents
INDICATOR_PATTERNS = {
    'percentage': r'\d+(?:[,\.]\d+)?\s*%',
    'currency': r'\d+(?:[,\.]\d+)?\s*(?:Millionen|Mio\.?|Tsd\.?|Mrd\.?)?\s*(?:Euro|EUR|€)',
    'measurement': r'\d+(?:[,\.]\d+)?\s*(?:km|m²|qm|MW|GW|kW|ha|t|kg|m)',
    'time_target': r'(?:bis|ab|von|nach|vor|zum|zur)\s+(?:20\d{2}|Jahr\s+20\d{2})',
    'quantity': r'\d+(?:[,\.]\d+)?\s*(?:neue|mehr|weniger|zusätzliche)',
    'rate': r'\d+(?:[,\.]\d+)?\s*%?\s*(?:pro|je|per)\s+(?:Jahr|Monat|Tag|Einwohner|km)',
    'compound': r'\d+(?:[,\.]\d+)?\s*(?:von|aus)\s+\d+',  # "5 von 10", "3 aus 20"
}


def contains_indicator_context(text: str, position: int, window: int = 150) -> Tuple[bool, Optional[int]]:
    """
    Check if there's an indicator pattern near a split position.
    
    Args:
        text: The full text
        position: The proposed split position
        window: How many characters before/after to check
    
    Returns:
        (has_indicator, safe_split_position)
    """
    # Get context around the split point
    start = max(0, position - window)
    end = min(len(text), position + window)
    context = text[start:end]
    
    # Check each indicator pattern
    for pattern_name, pattern in INDICATOR_PATTERNS.items():
        matches = list(re.finditer(pattern, context, re.IGNORECASE))
        
        for match in matches:
            # Get actual position in original text
            match_start = start + match.start()
            match_end = start + match.end()
            
            # If split would occur within or very close to indicator
            if match_start - 50 <= position <= match_end + 50:
                # Find a safer split position
                # Try after the indicator context
                safe_pos = match_end + 50
                if safe_pos < len(text):
                    # Look for sentence end
                    sentence_end = text.find('. ', match_end, safe_pos + 100)
                    if sentence_end != -1:
                        return True, sentence_end + 2
                
                # Try before the indicator context
                safe_pos = match_start - 50
                if safe_pos > 0:
                    # Look for sentence end before indicator
                    sentence_end = text.rfind('. ', safe_pos - 100, match_start)
                    if sentence_end != -1:
                        return True, sentence_end + 2
                
                return True, None
    
    return False, position


def find_safe_split_point(text: str, target_pos: int, max_chars: int) -> int:
    """
    Find a safe position to split text that doesn't break indicator context.
    
    Args:
        text: The text to split
        target_pos: The desired split position
        max_chars: Maximum allowed characters
    
    Returns:
        Safe split position
    """
    # First check if there's an indicator near the target position
    has_indicator, safe_pos = contains_indicator_context(text, target_pos)
    
    if has_indicator and safe_pos:
        # Use the safe position if it's within bounds
        if safe_pos <= max_chars:
            return safe_pos
    
    # Try to find paragraph boundary near target
    para_before = text.rfind('\n\n', max(0, target_pos - 200), target_pos)
    para_after = text.find('\n\n', target_pos, min(len(text), target_pos + 200))
    
    if para_before != -1 and (target_pos - para_before) < 200:
        return para_before + 2
    elif para_after != -1 and (para_after - target_pos) < 200:
        return para_after
    
    # Try to find sentence boundary
    sentence_before = text.rfind('. ', max(0, target_pos - 100), target_pos)
    if sentence_before != -1:
        return sentence_before + 2
    
    # Last resort: return original position
    return target_pos


def is_heading(line: str) -> bool:
    """Heuristic: check if a line looks like a section heading.
    
    Enhanced for German municipal documents with common heading patterns.
    """
    line = line.strip()
    if len(line) > 100 or len(line) < 3:
        return False
    
    # Numbered sections (e.g. "1. Klimaschutz", "2.1 Maßnahmen", "III. Ziele")
    if re.match(r"^\d{1,2}(\.\d{1,2})*\.?\s+\w", line):
        return True
    if re.match(r"^[IVX]+\.\s+\w", line):  # Roman numerals
        return True
    if re.match(r"^[a-z]\)\s+\w", line):  # Letter enumeration "a) Ziel"
        return True
    
    # German document structure keywords
    german_structure_keywords = [
        "Kapitel", "Abschnitt", "Teil", "Anlage", "Anhang",
        "Maßnahmen:", "Projekte:", "Ziele:", "Indikatoren:",
        "Handlungsfeld:", "Handlungsfelder:", "Zielstellung:",
        "Ausgangslage:", "Umsetzung:", "Zeitplan:", "Monitoring:"
    ]
    for keyword in german_structure_keywords:
        if line.startswith(keyword):
            return True
    
    # All uppercase (common for main sections)
    if line.isupper() and len(line.split()) <= 8 and len(line) > 3:
        return True
    
    # Title Case (allowing German characters and common prepositions)
    # Allow lowercase words like "für", "und", "der", etc. in the middle
    if re.match(r"^[A-ZÄÖÜ][a-zäöüß]+(\s+([A-ZÄÖÜ\-][a-zäöüß]+|für|und|der|die|das|von|zu|mit|in|auf))*$", line):
        # Extra check: should be relatively short and not a regular sentence
        word_count = len(line.split())
        # Must have at least 2 words or be longer than 10 chars for single words
        if (word_count >= 2 or len(line) > 10) and word_count <= 6 and not line.endswith(('.', ',', ';')):
            return True
    
    # Lines ending with colon (often introduce sections)
    if line.endswith(':') and 3 < len(line) < 50:
        return True
    
    return False


def extract_chunk_topic(chunk: str) -> dict:
    """
    Extract the primary topic and subtopic from a chunk.
    
    Returns metadata about the chunk's semantic context:
    - topic: Main Handlungsfeld or major section
    - subtopic: Project, measure, or subsection
    - level: Hierarchy level (1 for main topics, 2 for subtopics)
    - confidence: How confident we are in the classification
    """
    lines = chunk.strip().split('\n')
    topic = None
    subtopic = None
    level = 0
    confidence = 0.0
    
    # Primary topic patterns (Handlungsfeld level)
    primary_patterns = [
        (r'Handlungsfeld:\s*([A-ZÄÖÜ][a-zäöüß]+(?:\s+(?:und|&)\s+[A-ZÄÖÜ][a-zäöüß]+)*)', 'handlungsfeld'),  # Match topic words only
        (r'Handlungsfelder:\s*([A-ZÄÖÜ][a-zäöüß]+(?:\s+(?:und|&)\s+[A-ZÄÖÜ][a-zäöüß]+)*)', 'handlungsfeld'),
        (r'^\d+\.\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)', 'numbered_section'),  # "1. Klimaschutz"
        (r'^[IVX]+\.\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*)', 'roman_section'),  # "III. Mobilität"
    ]
    
    # Subtopic patterns
    subtopic_patterns = [
        (r'Projekte?:\s*(.+)', 'project'),
        (r'Maßnahmen?:\s*(.+)', 'measure'),
        (r'Ziele?:\s*(.+)', 'goal'),
        (r'Indikatoren?:\s*(.+)', 'indicator'),
        (r'^\d+\.\d+\s+(.+)$', 'subsection'),  # "2.1 Unterpunkt"
    ]
    
    # Check entire chunk for topic indicators, but prioritize early occurrences
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Check for primary topics
        for pattern, pattern_type in primary_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                level = 1
                confidence = 1.0 if pattern_type == 'handlungsfeld' else 0.9
                
                # Look for subtopic in next few lines
                for j in range(i+1, min(i+5, len(lines))):
                    subline = lines[j].strip()
                    for subpattern, _ in subtopic_patterns:
                        submatch = re.match(subpattern, subline, re.IGNORECASE)
                        if submatch:
                            subtopic = submatch.group(1).strip()
                            break
                    if subtopic:
                        break
                # Found primary topic - reduce confidence if found late in chunk
                if i > 10:
                    confidence = confidence * 0.8  # Reduce confidence for late finds
                return {
                    'topic': topic,
                    'subtopic': subtopic,
                    'level': level,
                    'confidence': confidence
                }
        
        # If no primary topic yet, check for subtopics
        if not topic:
            for pattern, pattern_type in subtopic_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    subtopic = match.group(1).strip()
                    level = 2
                    confidence = 0.7
                    break
    
    # If no explicit topic found, try to infer from content
    if not topic and not subtopic:
        # Check for topic keywords in the chunk
        topic_keywords = {
            'Klimaschutz': ['CO2', 'Emission', 'Klimawandel', 'Treibhausgas', 'erneuerbar'],
            'Mobilität': ['Verkehr', 'ÖPNV', 'Radweg', 'Stadtbahn', 'Modal Split'],
            'Stadtentwicklung': ['Quartier', 'Siedlung', 'Baukultur', 'Wohnen'],
            'Digitalisierung': ['digital', 'IT', 'Smart City', 'Daten'],
            'Freiräume': ['Grünfläche', 'Park', 'Biotop', 'ökologisch'],
        }
        
        # Only apply keyword fallback to substantial chunks
        if len(chunk) > 500:
            chunk_lower = chunk.lower()
            chunk_length = len(chunk)
            
            for topic_name, keywords in topic_keywords.items():
                keyword_count = sum(1 for kw in keywords if kw.lower() in chunk_lower)
                # Require at least 3 keywords OR high keyword density
                keyword_density = (keyword_count * 1000) / chunk_length  # keywords per 1000 chars
                
                if keyword_count >= 3 or (keyword_count >= 2 and keyword_density > 2.0):
                    topic = topic_name
                    # Lower confidence for keyword-based assignment
                    confidence = min(0.5 + keyword_count * 0.05, 0.7)
                    level = 1
                    break
    
    return {
        'topic': topic,
        'subtopic': subtopic,
        'level': level,
        'confidence': confidence
    }


def split_by_heading(text: str) -> list[str]:
    """Split text by section headings, preserving OCR tags.
    
    Handles multi-line German headings by checking if consecutive lines
    form a heading pattern together.
    """
    lines = text.splitlines()
    chunks = []
    current = []
    i = 0

    while i < len(lines):
        line = lines[i]
        
        # Handle OCR page markers
        if re.match(r"\[OCR Page \d+\]", line):
            if current:
                chunks.append("\n".join(current).strip())
                current = []
            current.append(line)
            i += 1
            continue

        # Check for multi-line headings (common in German documents)
        is_multiline_heading = False
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Check if current + next line form a heading pattern
            # e.g., "Handlungsfeld 1:" on one line, "Klimaschutz" on next
            if (line.strip().endswith(':') and len(line.strip()) < 30 and 
                next_line and len(next_line) < 50 and 
                next_line and not (next_line[0].islower() if next_line else False)):
                is_multiline_heading = True
            # Check for numbered heading continuing on next line
            elif (re.match(r"^\d{1,2}(\.\d{1,2})*\.?\s*$", line.strip()) and
                  next_line and not (next_line[0].islower() if next_line else False)):
                is_multiline_heading = True

        if is_heading(line) and current:
            chunks.append("\n".join(current).strip())
            current = [line]
            if is_multiline_heading:
                i += 1
                current.append(lines[i])
        elif is_multiline_heading and current:
            chunks.append("\n".join(current).strip())
            current = [line, lines[i + 1]]
            i += 1
        else:
            current.append(line)
        
        i += 1

    if current:
        chunks.append("\n".join(current).strip())

    return chunks


def split_large_chunk(text: str, max_chars: int) -> list[str]:
    """Split a chunk that's too large into smaller pieces at natural boundaries."""
    if len(text) <= max_chars:
        return [text]
    
    # Try to split by double newlines (paragraphs) first
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 1:
        result = []
        current = []
        current_size = 0
        
        for i, para in enumerate(paragraphs):
            para_size = len(para) + (2 if current else 0)  # +2 for \n\n
            
            if current_size + para_size > max_chars and current:
                # Check if we should keep this paragraph with current chunk
                # to preserve indicator context
                combined_text = '\n\n'.join(current + [para])
                
                # Check if splitting here would break indicators
                split_pos = len('\n\n'.join(current))
                has_indicator, safe_pos = contains_indicator_context(combined_text, split_pos)
                
                if has_indicator and safe_pos and safe_pos > split_pos and safe_pos <= max_chars:
                    # Keep the paragraph to preserve indicator
                    current.append(para)
                    current_size += para_size
                else:
                    # Safe to split here
                    result.append('\n\n'.join(current))
                    current = [para]
                    current_size = len(para)
            else:
                current.append(para)
                current_size += para_size
        
        if current:
            result.append('\n\n'.join(current))
        
        # Recursively split any still-too-large chunks
        final_result = []
        for chunk in result:
            if len(chunk) > max_chars:
                # Try splitting by single newlines with indicator awareness
                final_result.extend(split_by_lines(chunk, max_chars))
            else:
                final_result.append(chunk)
        
        return final_result
    
    # If no paragraphs, try splitting by lines
    return split_by_lines(text, max_chars)


def split_by_lines(text: str, max_chars: int) -> list[str]:
    """Split text by lines when paragraph splitting isn't enough."""
    lines = text.split('\n')
    result = []
    current = []
    current_size = 0
    i = 0
    
    while i < len(lines):
        line = lines[i]
        line_size = len(line) + (1 if current else 0)  # +1 for \n
        
        if current_size + line_size > max_chars and current:
            # Check if this line or next few lines contain indicators
            # that relate to content in current chunk
            look_ahead_lines = []
            look_ahead_size = 0
            
            # Look ahead up to 3 lines for related indicators
            for j in range(i, min(i + 3, len(lines))):
                look_ahead_lines.append(lines[j])
                look_ahead_size += len(lines[j]) + 1
                
                # Check if these lines contain indicators
                lookahead_text = '\n'.join(look_ahead_lines)
                has_indicators = any(re.search(pattern, lookahead_text, re.IGNORECASE) 
                                   for pattern in INDICATOR_PATTERNS.values())
                
                # If we found indicators and can fit them, keep looking
                if has_indicators and current_size + look_ahead_size <= max_chars:
                    continue
                else:
                    break
            
            # If we found related indicators that fit, include them
            if has_indicators and current_size + look_ahead_size <= max_chars:
                for j in range(len(look_ahead_lines)):
                    current.append(look_ahead_lines[j])
                    current_size += len(look_ahead_lines[j]) + 1
                # Skip the lines we just added
                i += len(look_ahead_lines)
                continue
            else:
                # Find safe split point considering current content
                combined_text = '\n'.join(current + [line])
                split_pos = len('\n'.join(current))
                
                has_indicator, safe_pos = contains_indicator_context(combined_text, split_pos)
                
                if has_indicator and safe_pos and safe_pos <= max_chars:
                    # Try to include the line to preserve context
                    current.append(line)
                    current_size += line_size
                else:
                    # Safe to split here
                    result.append('\n'.join(current))
                    current = [line]
                    current_size = len(line)
        else:
            current.append(line)
            current_size += line_size
        
        i += 1
    
    if current:
        result.append('\n'.join(current))
    
    return result


def merge_short_chunks(chunks: list[str], min_chars=3000, max_chars=5000) -> list[str]:
    """Merge small chunks to optimize for embedding and LLM processing."""
    merged = []
    buffer = ""

    for chunk in chunks:
        if not chunk.strip():
            continue
        
        # First check if chunk itself is too large
        if len(chunk) > max_chars:
            # Flush buffer first
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            # Split the large chunk
            split_chunks = split_large_chunk(chunk, max_chars)
            merged.extend(split_chunks)
            continue

        combined = buffer + "\n\n" + chunk if buffer else chunk
        if len(combined) < max_chars:
            buffer = combined
        else:
            if buffer:
                merged.append(buffer.strip())
            buffer = chunk

    if buffer.strip():
        merged.append(buffer.strip())

    return merged


def smart_chunk(cleaned_text: str, max_chars: int = 5000) -> list[str]:
    """Create semantic chunks respecting document structure and size limits."""
    chunks = split_by_heading(cleaned_text)
    # Use configured min chars for semantic chunks
    from config import SEMANTIC_CHUNK_MIN_CHARS
    final_chunks = merge_short_chunks(
        chunks, 
        min_chars=SEMANTIC_CHUNK_MIN_CHARS,
        max_chars=max_chars
    )
    return final_chunks
