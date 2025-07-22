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


def split_by_structure(text: str) -> list[str]:
    """Split text by structural boundaries only (double newlines, page markers)."""
    # First split by OCR page markers
    chunks = []
    current_chunk = []
    
    lines = text.splitlines()
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Handle OCR page markers as boundaries
        if re.match(r"\[OCR Page \d+\]", line):
            if current_chunk:
                chunks.append("\n".join(current_chunk).strip())
                current_chunk = []
        
        current_chunk.append(line)
        
        # Check for double newline (paragraph boundary)
        if i + 1 < len(lines) and line.strip() == "" and lines[i + 1].strip() == "":
            if len("\n".join(current_chunk)) > 500:  # Only split if chunk is substantial
                chunks.append("\n".join(current_chunk).strip())
                current_chunk = []
        
        i += 1
    
    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())
    
    # Filter out empty chunks
    chunks = [c for c in chunks if c.strip()]
    
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
    # Simple structural splitting
    chunks = split_by_structure(cleaned_text)
    
    # Merge short chunks while respecting max size
    from config import SEMANTIC_CHUNK_MIN_CHARS
    final_chunks = merge_short_chunks(
        chunks, 
        min_chars=SEMANTIC_CHUNK_MIN_CHARS,
        max_chars=max_chars
    )
    
    return final_chunks