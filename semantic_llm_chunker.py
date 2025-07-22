"""
Semantic-aware LLM chunk preparation that respects document structure.

This module provides functions to prepare chunks for LLM processing while
maintaining semantic boundaries identified during initial chunking.
"""

import re
from typing import List, Tuple

from semantic_chunker import is_heading, extract_chunk_topic


def chunk_starts_with_heading(chunk: str) -> bool:
    """Check if a chunk starts with a heading."""
    lines = chunk.strip().split('\n')
    if not lines:
        return False
    return is_heading(lines[0])


def contains_section_marker(chunk: str) -> bool:
    """Check if chunk contains important section markers."""
    section_markers = [
        "Handlungsfeld:", "Handlungsfelder:",
        "Maßnahmen:", "Projekte:", "Ziele:", "Indikatoren:",
        "Ausgangslage:", "Umsetzung:", "Zeitplan:", "Monitoring:",
        "Kapitel", "Abschnitt"
    ]
    
    chunk_lower = chunk.lower()
    return any(marker.lower() in chunk_lower for marker in section_markers)


def get_chunk_priority(chunk: str) -> int:
    """
    Assign priority to chunks based on content importance.
    Higher priority chunks should not be split or merged carelessly.
    """
    priority = 0
    
    # High priority for chunks with section headers
    if chunk_starts_with_heading(chunk):
        priority += 10
    
    # High priority for chunks with key markers
    if contains_section_marker(chunk):
        priority += 8
    
    # Medium priority for chunks with indicators
    indicator_patterns = [
        r'\d+\s*%',  # Percentages
        r'bis\s+20\d{2}',  # Year targets
        r'\d+\s*(km|m²|MW|ha|t)',  # Units
        r'CO2|CO₂',  # Climate indicators
    ]
    
    for pattern in indicator_patterns:
        if re.search(pattern, chunk):
            priority += 2
    
    return priority


def prepare_semantic_llm_chunks(
    chunks: List[str], 
    max_chars: int = 20000, 
    min_chars: int = 15000,
    force_boundaries: bool = True
) -> List[str]:
    """
    Prepare chunks for LLM processing while respecting semantic boundaries.
    
    Args:
        chunks: List of semantically chunked text pieces
        max_chars: Maximum characters per LLM chunk
        min_chars: Minimum characters per LLM chunk (target)
        force_boundaries: If True, never merge chunks that start with headings
    
    Returns:
        List of optimized chunks for LLM processing
    """
    if not chunks:
        return []
    
    result_chunks = []
    current_merge = []
    current_size = 0
    
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        
        chunk_size = len(chunk)
        
        # Check if this chunk starts a new section
        starts_new_section = chunk_starts_with_heading(chunk)
        
        # Decide whether to merge with current or start new
        if not current_merge:
            # First chunk
            current_merge = [chunk]
            current_size = chunk_size
        elif chunk_size > max_chars:
            # Single chunk too large - flush current and handle separately
            if current_merge:
                result_chunks.append("\n\n".join(current_merge))
            result_chunks.append(chunk)  # Add oversized chunk as-is
            current_merge = []
            current_size = 0
        elif force_boundaries and starts_new_section and current_size > 0:
            # This chunk starts a new section - don't merge
            if current_size >= min_chars or len(current_merge) > 1:
                # Current merge is good enough
                result_chunks.append("\n\n".join(current_merge))
                current_merge = [chunk]
                current_size = chunk_size
            else:
                # Current merge too small, but we respect boundaries
                # Check if we can merge without exceeding max
                if current_size + chunk_size + 2 <= max_chars:
                    # Exception: merge if result still under max
                    current_merge.append(chunk)
                    current_size += chunk_size + 2
                else:
                    # Flush small chunk and start new
                    result_chunks.append("\n\n".join(current_merge))
                    current_merge = [chunk]
                    current_size = chunk_size
        elif current_size + chunk_size + 2 > max_chars:
            # Would exceed max size
            result_chunks.append("\n\n".join(current_merge))
            current_merge = [chunk]
            current_size = chunk_size
        else:
            # Can merge
            current_merge.append(chunk)
            current_size += chunk_size + 2  # +2 for \n\n separator
    
    # Don't forget the last merge
    if current_merge:
        result_chunks.append("\n\n".join(current_merge))
    
    # Post-process: ensure quality
    final_chunks = []
    for chunk in result_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        
        # If chunk is too small and doesn't start with heading, 
        # try to merge with previous
        if (len(chunk) < min_chars and 
            not chunk_starts_with_heading(chunk) and 
            final_chunks and 
            len(final_chunks[-1]) + len(chunk) + 2 <= max_chars):
            final_chunks[-1] = final_chunks[-1] + "\n\n" + chunk
        else:
            final_chunks.append(chunk)
    
    return final_chunks


def analyze_chunk_quality(chunks: List[str], stage: str = "unknown") -> dict:
    """Analyze the quality of prepared chunks."""
    if not chunks:
        return {"error": "No chunks provided"}
    
    sizes = [len(chunk) for chunk in chunks]
    
    # Count chunks with section markers
    chunks_with_sections = sum(1 for chunk in chunks if contains_section_marker(chunk))
    
    # Count chunks starting with headings
    chunks_with_headings = sum(1 for chunk in chunks if chunk_starts_with_heading(chunk))
    
    # Find mixed sections (multiple different Handlungsfelder in one chunk)
    mixed_sections = 0
    topic_coherence_stats = {
        "chunks_with_single_topic": 0,
        "chunks_with_multiple_topics": 0,
        "chunks_with_no_topic": 0,
        "topics_found": set()
    }
    
    for chunk in chunks:
        handlungsfeld_count = chunk.count("Handlungsfeld:")
        if handlungsfeld_count > 1:
            mixed_sections += 1
        
        # Analyze topic coherence
        chunk_topics = []
        for line in chunk.split('\n'):
            if "Handlungsfeld:" in line:
                topic_match = re.search(r'Handlungsfeld:\s*(.+)', line)
                if topic_match:
                    topic = topic_match.group(1).strip()
                    chunk_topics.append(topic)
                    topic_coherence_stats["topics_found"].add(topic)
        
        if len(chunk_topics) == 0:
            topic_coherence_stats["chunks_with_no_topic"] += 1
        elif len(chunk_topics) == 1:
            topic_coherence_stats["chunks_with_single_topic"] += 1
        else:
            topic_coherence_stats["chunks_with_multiple_topics"] += 1
    
    # Convert set to list for JSON serialization
    topic_coherence_stats["topics_found"] = list(topic_coherence_stats["topics_found"])
    
    return {
        "stage": stage,
        "total_chunks": len(chunks),
        "size_stats": {
            "avg": sum(sizes) / len(sizes),
            "min": min(sizes),
            "max": max(sizes),
            "total": sum(sizes)
        },
        "heading_stats": {
            "chunks_with_sections": chunks_with_sections,
            "chunks_with_headings": chunks_with_headings,
            "chunks_starting_with_heading": chunks_with_headings
        },
        "mixed_sections": mixed_sections,
        "topic_coherence": topic_coherence_stats,
        "size_distribution": {
            "<10k": sum(1 for s in sizes if s < 10000),
            "10k-15k": sum(1 for s in sizes if 10000 <= s < 15000),
            "15k-20k": sum(1 for s in sizes if 15000 <= s < 20000),
            ">20k": sum(1 for s in sizes if s > 20000),
        }
    }


def prepare_semantic_llm_chunks_v2(
    chunks: List[str], 
    max_chars: int = 20000, 
    min_chars: int = 15000
) -> List[str]:
    """
    Prepare chunks for LLM processing with semantic-aware grouping.
    
    This version groups chunks by topic first, then merges within groups
    to preserve semantic coherence. It prevents mixing different Handlungsfelder
    in the same LLM chunk.
    
    Args:
        chunks: List of semantically chunked text pieces
        max_chars: Maximum characters per LLM chunk
        min_chars: Minimum characters per LLM chunk (target)
    
    Returns:
        List of semantically coherent chunks for LLM processing
    """
    if not chunks:
        return []
    
    # Step 1: Extract topic metadata for each chunk
    chunk_metadata = []
    for i, chunk in enumerate(chunks):
        metadata = extract_chunk_topic(chunk)
        chunk_metadata.append({
            'index': i,
            'chunk': chunk.strip(),
            'size': len(chunk.strip()),
            'topic': metadata['topic'],
            'subtopic': metadata['subtopic'],
            'level': metadata['level'],
            'confidence': metadata['confidence']
        })
    
    # Step 2: Group consecutive chunks by topic
    topic_groups = []
    current_group = []
    current_topic = None
    
    for metadata in chunk_metadata:
        chunk_topic = metadata['topic']
        
        # If no topic detected, use previous topic if confidence is 0
        if not chunk_topic and current_topic and metadata['confidence'] == 0.0:
            chunk_topic = current_topic
        
        # Start new group if topic changes
        if current_topic and chunk_topic != current_topic:
            if current_group:
                topic_groups.append({
                    'topic': current_topic,
                    'chunks': current_group,
                    'total_size': sum(m['size'] for m in current_group)
                })
            current_group = [metadata]
            current_topic = chunk_topic
        else:
            current_group.append(metadata)
            if not current_topic:
                current_topic = chunk_topic
    
    # Don't forget the last group
    if current_group:
        topic_groups.append({
            'topic': current_topic,
            'chunks': current_group,
            'total_size': sum(m['size'] for m in current_group)
        })
    
    # Step 3: Process each topic group
    result_chunks = []
    
    for group in topic_groups:
        group_chunks = group['chunks']
        group_size = group['total_size']
        
        # If entire group fits within max_chars, keep it together
        if group_size <= max_chars:
            merged_text = "\n\n".join(m['chunk'] for m in group_chunks)
            result_chunks.append(merged_text)
        
        # If group is too large, split intelligently
        else:
            # Try to split by subtopics first
            subtopic_groups = []
            current_subtopic_chunks = []
            current_subtopic = None
            
            for metadata in group_chunks:
                if metadata['subtopic'] != current_subtopic and current_subtopic_chunks:
                    subtopic_groups.append(current_subtopic_chunks)
                    current_subtopic_chunks = [metadata]
                    current_subtopic = metadata['subtopic']
                else:
                    current_subtopic_chunks.append(metadata)
                    if not current_subtopic:
                        current_subtopic = metadata['subtopic']
            
            if current_subtopic_chunks:
                subtopic_groups.append(current_subtopic_chunks)
            
            # Merge subtopic groups respecting size limits
            current_merge = []
            current_merge_size = 0
            
            for subtopic_chunk_list in subtopic_groups:
                subtopic_size = sum(m['size'] for m in subtopic_chunk_list)
                
                # If adding this subtopic would exceed max, flush current
                if current_merge and current_merge_size + subtopic_size + 2 > max_chars:
                    merged_text = "\n\n".join(m['chunk'] for m in current_merge)
                    result_chunks.append(merged_text)
                    current_merge = subtopic_chunk_list
                    current_merge_size = subtopic_size
                
                # If single subtopic is too large, split it further
                elif subtopic_size > max_chars:
                    # Flush current merge first
                    if current_merge:
                        merged_text = "\n\n".join(m['chunk'] for m in current_merge)
                        result_chunks.append(merged_text)
                        current_merge = []
                        current_merge_size = 0
                    
                    # Split large subtopic by chunks
                    for metadata in subtopic_chunk_list:
                        if current_merge_size + metadata['size'] + 2 > max_chars and current_merge:
                            merged_text = "\n\n".join(m['chunk'] for m in current_merge)
                            result_chunks.append(merged_text)
                            current_merge = [metadata]
                            current_merge_size = metadata['size']
                        else:
                            current_merge.append(metadata)
                            current_merge_size += metadata['size'] + 2
                
                # Otherwise, add to current merge
                else:
                    current_merge.extend(subtopic_chunk_list)
                    current_merge_size += subtopic_size + (2 if current_merge else 0)
            
            # Don't forget the last merge
            if current_merge:
                merged_text = "\n\n".join(m['chunk'] for m in current_merge)
                result_chunks.append(merged_text)
    
    # Step 4: Post-process to ensure quality
    final_chunks = []
    for chunk in result_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # If chunk is very small and doesn't start with a heading, 
        # try to merge with previous (but only within same topic)
        if (len(chunk) < min_chars and 
            not chunk_starts_with_heading(chunk) and 
            final_chunks and 
            len(final_chunks[-1]) + len(chunk) + 2 <= max_chars):
            
            # Extract topics to ensure they match
            prev_topic = extract_chunk_topic(final_chunks[-1])['topic']
            curr_topic = extract_chunk_topic(chunk)['topic']
            
            # Only merge if topics match and no boundary detected
            if prev_topic == curr_topic or (not curr_topic and not has_topic_boundary(final_chunks[-1], chunk)):
                final_chunks[-1] = final_chunks[-1] + "\n\n" + chunk
            else:
                final_chunks.append(chunk)
        else:
            final_chunks.append(chunk)
    
    return final_chunks


def has_topic_boundary(chunk1: str, chunk2: str) -> bool:
    """
    Check if chunk2 starts with a different topic than chunk1.
    This helps prevent merging chunks across topic boundaries.
    """
    # Extract topic from end of chunk1
    topic1 = extract_chunk_topic(chunk1)['topic']
    
    # Check first 5 lines of chunk2 for topic marker
    lines = chunk2.strip().split('\n')[:5]
    for line in lines:
        # Look for explicit Handlungsfeld marker
        if 'Handlungsfeld:' in line:
            # Extract the topic name using same pattern as in extract_chunk_topic
            match = re.search(r'Handlungsfeld:\s*([A-ZÄÖÜ][a-zäöüß]+(?:\s+(?:und|&)\s+[A-ZÄÖÜ][a-zäöüß]+)*)', line)
            if match:
                topic2 = match.group(1).strip()
                # If topics differ, there's a boundary
                if topic1 and topic2 and topic1 != topic2:
                    return True
                # Even if topic1 is unknown, a new Handlungsfeld marks a boundary
                elif not topic1 and topic2:
                    return True
    
    return False