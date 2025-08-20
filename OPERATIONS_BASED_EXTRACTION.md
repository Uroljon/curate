# Operations-Based Extraction in CURATE

## Overview

Operations-Based Extraction is CURATE's advanced approach for processing large PDF documents while maintaining entity consistency across chunks. Instead of extracting complete structures from each chunk independently, the LLM returns simple operations (CREATE/UPDATE/CONNECT) that are applied incrementally to build the final knowledge graph.

## Why Operations-Based Approach

### Problems with Traditional Extraction
- **Entity Duplication**: Same entities created with different IDs across chunks
- **Context Degradation**: LLM context becomes bloated with accumulated state
- **Inconsistent Relationships**: Cross-references between entities get lost
- **Copy Degradation**: Quality decreases as more content is accumulated

### Operations-Based Solutions
- **Incremental State Building**: Each chunk contributes operations, not full state
- **No Context Bloat**: LLM only sees current chunk + existing entity summary
- **Consistent Entity IDs**: Global entity registry prevents ID conflicts
- **Deterministic Application**: Operations are applied systematically to build final graph

## Current Implementation Status

### ✅ What Works
- Operations schema with CREATE/UPDATE/CONNECT defined in `src/core/operations_schema.py`
- OperationExecutor applies operations to state in `src/extraction/operations_executor.py`
- API endpoint `/extract_enhanced_operations` available
- Chunking optimized for operations-based processing

### ❌ Critical Missing Piece
**Context-Aware Prompting is NOT implemented** in the operations-based extraction.

**Current Flow (BROKEN):**
```
Chunk 1 → LLM (no context) → Operations → Apply → State₁
Chunk 2 → LLM (no context) → Operations → Apply → State₂  ← PROBLEM: Unaware of State₁
Chunk 3 → LLM (no context) → Operations → Apply → State₃  ← PROBLEM: Unaware of State₁,₂
```

**Should Be (FIXED):**
```
Chunk 1 → LLM (empty state) → Operations → Apply → State₁
Chunk 2 → LLM (aware of State₁) → Operations → Apply → State₂
Chunk 3 → LLM (aware of State₂) → Operations → Apply → State₃
```

## How to Fix the Implementation

### Step 1: Enable Context-Aware Prompting

The functions already exist but are not used:
- `build_entity_context_summary(enhanced_structure)` - Creates summary of existing entities
- `get_available_connection_targets(enhanced_structure)` - Lists entities available for connections
- `get_next_available_ids(global_counters)` - Provides next sequential IDs

**Location**: `src/api/extraction_helpers.py` (lines 827-900+)

### Step 2: Modify Operations Extraction Function

In `extract_direct_to_enhanced_with_operations()`, change the chunk processing loop to:

```python
# Initialize empty state
accumulated_state = create_empty_enhanced_structure()
global_counters = {"af": 0, "proj": 0, "msr": 0, "ind": 0}

for chunk_idx, chunk_data in enumerate(chunks):
    # Build context from accumulated state
    entity_context = build_entity_context_summary(accumulated_state)
    available_targets = get_available_connection_targets(accumulated_state)
    next_ids = get_next_available_ids(global_counters)
    
    # Create context-aware prompt
    context_aware_prompt = f"""
EXISTING ENTITIES IN THE ENHANCED STRUCTURE:
{entity_context}

AVAILABLE CONNECTION TARGETS:
{available_targets}

NEXT AVAILABLE IDS:
{next_ids}

RULES:
1. Only create NEW entities if no similar entity exists
2. Use EXACT IDs from AVAILABLE CONNECTION TARGETS for connections
3. Use NEXT AVAILABLE IDS for new entities
4. When similar entities exist, UPDATE them instead of creating duplicates
"""
    
    # Extract operations with context awareness
    operations = extract_operations_from_chunk(chunk_data, context_aware_prompt)
    
    # Apply operations to accumulated state
    accumulated_state = operation_executor.apply_operations(accumulated_state, operations)
    
    # Update global counters
    update_global_counters(global_counters, operations)
```

### Step 3: Implement Context-Aware Operations Prompt

The operations prompt should include entity awareness:

```python
OPERATIONS_PROMPT_TEMPLATE = """
{context_aware_info}

IHRE AUFGABE: Analysieren Sie die Textpassage und erstellen Sie OPERATIONEN (nicht das vollständige JSON), um die bestehende Extraktionsstruktur zu erweitern.

VERFÜGBARE OPERATIONEN:
- CREATE: Neue Entity erstellen (wenn keine ähnliche Entity existiert)
- UPDATE: Bestehende Entity erweitern/verbessern  
- CONNECT: Verbindungen zwischen Entities erstellen

KRITISCHE REGELN:
1. KEINE DUPLIKATE: Prüfen Sie EXISTING ENTITIES bevor Sie CREATE verwenden
2. GENAUE IDS: Verwenden Sie exakte IDs aus AVAILABLE CONNECTION TARGETS
3. NEUE IDS: Verwenden Sie NEXT AVAILABLE IDS für CREATE-Operationen
4. INTELLIGENTE UPDATES: Nutzen Sie UPDATE um ähnliche Entities zu verbessern

Textpassage:
{chunk_text}
"""
```

## Operations Schema Reference

### CREATE Operation
Creates a new entity when no similar entity exists in current state.

```python
{
    "operation": "CREATE",
    "entity_type": "action_field|project|measure|indicator", 
    "content": {
        "name": "Entity name",
        "description": "Detailed description",
        # ... other fields based on entity type
    }
}
```

### UPDATE Operation  
Modifies existing entity by ID, intelligently merging new information.

```python
{
    "operation": "UPDATE",
    "entity_type": "action_field|project|measure|indicator",
    "entity_id": "af_1",  # Must exist in current state
    "content": {
        "description": "Enhanced description", 
        "new_field": "Additional information"
        # Only fields to add/modify
    }
}
```

### CONNECT Operation
Creates relationships between existing entities.

```python
{
    "operation": "CONNECT",
    "entity_type": "connection",
    "connections": [
        {
            "from_id": "af_1",
            "to_id": "proj_1", 
            "relationship": "contains",
            "confidence": 0.9
        }
    ]
}
```

## Testing the Fix

### 1. Enable Context Awareness
Modify `extract_direct_to_enhanced_with_operations()` to use context-aware prompting.

### 2. Test with Multi-Chunk Document
```bash
# Upload a document with multiple references to same entities
curl -X POST "http://127.0.0.1:8000/upload" -F "file=@test_document.pdf"

# Extract using operations-based approach  
curl -X GET "http://127.0.0.1:8000/extract_enhanced_operations?source_id=your-source-id"
```

### 3. Verify Entity Consistency
Check the result for:
- ✅ No duplicate entities with different IDs
- ✅ Proper cross-chunk entity references  
- ✅ Incremental entity enhancement via UPDATE operations
- ✅ Consistent relationship building via CONNECT operations

### 4. Quality Analysis
```bash
# Run quality analyzer on results
python -m json_analyzer analyze output.json --verbose

# Check for low duplicate rates and high consistency
```

## Expected Improvements After Fix

### Quantitative Improvements
- **80-90% reduction** in duplicate entities across chunks
- **15-20% improvement** in extraction accuracy (based on research)
- **50%+ reduction** in dangling references
- **Consistent entity IDs** across entire document

### Qualitative Improvements  
- Entities get progressively enriched as more chunks reference them
- Cross-chunk relationships properly maintained
- Context builds naturally without exponential complexity
- Operations provide clear audit trail of extraction decisions

## Advanced Enhancements

Once basic context awareness is working, consider:

### 1. LLM-as-Judge Validation
Add validation pass after each chunk's operations:
```python
validation_result = validate_operations_with_second_llm(operations, chunk_text, accumulated_state)
if validation_result.confidence < 0.8:
    operations = refine_operations_with_feedback(operations, validation_result.feedback)
```

### 2. Entity Disambiguation
Post-process final state with fuzzy matching:
```python  
final_state = entity_disambiguator.resolve_remaining_duplicates(accumulated_state)
```

### 3. Multi-Model Ensemble
Run operations through multiple models for consensus:
```python
consensus_operations = ensemble_resolver.get_consensus([
    extract_with_gpt4(chunk, context),
    extract_with_claude(chunk, context), 
    extract_with_llama(chunk, context)
])
```

## Implementation Priority

1. **HIGH PRIORITY**: Enable context-aware prompting in operations extraction
2. **MEDIUM PRIORITY**: Add LLM-as-Judge validation for quality improvement  
3. **LOW PRIORITY**: Implement entity disambiguation and ensemble methods

The context-aware fix alone should provide **80%+ of the benefits** with minimal implementation effort.