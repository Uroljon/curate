# CURATE Pipeline Architecture

This document outlines the complete pipeline from PDF upload to structure extraction in the CURATE system.

## Overview

The CURATE pipeline consists of two main phases:
1. **Document Upload**: PDF processing, text extraction, chunking, and embedding
2. **Structure Extraction**: Multi-stage or fast extraction of action fields, projects, measures, and indicators

## Complete Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 1: DOCUMENT UPLOAD                               │
└─────────────────────────────────────────────────────────────────────────────────┘

[PDF File] ──POST /upload──> FastAPI Server
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  1. FILE UPLOAD      │
                        │  src/api/routes.py:44│
                        └──────────┬───────────┘
                                   │ Save to data/uploads/
                                   ▼
                        ┌──────────────────────┐
                        │ 2. TEXT EXTRACTION   │
                        │ parser.py:extract_text_with_ocr_fallback│
                        └──────────┬───────────┘
                                   │
                        ┌──────────┴───────────┐
                        │   PyMuPDF (fitz)     │
                        │  Try text extraction │
                        └──────────┬───────────┘
                                   │
                           ┌───────┴────────┐
                           │ > 10 chars?    │
                           └───────┬────────┘
                                   │
                        ┌──────────┴──────────────┐
                        │                         │
                      ✓ YES                     ✗ NO
                        │                         │
                 [Keep as text]          [OCR FALLBACK]
                        │                         │
                        │              ┌──────────┴──────────┐
                        │              │    Tesseract OCR    │
                        │              │  (German language)  │
                        │              └──────────┬──────────┘
                        │                         │
                        │              ┌──────────┴──────────┐
                        │              │  OCR Quality Filter │
                        │              │  - Language detect  │
                        │              │  - Spell check 60%  │
                        │              │  - Symbol filter 30%│
                        │              └──────────┬──────────┘
                        │                         │
                        └─────────────┬───────────┘
                                      │
                                      ▼
                        ┌─────────────────────────┐
                        │ 3. SEMANTIC CHUNKING    │
                        │ chunker.py              │
                        │ chunk_for_embedding_    │
                        │ enhanced()              │
                        └──────────┬──────────────┘
                                   │
                        ┌──────────┴──────────────┐
                        │ Structure-Aware Split   │
                        │ - Detect headings       │
                        │ - Preserve indicators   │  
                        │ - 5K-7.5K chars/chunk   │
                        └──────────┬──────────────┘
                                   │
                                   ▼
                        ┌─────────────────────────┐
                        │ 4. EMBEDDING GENERATION │
                        │ embedder.py:embed_chunks│
                        │ - Delete old chunks     │
                        │ - Generate embeddings   │
                        │ - Store in ChromaDB     │
                        └─────────────────────────┘
                                   │
                                   ▼
                            [Return: chunks count, quality metrics]

┌─────────────────────────────────────────────────────────────────────────────────┐
│                       PHASE 2: STRUCTURE EXTRACTION                             │
└─────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │     User chooses endpoint:      │
                    └──────────────┬──────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                     │
    POST /extract_structure              POST /extract_structure_fast
    (Multi-stage extraction)             (Single-pass extraction)
                │                                     │
                ▼                                     ▼
    ┌───────────────────────┐          ┌────────────────────────┐
    │ Get ALL chunks from   │          │ Get chunks (with limit)│
    │ ChromaDB by source_id │          │ FAST_EXTRACTION_MAX_   │
    │                       │          │ CHUNKS = 50            │
    └──────────┬────────────┘          └───────────┬────────────┘
               │                                    │
               ▼                                    ▼
    ┌───────────────────────┐          ┌────────────────────────┐
    │ chunk_for_llm()       │          │ chunk_for_llm()        │
    │ Merge to 15K-20K chars│          │ Same merging logic     │
    └──────────┬────────────┘          └───────────┬────────────┘
               │                                    │
               ▼                                    ▼
    ┌───────────────────────┐          ┌────────────────────────┐
    │ STAGE 1: Find Action  │          │ Single-pass extraction │
    │ Fields (all chunks)   │          │ for each chunk:        │
    └──────────┬────────────┘          └───────────┬────────────┘
               │                                    │
               ▼                                    ▼
    ┌───────────────────────┐          ┌────────────────────────┐
    │ STAGE 2: For each     │          │ extract_structures_    │
    │ field, find projects  │          │ with_retry()           │
    └──────────┬────────────┘          └───────────┬────────────┘
               │                                    │
               ▼                                    │
    ┌───────────────────────┐                      │
    │ STAGE 3: For each     │                      │
    │ project, extract      │                      │
    │ measures & indicators │                      │
    └──────────┬────────────┘                      │
               │                                    │
               └────────────┬───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   DEDUPLICATION       │
                │ - Merge same fields   │
                │ - Merge same projects │
                └───────────┬───────────┘
                            │
                            ▼
                     [Return JSON]
```

## Retry & Fallback Mechanisms

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          RETRY & FALLBACK MECHANISMS                            │
└─────────────────────────────────────────────────────────────────────────────────┘

1. OCR FALLBACK (parser.py)
   ├─ Trigger: Page has < 10 characters
   └─ Fallback: Tesseract OCR with quality filtering

2. LLM RETRY (structure_extractor.py:501)
   ├─ Max retries: 1 (configurable via EXTRACTION_MAX_RETRIES)
   ├─ Per chunk in extract_structures_with_retry()
   └─ Fallback: Returns empty list [] if all retries fail

3. JSON REPAIR (llm.py:123)
   ├─ Trigger: Pydantic validation fails
   ├─ Uses json-repair library
   └─ Fallback: Returns None if repair fails

4. STRUCTURED OUTPUT FALLBACK (llm.py:116-129)
   ├─ Primary: Pydantic model validation
   ├─ Fallback 1: JSON repair + retry validation
   └─ Fallback 2: Return None (extraction continues)

5. REQUEST TIMEOUTS
   ├─ Ollama API: 180 seconds (MODEL_TIMEOUT)
   └─ No retry on timeout - fails immediately

6. PROGRESSIVE ACCUMULATION (multi-stage only)
   ├─ Each chunk builds on previous results
   ├─ Never removes data, only adds/merges
   └─ Prevents information loss across chunks
```

## Error Handling Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ERROR HANDLING FLOW                                │
└─────────────────────────────────────────────────────────────────────────────────┘

                            [LLM Query]
                                 │
                    ┌────────────┴────────────┐
                    │ query_ollama_structured │
                    └────────────┬────────────┘
                                 │
                         ┌───────┴────────┐
                         │ HTTP Request   │
                         │ Timeout: 180s  │
                         └───────┬────────┘
                                 │
                        ┌────────┴─────────┐
                        │                  │
                    Success            Failure
                        │                  │
                        ▼                  ▼
                ┌───────────────┐    [Return None]
                │ Parse JSON    │          │
                └───────┬───────┘          │
                        │                  │
               ┌────────┴─────────┐        │
               │                  │        │
           Valid JSON         Invalid      │
               │                  │        │
               ▼                  ▼        │
         [Return Model]    ┌──────────┐   │
                           │ JSON     │   │
                           │ Repair   │   │
                           └────┬─────┘   │
                                │         │
                       ┌────────┴──────┐  │
                       │               │  │
                   Success          Fail  │
                       │               │  │
                       ▼               ▼  ▼
                [Return Model]    [Return None]
                                       │
                                       ▼
                              (Chunk skipped,
                               continue with next)
```

## Key Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MIN_CHARS_FOR_VALID_PAGE` | 10 | Threshold for OCR fallback |
| `SPELL_CHECK_THRESHOLD` | 0.6 | OCR quality filter |
| `SYMBOL_FILTER_THRESHOLD` | 0.3 | OCR symbol filter |
| `SEMANTIC_CHUNK_MAX_CHARS` | 7,500 | Max size for embedding chunks |
| `CHUNK_MAX_CHARS` | 20,000 | Max size for LLM chunks |
| `EXTRACTION_MAX_RETRIES` | 1 | LLM retry attempts |
| `MODEL_TIMEOUT` | 180s | Ollama API timeout |
| `FAST_EXTRACTION_MAX_CHUNKS` | 50 | Chunk limit for fast mode |

## Summary

The CURATE pipeline is designed with multiple fallback mechanisms to ensure robust extraction:

1. **OCR Fallback**: Automatically handles scanned PDFs
2. **Quality Filtering**: Ensures clean text input
3. **Retry Logic**: Handles transient LLM failures
4. **JSON Repair**: Recovers from malformed responses
5. **Progressive Accumulation**: Prevents data loss
6. **Chunk Optimization**: Balances context vs. processing efficiency

The system provides two extraction modes:
- **Multi-stage**: Comprehensive but slower (3 stages)
- **Fast**: Single-pass with optional chunk limiting