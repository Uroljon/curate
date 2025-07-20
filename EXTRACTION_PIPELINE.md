# CURATE PDF Extraction Pipeline: Technical Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [System Architecture Overview](#system-architecture-overview)
3. [Phase 1: Document Ingestion](#phase-1-document-ingestion)
4. [Phase 2: Intelligent Text Extraction](#phase-2-intelligent-text-extraction)
5. [Phase 3: Semantic Document Chunking](#phase-3-semantic-document-chunking)
6. [Phase 4: Vector Embedding and Storage](#phase-4-vector-embedding-and-storage)
7. [Phase 5: Multi-Stage Information Extraction](#phase-5-multi-stage-information-extraction)
8. [Output and Results](#output-and-results)
9. [Technical Innovations](#technical-innovations)
10. [Handling Edge Cases](#handling-edge-cases)

## Introduction

CURATE is a specialized system designed to extract structured information from German municipal strategy documents. These documents, often hundreds of pages long, contain crucial information about city planning initiatives, climate protection measures, digitalization efforts, and social programs. The challenge lies in transforming these unstructured PDF documents into actionable, structured data that can be analyzed and tracked.

The system processes documents that vary widely in quality—from professionally typeset PDFs with clear structure to scanned documents with OCR challenges. It must understand German administrative language, recognize hierarchical document structures, and extract not just what cities plan to do, but also how they plan to measure success.

## System Architecture Overview

The pipeline consists of five major phases, each addressing specific challenges in document processing:

1. **Document Ingestion**: Accepting and validating PDF uploads
2. **Intelligent Text Extraction**: Converting PDF content to processable text
3. **Semantic Document Chunking**: Breaking documents into meaningful segments
4. **Vector Embedding and Storage**: Enabling semantic search capabilities
5. **Multi-Stage Information Extraction**: Systematically extracting structured data

Each phase builds upon the previous one, with careful error handling and quality checks throughout the process.

## Phase 1: Document Ingestion

When a user uploads a PDF document, the system first performs basic validation to ensure the file is indeed a PDF and is not corrupted. The document is assigned a unique identifier that will track it through the entire pipeline. This identifier is crucial for maintaining data lineage and enabling users to query the status of their extraction jobs.

The system stores the original PDF in a designated upload directory, maintaining a complete audit trail. This allows for re-processing if improvements are made to the extraction pipeline, and provides a reference point for quality assurance checks.

## Phase 2: Intelligent Text Extraction

This phase represents one of the most complex challenges in the pipeline. Municipal PDFs come in various formats:

### Digital-Native PDFs
These are PDFs created directly from digital sources like Word documents. They contain embedded text that can be extracted directly. The system uses PyMuPDF (also known as fitz) as the primary extraction tool for these documents. PyMuPDF is particularly effective because it:
- Preserves text formatting and structure
- Maintains reading order across complex layouts
- Handles multi-column formats common in municipal documents
- Extracts text with positional information for future enhancement

### Scanned Documents
Many older municipal strategies exist only as scanned documents. The system detects these by checking the character count per page—if a page yields fewer than 10 characters through direct extraction, it's flagged for OCR processing.

The OCR process uses Tesseract, configured specifically for German language recognition. However, OCR introduces unique challenges:

**Quality Control**: OCR often produces garbled text, especially with poor scan quality. The system implements several quality measures:
- **Language Detection**: Each line of OCR output is checked to ensure it's actually German or English text
- **Spell Checking**: Using German language dictionaries, the system calculates the ratio of recognized words to total words
- **Character Pattern Analysis**: Lines consisting mostly of symbols or random characters are filtered out

### Text Cleaning and Normalization

Once text is extracted, whether through direct extraction or OCR, it undergoes extensive cleaning:
- **Hyphenation Handling**: German documents often break words across lines with hyphens. These are intelligently merged.
- **Page Number Removal**: Headers and footers with page numbers are identified and removed
- **Whitespace Normalization**: Excessive spaces, tabs, and newlines are normalized
- **Bullet Point Standardization**: Various bullet point symbols are converted to a standard format

## Phase 3: Semantic Document Chunking

Traditional text chunking methods that split at arbitrary character counts would destroy the semantic structure of municipal documents. Instead, CURATE uses intelligent chunking that respects document hierarchy.

### Heading Detection
The system identifies section boundaries by recognizing:
- **Numbered Sections**: Patterns like "1.1", "2.3.4" that indicate hierarchical structure
- **Title Case Headers**: Sections that begin with capitalized words
- **Uppercase Headers**: Fully capitalized section titles
- **Special Markers**: Keywords like "Kapitel", "Abschnitt", or "Teil"

### Chunk Optimization
After initial splitting, chunks undergo optimization:
- **Small Chunk Merging**: Chunks below a minimum threshold are merged with adjacent chunks
- **Large Chunk Splitting**: Oversized chunks are split at paragraph boundaries to maintain readability
- **Semantic Coherence**: The system ensures that related content stays together

This approach typically transforms a 200-page document into 15-25 semantically meaningful chunks, each representing a complete thought or section.

## Phase 4: Vector Embedding and Storage

Each chunk is transformed into a high-dimensional vector representation using sentence transformer models. These models are specifically chosen for their multilingual capabilities and understanding of German text.

### Embedding Process
- **Model Selection**: The system uses models trained on multilingual data with strong German language representation
- **Batch Processing**: Chunks are processed in batches for efficiency
- **Dimensionality**: Typically 384-768 dimensional vectors that capture semantic meaning

### Storage Architecture
The embeddings are stored in ChromaDB, a vector database optimized for similarity search:
- **Metadata Preservation**: Each embedding is stored with its source text, chunk index, and document ID
- **Indexing**: Efficient indexing structures enable fast similarity searches
- **Persistence**: The database persists across system restarts

This vector storage enables semantic search capabilities—finding chunks based on meaning rather than just keyword matching.

## Phase 5: Multi-Stage Information Extraction

This is where CURATE's innovation truly shines. The system uses a three-stage extraction process, each stage building upon the previous one with increasing specificity.

### Stage 1: Action Field Discovery

The first stage casts a wide net across all document chunks with a simple question: "What are the main thematic areas (Handlungsfelder) in this document?"

**Process Details**:
- Each chunk is analyzed independently to find category names
- The LLM is instructed to ignore project details and focus only on high-level themes
- Common action fields include: Klimaschutz (Climate Protection), Mobilität (Mobility), Digitalisierung (Digitalization), Bildung (Education), etc.

**Intelligent Merging**:
After all chunks are processed, the system merges similar action fields:
- "Klimaschutz" and "Klimaschutz und Energie" become one field (the more comprehensive name is kept)
- "Mobilität" and "Verkehr und Mobilität" are recognized as the same concept
- Pure duplicates are automatically removed

This stage typically identifies 8-15 main action fields for a comprehensive municipal strategy.

### Stage 2: Project Identification

For each action field discovered in Stage 1, the system now searches for specific projects and initiatives.

**Targeted Search**:
- Only chunks containing keywords related to the action field are processed
- The LLM receives clear context: "Find all projects related to [Klimaschutz] in this text"
- This focused approach prevents confusion between hierarchical levels

**Project Recognition**:
The system is trained to recognize various project indicators:
- Named initiatives: "Stadtbahn Regensburg", "Smart City Initiative"
- Program titles: "Förderprogramm Energieeffizienz"
- Specific plans: "Masterplan Radverkehr", "Digitalisierungsstrategie 2030"

**Quality Control**:
- Generic statements are filtered out
- Only concrete, named projects are retained
- Duplicate project names are merged

### Stage 3: Detail Extraction

The final stage performs surgical extraction of measures and indicators for each identified project.

**Chunk Selection**:
- The system uses fuzzy matching to find all chunks mentioning a specific project
- Multiple mentions across different sections are processed to gather complete information

**Measure Extraction**:
Measures are concrete actions or steps, identified by:
- Action verbs: "errichten" (construct), "einführen" (introduce), "verbessern" (improve)
- Implementation language: "Umsetzung von", "Durchführung der"
- Specific activities: "Bau von Radwegen", "Installation von Solaranlagen"

**Indicator Extraction** (The Most Challenging Part):
The system looks for quantitative metrics with special attention to:
- **Percentages**: "Reduktion um 40%", "Steigerung auf 75%"
- **Temporal Targets**: "bis 2030", "innerhalb von 5 Jahren", "ab 2025"
- **Quantities**: "500 neue Ladestationen", "10 km Radweg", "1000 Wohneinheiten"
- **Frequencies**: "jährlich", "monatlich", "pro Quartal"
- **Comparisons**: "Verdopplung", "Halbierung", "30% weniger als 2020"

The indicator extraction uses extensive example-based prompting to help the LLM recognize these patterns in context.

## Output and Results

The final output is a hierarchical JSON structure that preserves the relationships discovered through the three stages:

- Each action field contains multiple projects
- Each project contains its associated measures and indicators
- Metadata includes extraction confidence and source chunk references

The system also provides comprehensive statistics:
- Total number of action fields discovered
- Number of projects per action field
- Count of projects with measures vs. those with indicators
- Overall extraction quality metrics

## Technical Innovations

### Context Preservation Without Accumulation
Previous approaches faced a dilemma: 
- Progressive extraction understood context but lost information through over-consolidation
- Parallel extraction preserved everything but lost semantic understanding

The multi-stage approach solves this by maintaining context through focused queries rather than accumulated state.

### Intelligent Prompt Engineering
Each stage uses carefully crafted prompts with:
- German-language examples
- Clear boundary definitions
- Negative examples (what NOT to extract)
- Progressive refinement of focus

### Quality Through Modularity
Each stage can be optimized independently:
- Stage 1 prompts can be refined for better action field recognition
- Stage 2 can implement better project name parsing
- Stage 3 can add new indicator patterns as they're discovered

## Handling Edge Cases

### Poor Document Structure
Some PDFs lack clear hierarchical structure. The system handles this by:
- Falling back to paragraph-level analysis
- Using keyword clustering to identify implicit sections
- Applying more aggressive text mining techniques

### OCR Quality Issues
When OCR quality is poor:
- Multiple OCR passes with different parameters
- Contextual correction using surrounding text
- Fallback to more conservative extraction

### Ambiguous Categorization
When projects could belong to multiple action fields:
- The system maintains the association found in the source text
- Cross-references are preserved in metadata
- Users can review and adjust categorizations

### Missing Indicators
Many German municipal documents describe projects without quantitative targets. The system:
- Flags projects lacking indicators for manual review
- Attempts to extract implicit targets from surrounding context
- Maintains extraction logs for quality assurance

## Future Enhancements

The modular architecture enables several planned improvements:
- **Language Models**: Testing with different models optimized for German administrative text
- **Active Learning**: Using human feedback to improve extraction patterns
- **Cross-Document Learning**: Applying patterns learned from high-quality documents to improve extraction from poor-quality ones
- **Temporal Analysis**: Tracking how municipal strategies evolve over time

This pipeline represents a significant advancement in automated document analysis for German municipal governance, transforming hundreds of pages of plans into actionable, trackable data structures.