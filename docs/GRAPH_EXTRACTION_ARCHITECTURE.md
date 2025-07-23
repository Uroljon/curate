# Graph-Based Streaming Extraction Architecture for CURATE

## Executive Summary

This document outlines a complete architectural redesign of CURATE's PDF extraction system, implementing a graph-based streaming extraction pipeline with multi-agent orchestration, template learning, and progressive enhancement. This architecture fundamentally reimagines document processing as building a living knowledge graph rather than extracting isolated chunks.

## System Overview

The new architecture processes German municipal strategy documents through a sophisticated pipeline that constructs a knowledge graph in real-time as it streams through the document. Unlike traditional extraction methods that process chunks independently, this system maintains full document context, learns from previous extractions, and prevents information duplication through intelligent graph construction.

### Core Philosophy

Documents are not collections of independent text chunks but interconnected networks of concepts, relationships, and hierarchies. Our architecture treats each document as a potential knowledge graph where:

- Action fields connect to multiple projects
- Projects contain various measures
- Measures link to specific indicators
- Indicators reference temporal targets and baselines
- All entities maintain bidirectional relationships

## Architecture Components

### 1. Document Intelligence Layer

#### Document Structure Analyzer
The first component that touches any PDF, operating without LLM assistance to maximize speed and minimize cost. It performs deep structural analysis through:

- **Hierarchical Pattern Recognition**: Identifies German municipal document patterns including numbered sections (1.1, 1.2.1), heading styles, and table of contents structures
- **Layout Analysis**: Detects multi-column layouts, tables, figures, and their relationships to surrounding text
- **Template Fingerprinting**: Creates unique document signatures based on structural patterns, enabling rapid template matching
- **Language and Domain Detection**: Identifies not just German language but specific municipal terminology patterns

#### Visual Layout Processor
Leverages computer vision techniques to understand document structure beyond text:

- **Region Detection**: Identifies headers, footers, sidebars, and main content areas
- **Table Structure Recognition**: Extracts complex tables maintaining cell relationships
- **Figure-Text Association**: Links charts and diagrams to their textual descriptions
- **Spatial Relationship Mapping**: Understands which text elements belong together based on visual proximity

### 2. Knowledge Graph Core

#### Dynamic Graph Engine
The central nervous system of the architecture, maintaining a living representation of extracted knowledge:

- **Multi-Model Graph Structure**: Supports property graphs, RDF triples, and hypergraphs simultaneously
- **Embedding-Powered Nodes**: Each node contains both structured data and dense vector embeddings
- **Confidence-Weighted Edges**: Relationships carry confidence scores and provenance information
- **Temporal Versioning**: Tracks how entities and relationships evolve through the document

#### Semantic Deduplication Engine
Prevents duplicate information at the point of entry rather than post-processing:

- **Multi-Level Similarity Detection**: Operates at syntactic, semantic, and conceptual levels
- **Context-Aware Thresholds**: Adjusts similarity thresholds based on entity type and document section
- **Merge Strategies**: Implements sophisticated merging rules that preserve maximum information
- **Conflict Resolution**: Handles contradictory information through confidence scoring and source tracking

### 3. Streaming Processing Pipeline

#### Stateful Stream Processor
Maintains extraction state as it processes the document linearly:

- **Sliding Context Windows**: Maintains overlapping windows of configurable size (typically 3-5 chunks)
- **State Machine Management**: Tracks current extraction context (in measures section, reading indicators, etc.)
- **Progressive Enhancement**: Builds understanding incrementally, refining earlier extractions with later context
- **Checkpoint System**: Creates restoration points for error recovery and parallel processing

#### Adaptive Chunking System
Intelligently segments documents based on content rather than arbitrary boundaries:

- **Semantic Boundary Detection**: Identifies natural breaking points in German text
- **Dynamic Chunk Sizing**: Adjusts chunk size based on content density and LLM context windows
- **Cross-Reference Preservation**: Ensures references ("siehe Abschnitt 3.2") maintain connectivity
- **Table and List Handling**: Keeps structured elements intact within chunks

### 4. Multi-Agent Orchestration Layer

#### Agent Coordination Framework
Manages a team of specialized extraction agents working in concert:

- **Message Bus Architecture**: Enables asynchronous communication between agents
- **Task Distribution Engine**: Routes document sections to appropriate specialist agents
- **Consensus Mechanisms**: Resolves conflicts when multiple agents extract contradicting information
- **Resource Management**: Optimizes LLM usage across agents to minimize costs

#### Specialized Extraction Agents

**Structure Extraction Agent**
- Identifies document hierarchy and sectioning
- Extracts table of contents and cross-references
- Maps relationships between document sections

**Quantitative Analysis Agent**
- Specializes in numerical data extraction
- Understands German number formats, units, and percentages
- Links quantities to their semantic context

**Temporal Reasoning Agent**
- Extracts deadlines, timelines, and temporal relationships
- Understands relative time expressions ("bis Ende nächsten Jahres")
- Creates temporal graphs of project schedules

**Relationship Mapping Agent**
- Identifies connections between entities
- Extracts hierarchical relationships (parent-child)
- Maps dependencies and prerequisites

**Indicator Specialist Agent**
- Focuses exclusively on KPIs and measurement criteria
- Understands baseline vs. target values
- Links indicators to responsible parties and deadlines

**Validation and Consistency Agent**
- Performs cross-validation of extracted information
- Ensures logical consistency across the graph
- Identifies missing or suspicious data

### 5. Template Learning System

#### Pattern Recognition Engine
Learns from successful extractions to improve future performance:

- **Structural Pattern Library**: Stores successful extraction patterns for different document types
- **Extraction Path Recording**: Tracks which strategies led to successful extractions
- **Feature Engineering**: Automatically identifies discriminative features of document types
- **Pattern Generalization**: Abstracts specific patterns into reusable templates

#### Adaptive Strategy Selection
Chooses optimal extraction strategies based on document characteristics:

- **Similarity Matching**: Finds previously processed similar documents
- **Strategy Composition**: Combines multiple successful strategies for hybrid documents
- **Performance Prediction**: Estimates extraction quality before processing
- **Fallback Mechanisms**: Maintains multiple backup strategies for difficult documents

### 6. Quality Assurance Pipeline

#### Multi-Level Validation System
Ensures extraction quality through comprehensive validation:

- **Schema Validation**: Verifies extracted data against predefined schemas
- **Semantic Validation**: Checks logical consistency of extracted information
- **Cross-Reference Validation**: Ensures all references resolve correctly
- **Completeness Checking**: Identifies potentially missing information

#### Confidence Scoring Framework
Assigns confidence scores at multiple levels:

- **Entity-Level Confidence**: Based on extraction clarity and consistency
- **Relationship Confidence**: Evaluates strength of identified connections
- **Document-Level Confidence**: Aggregates individual scores into overall quality metric
- **Human-in-the-Loop Triggers**: Automatically flags low-confidence extractions for review

### 7. Storage and Retrieval Infrastructure

#### Graph Database Layer
Persistent storage optimized for relationship queries:

- **Native Graph Storage**: Uses dedicated graph databases for optimal traversal performance
- **Hybrid Storage Strategy**: Combines graph storage with vector databases for embeddings
- **Indexing Strategies**: Multiple index types for different query patterns
- **Sharding and Replication**: Ensures scalability and reliability

#### Intelligent Query Engine
Supports complex queries across the knowledge graph:

- **Natural Language Interface**: Converts German questions into graph queries
- **Multi-Hop Reasoning**: Traverses relationships to answer complex questions
- **Aggregation Capabilities**: Summarizes information across multiple entities
- **Temporal Queries**: Supports time-based filtering and analysis

## Processing Workflow

### Phase 1: Document Ingestion and Analysis

The system begins by accepting a PDF through the ingestion API. The Document Structure Analyzer performs a rapid scan without LLM assistance, creating a structural map of the document. This includes identifying major sections, detecting the document template, and creating a processing plan.

Simultaneously, the Visual Layout Processor analyzes the PDF's visual structure, identifying tables, figures, and layout patterns that inform the extraction strategy. The system creates a document fingerprint and searches the Template Library for similar previously processed documents.

### Phase 2: Strategic Planning

Based on the structural analysis and template matching results, the Adaptive Strategy Selection component creates an extraction plan. This plan determines:

- Which agents to deploy for which sections
- Optimal chunk sizing for different document parts
- Confidence thresholds for deduplication
- Expected extraction patterns based on similar documents

### Phase 3: Streaming Extraction

The document enters the Streaming Processing Pipeline, where the Stateful Stream Processor begins sequential processing. As each chunk is processed:

1. The Adaptive Chunking System creates semantically meaningful segments
2. The Agent Coordination Framework routes chunks to appropriate specialist agents
3. Multiple agents may process the same chunk from different perspectives
4. Results are immediately integrated into the growing Knowledge Graph

The system maintains a sliding window of context, allowing later chunks to refine earlier extractions. The State Machine tracks progress through the document, adjusting extraction strategies based on current location.

### Phase 4: Progressive Enhancement

As the Knowledge Graph grows, several enhancement processes run continuously:

- The Semantic Deduplication Engine merges similar entities
- The Relationship Mapping Agent identifies new connections
- The Validation Agent ensures consistency across the graph
- Confidence scores are continuously updated based on new evidence

### Phase 5: Quality Assurance

Once initial extraction completes, the Quality Assurance Pipeline performs comprehensive validation:

- Schema compliance verification
- Logical consistency checking
- Completeness analysis against expected patterns
- Cross-reference resolution

Low-confidence extractions are flagged for potential human review, with specific guidance on what requires attention.

### Phase 6: Learning and Adaptation

After successful extraction, the Template Learning System:

- Records successful extraction patterns
- Updates the pattern library with new insights
- Adjusts confidence thresholds based on outcomes
- Refines agent strategies for future documents

## Technical Infrastructure Requirements

### Compute Resources

The system requires substantial computational resources:

- **LLM Infrastructure**: Multiple concurrent LLM instances for multi-agent processing
- **GPU Acceleration**: For embedding generation and vision processing
- **High-Memory Servers**: Graph processing requires 2-3x memory of traditional approaches
- **Distributed Processing**: Agent coordination benefits from distributed architecture

### Model Requirements

- **Primary LLM**: Qwen2.5:14b or larger for complex reasoning tasks
- **Specialist Models**: Smaller focused models for specific extraction tasks
- **Embedding Models**: German-optimized models like deepset/gbert-large
- **Vision Models**: For layout analysis and table extraction

### Storage Systems

- **Graph Database**: Neo4j or Amazon Neptune for relationship storage
- **Vector Database**: Pinecone or Weaviate for embedding search
- **Document Store**: S3-compatible storage for original PDFs
- **Cache Layer**: Redis for intermediate state management

## Integration Points

### API Design

The system exposes a comprehensive API:

- **Document Ingestion**: Accepts PDFs with metadata
- **Extraction Monitoring**: Real-time progress tracking
- **Query Interface**: Natural language and structured queries
- **Learning Feedback**: Accepts corrections to improve future extractions
- **Template Management**: CRUD operations for extraction templates

### Event Streaming

All processing generates events for monitoring and integration:

- Document ingestion events
- Agent processing milestones
- Extraction completion notifications
- Quality assurance results
- Learning system updates

## Performance Characteristics

### Expected Metrics

Based on validated research and similar systems:

- **Extraction Accuracy**: 85-95% for well-structured documents
- **Processing Time**: 45-90 seconds per document
- **Deduplication Effectiveness**: 95%+ reduction in redundant information
- **Memory Usage**: 2-3GB per concurrent document
- **LLM Token Usage**: 15-20x traditional single-pass extraction

### Scalability Considerations

The architecture scales through:

- Horizontal scaling of agent workers
- Graph database sharding
- Caching of common patterns
- Parallel processing of document sections
- Async processing with queue management

## Future Enhancements

### Planned Capabilities

- **Multi-Language Support**: Extending beyond German
- **Real-time Collaboration**: Multiple users annotating extractions
- **Automated Report Generation**: Creating summaries from graphs
- **Change Detection**: Tracking modifications across document versions
- **Integration APIs**: Direct connection to municipal planning systems

### Research Directions

- Self-supervised learning from extraction outcomes
- Zero-shot transfer to new document types
- Automated quality improvement without human feedback
- Cross-document knowledge graph construction
- Real-time extraction during document creation

## Conclusion

This graph-based streaming extraction architecture represents a paradigm shift in document processing. By treating documents as interconnected knowledge networks rather than isolated text chunks, the system achieves superior extraction quality, eliminates redundancy by design, and continuously improves through learning.

The architecture's complexity is justified by the high-value nature of municipal strategy documents, where accuracy and completeness directly impact urban planning and citizen services. While requiring significant computational resources, the system's benefits—including dramatic improvements in extraction quality, relationship preservation, and adaptive learning—provide substantial return on investment for organizations processing complex strategic documents.