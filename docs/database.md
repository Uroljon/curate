# Database Structure & Entity Relationships

This document captures insights from analyzing the Appwrite database schema (`appwrite.ts`) to understand how extracted data should be structured and connected.

## Core Operational Hierarchy

The primary structure for day-to-day municipal sustainability work follows this hierarchy:

**Dimension → Measure → Indicator**

### Dimensions (Action Fields/Handlungsfelder)
Strategic sustainability areas - the foundational building blocks.

**Appwrite Entity**: `Dimensions`
```typescript
{
  name: string,                    // "Mobilität und Verkehr"
  description: string,             // What this field encompasses
  sustainabilityType?: string,     // "Environmental", "Social", "Economic"
  strategicGoal?: string[],        // High-level objectives
  sdgs?: string[],                 // UN SDG alignment ["SDG 11", "SDG 13"]
  parentDimensionId?: string,      // Hierarchical relationships
  iconRef?: string,                // Visual representation
  partnerId: string                // Multi-tenant organization ID
}
```

### Measures (Projects/Implementation Initiatives)
Concrete implementation projects that can belong to multiple Dimensions.

**Appwrite Entities**: `Measures` + `MeasuresExtended` (two-table structure)

**Measures** (Core Data):
```typescript
{
  title: string,                   // "Radverkehrsnetz Ausbau"
  description?: string,            // Brief overview
  fullDescription?: string,        // Complete details
  type: string,                    // "Infrastructure", "Policy", etc.
  status?: string,                 // "In Planung", "Aktiv", "Abgeschlossen"
  measureStart?: string,           // "2024-01-01"
  measureEnd?: string,             // "2026-12-31"
  budget?: number,                 // 2500000
  department?: string,             // "Tiefbauamt"
  responsiblePreson?: string[],    // ["Max Mustermann"]
  operativeGoal?: string,          // Specific objective
  parentMeasure?: string[],        // Sub-project relationships
  isParent?: boolean,              // Has child projects
  sdgs?: string[],                 // SDG alignment
  partnerId: string                // Organization ID
}
```

**MeasuresExtended** (Relationship Metadata):
```typescript
{
  measureId: string,               // Reference to base measure
  indicatorIds?: string[],         // Connected metrics
  dimensionIds?: string[],         // Action fields this serves
  relatedUrls?: string[],          // Reference links
  costs?: string[],                // Detailed cost breakdown
  budgetOrigin?: string[],         // Funding sources
  milestones?: string,             // Implementation milestones
  published?: boolean,             // Publication status
  publishedUser?: string,          // Who published
  publishedDate?: string,          // When published
  partnerId: string                // Organization ID
}
```

### Indicators (Quantitative Metrics)
Metrics that can span multiple Dimensions and connect to multiple Measures.

**Appwrite Entity**: `Indicators`
```typescript
{
  title: string,                   // "CO2-Reduktion Verkehrssektor"
  description: string,             // "Jährliche CO2-Einsparung durch Maßnahmen"
  unit?: string,                   // "Tonnen CO2/Jahr"
  granularity: string,             // "annual", "monthly", "quarterly"
  targetValues?: string,           // "500 Tonnen bis 2030"
  actualValues?: string,           // "120 Tonnen (2023)"
  shouldIncrease?: boolean,        // false (less CO2 is better)
  calculation?: string,            // "Baseline - Current emissions"
  valuesSource?: string,           // "Umweltamt Monitoring"
  sourceUrl?: string,              // Reference URL
  operationalGoal?: string,        // What it tracks specifically
  dimensionIds?: string[],         // Multiple action fields it spans
  sdgs?: string[],                 // SDG alignment
  partnerId: string                // Organization ID
}
```

## Junction Tables (Many-to-Many Relationships)

### Measure2indicator
Links measures to the indicators that track their progress:
```typescript
{
  measureId: string,
  indicatorId: string
}
```

### StrategiesRelations
Links strategic documents to their constituent elements:
```typescript
{
  strategyId: string,
  relationType: "Dimension" | "Measure" | "Indicator" | "Post",
  relationItemId: string,
  relationSubType?: string,
  partnerId: string
}
```

## Strategy as Contextual Layer

**Important**: Strategies are NOT hierarchical parents. They are contextual groupings.

**Appwrite Entity**: `Strategies`
```typescript
{
  name: string,                    // "Klimaschutzkonzept 2030"
  description?: string,            // Strategic document description
  type?: "Concept" | "Strategy",   // Document type
  strategicGoals?: string[],       // High-level goals
  sdgs?: string[],                 // SDG alignment
  partnerId: string                // Organization ID
}
```

**Key Insight**: A single Dimension, Measure, or Indicator can be referenced by multiple Strategy documents. For example, "Sustainable Mobility" dimension could be part of both "Climate Action Plan" and "City Master Plan 2040".

## Implications for PDF Extraction

### Focus Areas
1. **Operational entities**: Dimensions → Measures → Indicators as core structure
2. **Many-to-many relationships**: Capture when projects serve multiple action fields
3. **Rich metadata**: Extract all the detailed information shown in the schema
4. **Document context**: Treat the PDF as a Strategy document that references existing or new operational entities

### Data Structure Mapping
- **Our ExtractionResult**: Hierarchical (action_fields → projects → measures/indicators) - good for initial extraction
- **Our EnrichedReviewJSON**: Flat 4-bucket with connections - better for final storage, matches Appwrite structure
- **Transform needed**: From hierarchical to relational many-to-many structure

### Meeting Requirements (August 2025)
- **Always with explanation**: Extract description/fullDescription fields
- **Headlines & explanation**: Capture both brief and detailed descriptions
- **Fill existing schema**: Map to Appwrite-compatible structure
- **Link files**: Use source attribution and document context
- **Handle 50-70 documents**: Prepare for deduplication across strategy papers
- **Synergies detection**: Identify when same projects appear in multiple documents

## Next Steps for Implementation

1. **External Models**: Research OpenAI GPT-4o/o1 vs Gemini 2.5 for better explanations
2. **Schema Enhancement**: Ensure our Pydantic schemas capture all Appwrite fields
3. **Relationship Extraction**: Improve many-to-many relationship detection
4. **Deduplication Logic**: Handle when same entities appear across multiple strategy documents
5. **API Integration**: Prepare for external model connections while maintaining local fallback

This understanding ensures our extraction captures the operational complexity and relationship richness of real municipal sustainability management systems.