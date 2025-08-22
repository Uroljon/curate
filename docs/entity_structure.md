# Entity Structure

## Core Structure

Yes, the 4 main entities are:
- **Dimensions** (Action Fields) - Strategic areas
- **Measures** (Projects/Measures) - Implementation initiatives
- **Indicators** - Quantitative metrics
- **Measure2indicator** - Junction table for connections

## Key Clarifications

1. **Projects vs Measures**:
   - In Appwrite, there's only `Measures` entity
   - Measures can be hierarchical via `parentMeasure` field
   - A "Project" is just a parent Measure with `isParent: true`

2. **The Actual Hierarchy**:
```
Dimensions (Action Fields)
    ↓ (via dimensionIds in MeasuresExtended)
Measures (Projects/Measures combined)
    ↓ (via parentMeasure)
Child Measures
    ↓ (via Measure2indicator junction)
Indicators
```

3. **Connection Patterns**:
- ✅ **Dimensions → Dimensions** (via `parentDimensionId`)
- ✅ **Dimensions → Measures** (via `dimensionIds` in `MeasuresExtended`)
- ✅ **Indicators → Dimensions** (via `dimensionId`/`dimensionIds`)
- ✅ **Measures → Indicators** (via `Measure2indicator` junction table)
- ✅ **Measures → Measures** (via `parentMeasure` array)

4. **MeasuresExtended Role**:
- Supplements `Measures` with relationship data
- Contains `dimensionIds` array that connects Measures to Dimensions
- Contains `indicatorIds` array for indicator relationships

## Summary

- **Dimensions** (Action Fields) connect to Measures
- **Measures** group into Projects (parent-child hierarchy)
- **Indicators** can connect to Dimensions, Measures, and Projects
- Everything has potential hierarchical relationships