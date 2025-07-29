# Table Extraction Feature

## Overview

The CURATE PDF extraction pipeline now includes native table detection and extraction using PyMuPDF v1.23.0+. This feature automatically identifies tables in PDF documents and converts them to markdown format for better processing by the LLM.

## How It Works

1. **Detection**: For each page, the system uses PyMuPDF's `find_tables()` method to detect all tables
2. **Conversion**: Each detected table is converted to markdown format using `to_markdown()`
3. **Integration**: Tables are appended to the page text with HTML comments for context
4. **Preservation**: Table structure is maintained through the chunking process

## Benefits

- **Structure Preservation**: Tables maintain their row/column structure in markdown format
- **Better Extraction**: Indicators and measures in tables are now properly captured
- **LLM-Friendly**: Markdown format is ideal for LLM processing
- **Page Attribution**: Each table is tagged with its source page number

## Metadata

The extraction process now includes table statistics in the metadata:

```json
{
  "table_extraction": {
    "total_tables": 15,
    "pages_with_tables": [3, 7, 12, 15, 23],
    "table_errors": 0
  }
}
```

## Example Output

Tables are embedded in the text with clear markers:

```markdown
This is the regular text content...

<!-- Table 1 on page 7 -->
| Maßnahme | Indikator | Zielwert 2030 |
|----------|-----------|---------------|
| Ausbau Radwegenetz | Kilometer Radweg | 500 km |
| E-Bus Flotte | Anzahl E-Busse | 100 |
<!-- End of table -->

More regular text continues...
```

## Testing

To test table extraction on a PDF:

```bash
python tests/test_table_extraction.py path/to/your.pdf
```

## Error Handling

- If table detection fails on a page, regular text extraction continues
- Individual table conversion errors are logged but don't stop the process
- The system gracefully handles PDFs without tables