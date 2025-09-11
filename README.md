# CURATE — Incremental Extraction and Diff Workflows

This project extracts structured entities and connections from PDFs and now supports incremental extraction across multiple PDFs, plus a diff utility to visualize what changed between runs.

Below is a concise guide to the new endpoints and the end‑to‑end workflow, including saving, filenames, and the bundled visualizer.

## Endpoints

- `POST /upload`
  - Upload a PDF and save a page‑aware text file for downstream extraction.
  - Returns a `source_id` (safe filename) you will use with other endpoints.

- `GET /extract_enhanced_operations?source_id=<PDF_ID>`
  - Runs the operations‑based extractor (two‑pass: nodes → connections) on a single PDF.
  - Saves: `data/uploads/<PDF_ID>_operations_result.json`

- `POST /extract_incremental?source_id=<NEW_PDF_ID>&base_source_id=<BASE_ID>`
  - New. Runs the same operations‑based pipeline but initializes from an existing JSON state.
  - Base can be any previously saved JSON (auto‑loaded in this order):
    1. `data/uploads/<BASE_ID>_combined_result.json`
    2. `data/uploads/<BASE_ID>_operations_result.json`
    3. `data/uploads/<BASE_ID>_enhanced_structure.json`
  - Alternative: provide the base directly in the request body: `{ "existing_json": <EnrichedReviewJSON> }`.
  - Saves: `data/uploads/<NEW_PDF_ID>_combined_result.json`

- `POST /compute_diff?json1_source_id=<BASE_ID>&json2_source_id=<NEW_ID>[&diff_filename=…][&save=false]`
  - New. Computes a structured diff between two saved JSONs (IDs may be single‑PDF or combined results).
  - Always saves by default to: `data/uploads/<NEW_ID>_diff_from_<BASE_ID>.json`
  - You can pass `diff_filename` to override the output name, or `save=false` to skip saving.
  - Returns the diff JSON in the HTTP response.

Notes:
- Always pass full `source_id` including the `.pdf` suffix (e.g., `20250911_141901_76384d7f_schlangen.pdf`).
- Swagger (http://127.0.0.1:8000/docs) shows typed request bodies for endpoints that accept JSON payloads.

## Typical Workflows

### A) Two PDFs (build PDF2 on PDF1, then compute diff)

1. Upload both PDFs; note `pdf1_id` and `pdf2_id`.
2. Extract PDF 1 to create the base JSON (either is fine):
   - `GET /extract_enhanced_operations?source_id=<pdf1_id>`
     - Saves `<pdf1_id>_operations_result.json`
3. Incremental extraction for PDF 2 using PDF 1 as base:
   - `POST /extract_incremental?source_id=<pdf2_id>&base_source_id=<pdf1_id>`
     - Saves `<pdf2_id>_combined_result.json` (contains PDF1 + PDF2)
4. Compute diff (what PDF 2 added/modified vs. PDF 1):
   - `POST /compute_diff?json1_source_id=<pdf1_id>&json2_source_id=<pdf2_id>`
     - Saves `<pdf2_id>_diff_from_<pdf1_id>.json`

Curl examples:

```bash
# Step 2
curl -s "http://127.0.0.1:8000/extract_enhanced_operations?source_id=<pdf1_id>"

# Step 3
curl -s -X POST "http://127.0.0.1:8000/extract_incremental?source_id=<pdf2_id>&base_source_id=<pdf1_id>"

# Step 4
curl -s -X POST "http://127.0.0.1:8000/compute_diff?json1_source_id=<pdf1_id>&json2_source_id=<pdf2_id>"
```

### B) Third PDF (build PDF3 on the combined result of PDF1+PDF2)

1. Upload PDF 3; note `pdf3_id`.
2. Incremental extraction using the combined JSON from PDF 2 as base:
   - `POST /extract_incremental?source_id=<pdf3_id>&base_source_id=<pdf2_id>`
     - Auto‑loads `<pdf2_id>_combined_result.json` as the base.
     - Saves `<pdf3_id>_combined_result.json` (PDF1+PDF2+PDF3)
3. Diff PDF3 vs. PDF1+PDF2:
   - `POST /compute_diff?json1_source_id=<pdf2_id>&json2_source_id=<pdf3_id>`
     - Saves `<pdf3_id>_diff_from_<pdf2_id>.json`

## Visualization

Open `visualization_tool.html` in a browser and:

- Load Combined JSON (the latest `<NEW_ID>_combined_result.json`).
- Load Diff JSON (the saved `<NEW_ID>_diff_from_<BASE_ID>.json`).
- Toggle “Diff View → Spotlight changes” to highlight:
  - Added nodes/edges (green), Modified nodes (blue), Unchanged dimmed.
- Legends:
  - Dashed green = added connection from diff
  - Dashed gray = derived AF hierarchy link (from `parent_action_field_name`)
  - Solid colors = existing connections (color reflects confidence)

## Configuration Tips

- Process more content per run:
  - `.env`: `FAST_EXTRACTION_MAX_CHUNKS=0` (0 = all chunks)
- Improve cross‑chunk edge recall (optional third pass):
  - `src/core/config.py`: `OPERATIONS_CONNECTION_SWEEP_ENABLED = True`
- The incremental run continues ID counters from the base (e.g., `af_5 → af_6`).

## Saved File Naming

- Pages: `data/uploads/<source_id>_pages.txt`
- Single‑PDF result (operations): `data/uploads/<source_id>_operations_result.json`
- Incremental combined result: `data/uploads/<new_source_id>_combined_result.json`
- Diff: `data/uploads/<new_source_id>_diff_from_<base_source_id>.json`

## Troubleshooting

- 400 “No saved JSON found …”: ensure `source_id` includes `.pdf` and has no stray characters (retype if copied from UI).
- Swagger shows a request body: leave it empty when using `base_source_id`/`json*_source_id` query params, or provide full valid JSON structures (no placeholders).

