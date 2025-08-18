1. JSON Output -> Do you have any examples?

2. Can we use external Models?

3. How is the perfect JSON Structure?

4. Do we have enough data to train our own models?
- PDFs & JSON Outputs

5. Workflow -> Upload -> PDF to Text -> LLM Chunk for ContextWindow -> Analzye and to JSON -> Improve JSON by Classifiying into 4 Buckets


Original 4-Field Structure

  1. Action Fields (action_fields)
  - action_field: Name of the action field/domain (string)
  - projects: Array of Project objects

  2. Projects (nested within action fields)
  - title: Project title/name (string)
  - measures: Array of measure strings (optional)
  - indicators: Array of indicator strings (optional)
  - sources: Array of SourceAttribution objects (optional)

  3. Measures (as string arrays within projects)
  - Simple list of measure titles/descriptions

  4. Indicators (as string arrays within projects)
  - Simple list of indicator names/metrics

  Source Attribution (when available):
  - page_number: Page in original PDF (1-based)
  - quote: Relevant text excerpt
  - chunk_id: Internal chunk ID for debugging





---

Further Features:
- Confidence
- Quote
- Page Number


Step 1: PDF Upload and Text Extraction

   1. You upload a PDF file via the /upload API endpoint.
   2. The system saves the file with a unique ID and then extracts all the text from it, using OCR as a fallback for any images.
   3. The extracted text is saved, page by page, into a text file. This unique ID is used to track the file through the rest of the process.

  Step 2: Initial Data Extraction

   1. Using the ID from the upload, you call the /extract_structure endpoint.
   2. The system loads the extracted text and splits it into smaller chunks.
   3. Each chunk is sent to a Large Language Model (LLM) to identify and extract key information, which is returned as a structured, but still fairly raw, JSON object.
   4. This initial JSON output is saved as an "intermediate" file.

  Step 3: Final JSON Enhancement and Structuring

   1. You then call the final /enhance_structure endpoint, again using the file's unique ID.
   2. The system loads the intermediate JSON file.
   3. It uses another LLM to transform this raw data into a more refined and organized "four-bucket" relational structure. This step also involves resolving any inconsistencies and adding
      confidence scores to the data.
   4. The final, enhanced JSON file is saved to the data/outputs folder, completing the process.



----


Hanglungsdelfer

Projects

Alaways with explanation

headlines & explanation

---

Fill the existing schema and entities

A way to link the files

typescript that descripes the collections? and fields and types??


upload 50 to 70 documents

duplication and synergies between strategy papers
