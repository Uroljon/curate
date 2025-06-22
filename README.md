# 📄 PDF Strategy Extractor with LLM (FastAPI + Ollama)

This project extracts structured information (e.g. **action fields**, **projects**, **measures**, **indicators**) from long strategy PDF documents (often in German). It uses PDF parsing with OCR fallback and queries a **local LLM (e.g., LLaMA3 via Ollama)** for semantic understanding and structured extraction.

---

## 🚀 What It Does

- 📤 Upload PDF documents via FastAPI
- 🧾 Extract text using PyMuPDF + OCR fallback (`pytesseract`)
- 🧠 Query LLaMA3 (via Ollama) to summarize or extract structured content
- 💬 Future-ready for chat interface and vector-based retrieval

---

## 🛠️ Technologies Used

| Component      | Library / Tool           |
|----------------|---------------------------|
| API Server     | `FastAPI` + `Uvicorn`     |
| PDF Parsing    | `PyMuPDF` (`fitz`)        |
| OCR            | `pytesseract` + `pdf2image` |
| LLM Access     | [`Ollama`](https://ollama.com) running `llama3` |
| File Uploads   | `python-multipart`        |
| HTTP Requests  | `requests`                |

---

## 📦 Install Python Dependencies

```
pip install fastapi uvicorn pymupdf python-multipart pytesseract pdf2image requests

Additional System Packages (macOS example)
```
brew install tesseract poppler

## 🚦 Run the API Server
```
uvicorn main:app --reload

Url for local host: http://127.0.0.1:8000/docs#/default/upload_pdf_upload_post