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

| Component      | Library / Tool            |
|----------------|---------------------------|
| API Server     | `FastAPI` + `Uvicorn`     |
| PDF Parsing    | `PyMuPDF` (`fitz`)        |
| OCR            | `pytesseract` + `pdf2image` |
| LLM Access     | [`Ollama`](https://ollama.com) running `llama3` |
| File Uploads   | `python-multipart`        |
| HTTP Requests  | `requests`                |
| Embeddings     | `sentence-transformers`   |
| Store embed.   | `chromadb`                |

---

## ⚙️ Environment set up

✅ Step 1: Check Your Python Installation
Open a terminal (or Command Prompt on Windows), and type:
```
python3 --version
```
You should see python version, otherwise install python:
```
🪟 Windows: Download Python

🍎 Mac: Already included, or install via Homebrew: brew install python

🐧 Linux: sudo apt install python3 python3-pip
```

✅ Step 2: Set Up a Virtual Environment
Navigate to your project folder, create virtual environment:
```
python3 -m venv venv
```
Activate it:
- On Mac/Linux: source venv/bin/activate
- On Windows: venv\Scripts\activate

✅ Step 3: Install Required Libraries
```
pip install -r requirements.txt
```
Or, you can also install libraries manually:
```
pip install fastapi uvicorn pymupdf python-multipart pytesseract pdf2image requests... (check full list in requirements.txt)
```
**Additional System Packages (macOS example)**  
Ollama -  a tool to easily run local LLMs like Mixtral
```
Go to https://ollama.com and download the installer for your OS.
```
llama3:8b - lighter LLM model, supports up to 8,192 tokens
```
ollama pull llama3:8b
```
tesseract - OCR engine for scanned pages
poppler - Needed by pdf2image
```
brew install tesseract poppler
```
On Windows:
- Install [Tesseract OCR for Windows](https://github.com/tesseract-ocr/tesseract)
- Install [Poppler for Windows](https://blog.alivate.com.au/poppler-windows/) - 
Add both to your PATH environment variable

✅ Step 4: Project Structure
- main.py            ← FastAPI server
- parser.py          ← PDF text extractor
- uploads/           ← Folder for uploaded PDFs

✅ Step 7: Run the Servers
Manually run Ollama before starting FastAPI:
(If you downloaded Ollama as a macOS app, just open the app, no need for the code)
```
ollama serve
```
Keep this running — this is your local LLM server. Confirm it works:
```
ollama run llama3:8b "Say hello"
```
Then in another terminal, you start FastAPI:
```
uvicorn main:app --reload
```
Url for local host: http://127.0.0.1:8000/docs