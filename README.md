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
| LLM Access     | [`Ollama`](https://ollama.com) or [`vLLM`](https://github.com/vllm-project/vllm) |
| File Uploads   | `python-multipart`        |
| HTTP Requests  | `requests`                |
| Embeddings     | `sentence-transformers`   |
| Vector Store   | `chromadb`                |
| Schemas        | `pydantic`                |
| Code Quality   | `ruff`, `black`, `mypy`   |

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
pip install -r requirements-dev.txt  # For development tools
```
Or, you can also install libraries manually:
```
pip install fastapi uvicorn pymupdf python-multipart pytesseract pdf2image requests... (check full list in requirements.txt)
```

**German Language Model for Structure-Aware Chunking**
After installing the requirements, download the German spaCy model:
```
python -m spacy download de_core_news_lg
```
Note: This model is ~568MB and improves document structure detection for German texts.

**Additional System Packages (macOS example)**
Ollama -  a tool to easily run local LLMs like Mixtral
```
Go to https://ollama.com and download the installer for your OS.
```
qwen3:14b - Optimized for structured output and multilingual tasks
```
ollama pull qwen3:14b
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
```
curate/
├── src/                      # Source code
│   ├── api/                  # API routes and endpoints
│   ├── core/                 # Core functionality (config, schemas, LLM)
│   ├── extraction/           # Document extraction logic
│   ├── processing/           # Document processing (parsing, chunking, embedding)
│   └── utils/                # Utilities (monitoring, logging)
├── data/                     # Data storage
│   ├── uploads/              # Uploaded PDFs
│   ├── chroma_store/         # Vector database
│   └── outputs/              # Extraction results
├── tests/                    # Test suite
├── scripts/                  # Utility scripts
├── docs/                     # Documentation
├── main.py                   # FastAPI server entry point
└── requirements.txt          # Python dependencies
```

✅ Step 5: Code Quality (Development)
Format and lint your code:
```bash
black .              # Format code
ruff check --fix .   # Fix linting issues
mypy .              # Type checking
```

✅ Step 6: Run the Servers

### Option A: Using Ollama (Default)
Manually run Ollama before starting FastAPI:
(If you downloaded Ollama as a macOS app, just open the app, no need for the code)
```
ollama serve
```
Keep this running — this is your local LLM server. Confirm it works:
```
ollama run qwen3:14b "Say hello"
```

### Option B: Using vLLM (High Performance)
CURATE now supports vLLM as an alternative to Ollama for better performance. To use vLLM:

1. Set up vLLM server (on a machine with GPU):
```bash
# Install vLLM
pip install vllm

# Run vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model RedHatAI/Qwen3-14B-quantized.w4a16 \
    --host 0.0.0.0 \
    --port 8001
```

2. Configure CURATE to use vLLM:
```bash
export LLM_BACKEND=vllm
export VLLM_HOST=your-vllm-server:8001  # e.g., 10.67.142.34:8001
```

### Start FastAPI Server
Then in another terminal, start FastAPI:
```
# For Ollama (default)
uvicorn main:app --reload

# For vLLM
LLM_BACKEND=vllm VLLM_HOST=your-server:8001 uvicorn main:app --reload
```
Url for local host: http://127.0.0.1:8000/docs

**Benefits of vLLM:**
- **Higher throughput**: PagedAttention reduces GPU memory waste (<4% vs 60-80%)
- **Continuous batching**: Better GPU utilization for multiple requests
- **OpenAI-compatible API**: Easy integration
- **Production-ready**: Built for high-performance serving

**Model Name Mappings:**
When using vLLM, Ollama model names are automatically mapped:
- `qwen3:14b` → `RedHatAI/Qwen3-14B-quantized.w4a16`
- `qwen3:7b` → `Qwen/Qwen3-7B-Instruct`
- `qwen3:8b` → `Qwen/Qwen3-8B-Instruct`
