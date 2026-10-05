# Enterprise Vision-RAG Invoice Extractor

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-FF4B4B.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-4169E1.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

An enterprise-grade, multi-modal AI pipeline designed to automate Accounts Payable (AP) workflows. This system extracts structured JSON data from messy invoice PDFs/Images, validates financial math, and performs 2-way semantic Purchase Order (PO) matching using a local Vector Database to prevent fraud.

## Key Enterprise Features

* **Multi-Modal Vision Extraction:** Uses LiteLLM & Instructor to enforce strict Pydantic schemas, routing heavy PDF/JPG/PNG payloads to Vision LLMs (e.g., Gemini, Claude, OpenAI) for zero-shot data extraction.
* **Semantic RAG PO Matching:** Replaces fragile regex matching with a local **ChromaDB** vector database and HuggingFace `sentence-transformers`. Automatically flags unauthorized/fraudulent vendors.
* **Agentic Self-Healing:** Implements a retry loop that feeds Pydantic validation errors back to the LLM, correcting math hallucinations (e.g., Subtotal + Tax != Total) before presenting data to the user.
* **FinOps & Observability:** Tracks API latency, token usage, cost (USD), and success rates in real-time via a local SQLite tracker.
* **Payload Optimization:** Dynamically downscales massive image files using Lanczos resampling and lossy JPEG compression, reducing API latency by over 60%.
* **Batch Processing & HITL UI:** Features an asynchronous Batch Upload engine for ZIP files and a Persona-Based Streamlit UI enabling Human-in-the-Loop (HITL) manual review and database commits.

---

## Architecture

1. **Frontend:** Persona-based Streamlit Dashboard (Accountant View vs. Developer Console).
2. **Backend:** Asynchronous FastAPI microservice.
3. **AI Router:** LiteLLM for multi-provider fallback + Instructor for structured JSON decoding.
4. **Vector Store:** ChromaDB (Persistent local storage).
5. **Deployment:** Fully containerized via Docker & Docker Compose.

---

## Quickstart (Docker)

The easiest way to run this application is via Docker. 

**1. Clone the repository and add your API key:**
```bash
git clone https://github.com/YOUR_USERNAME/invoice-extractor.git
cd invoice-extractor
```

Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_api_key_here
```

**2. Spin up the cluster:**
```bash
docker compose up --build
```

**3. Access the Application:**
* **Streamlit UI:** `http://localhost:8501`
* **FastAPI Swagger Docs:** `http://localhost:8000/docs`

---

## Manual Local Setup

If you prefer to run it without Docker:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies (requires system poppler-utils for pdf2image)
pip install -r requirements.txt

# Start the Backend (Terminal 1)
uvicorn app.main:app --reload --port 8000

# Start the Frontend (Terminal 2)
streamlit run frontend/streamlit_app.py
```

## License
MIT License. Feel free to use this for your portfolio or adapt it for your organization's internal workflows.
