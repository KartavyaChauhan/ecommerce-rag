# E-Commerce RAG Engine

An intelligent **Retrieval-Augmented Generation (RAG)** system for e-commerce product information retrieval. Built with **FastAPI** and **Google Gemini**, this application allows users to upload product catalogs (PDF format), then ask natural-language questions — the system retrieves the most relevant context and generates accurate, grounded answers.

---

## Approach

This project implements a **RAG (Retrieval-Augmented Generation)** pipeline that combines semantic search with large language model generation:

1. **Document Ingestion**: PDF documents are uploaded, validated, and extracted using PyPDF
2. **Text Chunking**: Documents are split into overlapping chunks using LangChain's RecursiveCharacterTextSplitter
3. **Embedding & Storage**: Chunks are embedded using Google's Gemini embedding model and stored in ChromaDB
4. **Semantic Retrieval**: User queries are embedded and matched against stored chunks using similarity search
5. **Answer Generation**: Retrieved context is passed to Google Gemini LLM to generate grounded, contextual answers

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Google Gemini** (embedding + LLM) | Free-tier API with generous limits, modern models, single provider simplifies authentication and configuration |
| **LLM Fallback Chain** | `gemini-2.0-flash → gemini-2.0-flash-lite → gemini-2.5-flash` — automatic failover on rate-limit (429) or quota errors ensures high availability |
| **ChromaDB** (persistent storage) | Lightweight, file-based vector store requiring zero infrastructure — data persists across restarts without external databases |
| **RecursiveCharacterTextSplitter** | 1,000-character chunks with 200-character overlap balances context richness vs. embedding quality, prevents information loss at boundaries |
| **Source Deduplication** | Duplicate chunks are filtered before LLM invocation for cleaner, non-repetitive responses |
| **Pydantic v2 Settings** | Type-safe configuration loaded from `.env` with validation at startup, catches configuration errors early |
| **Service Layer Architecture** | Separation of concerns (document, vector, RAG services) enables testability, maintainability, and future extensibility |

---

## Tech Stack & Dependencies

### Core Framework
- **Python 3.11+** — Modern Python with improved performance
- **FastAPI** — High-performance async REST API framework with automatic OpenAPI docs
- **Uvicorn** — ASGI server for running the application

### AI/ML Components
- **LangChain** — Orchestration framework for prompts, chains, and output parsing
- **Google Gemini** — `gemini-embedding-001` (embeddings) + `gemini-2.0-flash` (text generation)
- **ChromaDB** — Persistent vector database for similarity search

### Document Processing
- **PyPDF** — PDF text extraction
- **LangChain Text Splitters** — Intelligent document chunking

### Data Validation
- **Pydantic v2** — Data validation and settings management
- **python-dotenv** — Environment variable loading

---

## Project Architecture

```
app/
├── main.py                  # FastAPI application factory, CORS, routing
├── core/
│   └── config.py            # Centralised settings (env vars, model config)
├── api/v1/
│   └── endpoints.py         # REST endpoints (/upload, /search, /chat)
├── models/
│   └── schemas.py           # Pydantic request/response schemas
├── services/
│   ├── document_service.py  # PDF ingestion & text chunking
│   ├── vector_service.py    # ChromaDB embeddings & similarity search
│   └── rag_service.py       # RAG pipeline with LLM fallback
└── utils/
    └── logger.py            # Logging configuration
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd ecommerce-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

> Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey).

### 5. Run the server

```bash
uvicorn app.main:app --reload
```

The API will be available at **http://127.0.0.1:8000**.  
Interactive docs at **http://127.0.0.1:8000/docs**.

---

## API Endpoints

### `GET /health`
Health check.

```json
{ "status": "healthy", "version": "1.0.0" }
```

### `POST /api/v1/upload`
Upload a PDF document for processing.

- **Content-Type:** `multipart/form-data`
- **Body:** `file` (PDF)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/upload \
  -F "file=@product_catalog.pdf"
```

### `POST /api/v1/search`
Raw semantic search — returns matching text chunks.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '"What wireless headphones do you have?"'
```

### `POST /api/v1/chat`
Ask a question — retrieves context and generates an LLM answer.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the best-selling products?", "k": 3}'
```

**Response:**
```json
{
  "answer": "Based on the catalog, the best-selling products are...",
  "sources": [
    { "source": "product_catalog.pdf", "content": "Our top sellers include..." }
  ]
}
```

---

## Project Structure

```
ecommerce-rag/
├── .env                 # API keys (not committed)
├── .gitignore
├── requirements.txt
├── README.md
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   └── v1/
│   │       └── endpoints.py
│   ├── core/
│   │   └── config.py
│   ├── models/
│   │   └── schemas.py
│   ├── services/
│   │   ├── document_service.py
│   │   ├── vector_service.py
│   │   └── rag_service.py
│   └── utils/
│       └── logger.py
└── data/
    └── chroma_db/       # Auto-created on first upload
```

---

## Dependencies (requirements.txt)

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.6.0
pydantic-settings>=2.1.0
python-multipart>=0.0.7
python-dotenv>=1.0.1
pypdf>=4.0.0
aiofiles>=23.2.1
langchain>=0.1.0
langchain-core>=0.1.0
langchain-google-genai>=0.0.9
langchain-community>=0.0.10
langchain-text-splitters>=0.0.1
langchain-chroma>=0.1.0
chromadb>=0.4.22
```

---

## Usage Example

```python
# 1. Upload a product catalog
curl -X POST http://127.0.0.1:8000/api/v1/upload -F "file=@products.pdf"

# 2. Ask a question about the products
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What laptops do you have under $1000?"}'

# Response:
# {
#   "answer": "Based on the catalog, we have several laptops under $1000...",
#   "sources": [{"source": "products.pdf", "content": "...relevant excerpt..."}]
# }
```

---

## License

This project is for educational purposes.
