# GitContext 🔍

**Ask questions about any GitHub repository and get AI-generated answers grounded in its source code.**

GitContext is an AI-powered repository understanding tool designed to help developers explore unfamiliar codebases. Instead of manually searching through files, users can ask natural-language questions and receive answers supported by relevant source-code references.

## ✨ Features

- **GitHub Repository Ingestion** — Load and process source files from a GitHub repository.
- **Semantic Search** — Find code relevant to a question using embeddings.
- **Hybrid Retrieval** — Combine semantic search with BM25 keyword-based retrieval.
- **Cross-Encoder Reranking** — Improve the relevance of retrieved code chunks.
- **AI-Powered Answers** — Generate answers using Google's Gemini API.
- **Source References** — Display relevant source chunks alongside generated answers.
- **Markdown Rendering** — Present answers with readable headings, lists, tables, and code blocks.
- **REST API** — Expose repository question-answering functionality through FastAPI.
- **Dark UI** — Explore repositories through a React-based interface.

## 🛠️ Tech Stack

| Component | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI |
| AI Model | Google Gemini API |
| Semantic Retrieval | Sentence Transformers |
| Keyword Retrieval | BM25 |
| Reranking | Cross-Encoder |
| Vector Search | Qdrant |
| Environment Configuration | python-dotenv |

## 🏗️ Architecture

1. **Repository ingestion:** Collect supported source files from a GitHub repository.
2. **Chunking:** Split source files into smaller code segments for retrieval.
3. **Embedding and indexing:** Create vector representations and store them in the vector database.
4. **Hybrid retrieval:** Retrieve relevant chunks using semantic and keyword-based search.
5. **Reranking:** Rank candidate chunks by relevance to the user's question.
6. **Answer generation:** Send the question and relevant code context to Gemini.
7. **Response display:** Show the generated answer and relevant source references in the frontend.

## 🚀 Getting Started

### Prerequisites

- Python installed
- Node.js and npm installed
- A Google Gemini API key
- Git installed

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd GitContext
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Set up the Python environment

From the project root, create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the Python dependencies using the project's dependency file, if available:

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_supported_gemini_model
```

Replace the placeholders with your own credentials and the model configured for your project.

**Security:** Never commit your `.env` file or expose your API key in source code.

### 4. Start the backend

From the project root, run:

```powershell
python -m uvicorn src.main:app --reload
```

The API should be available at:

- API: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

### 5. Start the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`.

## 💬 Example Questions

After entering a supported GitHub repository, try questions such as:

- How is authentication implemented?
- Where is the database connection configured?
- Explain the main application flow.
- Which files handle error handling?
- How does a particular function work?

The quality of answers depends on the repository content successfully retrieved and the available AI model context.

## ⚠️ Limitations

- Answer generation requires a valid Gemini API key and available API quota.
- API rate limits may temporarily prevent answer generation.
- Answer quality depends on retrieval quality and the source code available to the system.
- Large repositories may require additional ingestion and performance optimizations.
- Generated answers should be verified against the original source code.

## 🔮 Future Improvements

- Add repository indexing and cache persistence.
- Support larger repositories and additional programming languages.
- Add automated tests for ingestion, retrieval, and API endpoints.
- Improve error handling, observability, and request latency.
- Add deployment support and continuous integration.
- Improve source navigation with direct file and line references.

## 👩‍💻 Project Goal

GitContext explores how retrieval-augmented generation (RAG), hybrid search, and reranking can help developers understand unfamiliar codebases more efficiently.

---

*Built as an AI-powered developer productivity project.*
