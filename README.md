#  NERO.AI: Advanced RAG Document Q&A

Upload a PDF, ask questions in plain English, and get answers grounded in the document, with the source passages shown underneath.

**🔗 Live demo:** https://nero-ai-advanced-rag-ndcdxw66bshkdagdd2yn7a.streamlit.app/

> Try it with any PDF. Click **PROCESS**, then ask something about the document. (If the app has been idle, it may take a moment to wake up.)

---

## What it does

NERO.AI is an end-to-end retrieval-augmented generation (RAG) pipeline. Instead of a basic "embed and search" setup, it uses several techniques that improve answer quality on long documents:

- **Parent-child chunking:** small chunks are searched (precise matching), but the LLM is given the larger parent chunk (full context).
- **Hybrid search:** keyword search (BM25) combined with semantic vector search, so exact terms and meaning both count.
- **Reranking:** Cohere's reranker re-scores the candidates and keeps the best five.
- **Grounded answers:** the LLM only answers from the retrieved context, and says so when the answer isn't in the document.
- **Source display:** every answer comes with an expandable list of the passages it was based on.

## Architecture

```
PDF → parent chunks (2000 chars) → child chunks (500 chars)
          │                              │
          ▼                              ▼
   Qdrant: parent collection      Qdrant: child collection
                                         │
question → hybrid search on children (BM25 30% + semantic 70%)
                                         │
                              Cohere rerank → top 5 children
                                         │
                      fetch matching parents via parent_id
                                         │
                          Groq LLM → grounded answer + sources
```

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | Qdrant Cloud |
| Keyword search | BM25 (`rank_bm25`) |
| Reranker | Cohere `rerank-v3.5` |
| LLM | Groq `openai/gpt-oss-120b` |
| Orchestration | LangChain |

## Project structure

```
├── app.py           # Streamlit UI (upload, process, ask, sources)
├── ingest.py        # PDF loading, parent/child splitting, Qdrant storage
├── retrieval.py     # Hybrid search, rerank, parent fetch, answer generation
├── config.py        # Environment variables and collection names
└── requirements.txt
```

## Run it locally

```bash
git clone https://github.com/AbdulHai564/nero-ai-advanced-rag.git
cd nero-ai-advanced-rag

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key
COHERE_API_KEY=your_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_key
CHILD_COLLECTION_NAME=child_collection
PARENT_COLLECTION_NAME=parent_collection
```

Then start the app:

```bash
streamlit run app.py
```

## Known limitations

- **One document at a time.** Each PROCESS replaces the previous document in the shared Qdrant collections, so simultaneous users would overwrite each other. A per-user collection would fix this.
- **Re-process after a restart.** The BM25 index lives in memory, so after the app restarts or sleeps you need to click PROCESS again.
- **Text-based PDFs only.** Scanned PDFs need OCR first.

## Roadmap

- Per-session collections for multi-user use
- Offline RAGAS evaluation (faithfulness, answer relevancy, context precision, context recall) with scores published here
- OCR support for scanned documents

## Author

Built by **Abdulhai**. [GitHub](https://github.com/AbdulHai564)
