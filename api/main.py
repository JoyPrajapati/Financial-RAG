from fastapi import FastAPI
from api.routes.ingest import router as ingest_router
from fastapi.responses import JSONResponse
from api.routes.query import router as query_router

app = FastAPI(
    title="Financial RAG API v2",
    description=(
        "RAG system for SEC 10-K PDF filings — "
        "ChromaDB + paraphrase-MiniLM-L12-v2 + nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    ),
    version="2.0.0"
)

app.include_router(ingest_router, tags=["Ingestion"])
app.include_router(query_router, tags=["Query"])


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "financial-rag-v2",
        "vector_db": "chromadb",
        "embedding_model": "paraphrase-MiniLM-L12-v2",
        "llm": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    }
    
@app.get("/api/v1/indexed-files")
def get_indexed_files():
    """Return list of unique filenames currently indexed in ChromaDB."""
    try:
        from src.ingestion.embedder import load_chroma_index
        vs = load_chroma_index()
        data = vs.get(include=["metadatas"])
        files = sorted(set(
            m.get("source", "")
            for m in data["metadatas"]
            if m.get("source")
        ))
        return JSONResponse({"files": files, "count": len(files)})
    except Exception as e:
        return JSONResponse({"files": [], "error": str(e)})