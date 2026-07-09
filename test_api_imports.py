import sys
sys.path.insert(0, ".")

tests = [
    ("mlflow_logger",    "from src.tracking.mlflow_logger import log_ingestion_run"),
    ("pdf_parser",       "from src.ingestion.pdf_parser import extract_text_from_pdf"),
    ("section_parser",   "from src.ingestion.section_parser import extract_sections"),
    ("chunker",          "from src.ingestion.chunker import chunk_documents"),
    ("embedder",         "from src.ingestion.embedder import build_chroma_index, load_chroma_index"),
    ("retriever",        "from src.retrieval.retriever import retrieve_candidates"),
    ("reranker",         "from src.retrieval.reranker import rerank"),
    ("query_transform",  "from src.retrieval.query_transform import generate_multi_queries"),
    ("generator",        "from src.generation.generator import generate_answer"),
    ("ingest router",    "from api.routes.ingest import router"),
    ("query router",     "from api.routes.query import router"),
    ("main app",         "from api.main import app"),
]

for name, stmt in tests:
    try:
        exec(stmt)
        print(f"✅ {name}")
    except Exception as e:
        print(f"❌ {name}: {e}")