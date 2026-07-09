from pydantic import BaseModel
from typing import Optional

class IngestResponse(BaseModel):
    status: str
    filename: str
    file_type: str        # ← new
    total_chunks: int

class QueryRequest(BaseModel):
    question: str
    use_multi_query: bool = False
    use_reranker: bool = False
    top_k_retrieve: int = 10
    top_k_final: int = 5
    source_filter: Optional[str] = None    # ← new

class Source(BaseModel):
    section: Optional[str] = None
    source: Optional[str] = None
    text_snippet: str

class QueryResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    answer: str
    sources: list[Source]
    model_used: str
    num_sources: int