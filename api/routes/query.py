from fastapi import APIRouter, HTTPException
from api.schemas import QueryRequest, QueryResponse, Source
from src.ingestion.embedder import load_chroma_index
from src.retrieval.retriever import retrieve_candidates
from src.retrieval.reranker import rerank
from src.retrieval.query_transform import generate_multi_queries
from src.generation.generator import generate_answer
from src.tracking.mlflow_logger import log_query_run

router = APIRouter()
_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_chroma_index()
    return _vectorstore

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        vectorstore = get_vectorstore()

        if request.use_multi_query:
            queries = generate_multi_queries(request.question, n=3)
        else:
            queries = [request.question]

        all_candidates = []
        seen_contents = set()
        for q in queries:
            candidates = retrieve_candidates(
                q,
                vectorstore,
                top_k=request.top_k_retrieve,
                source_filter=request.source_filter   # ← pass filter
            )
            for doc in candidates:
                if doc.page_content not in seen_contents:
                    all_candidates.append(doc)
                    seen_contents.add(doc.page_content)

        print(f"Total candidates retrieved: {len(all_candidates)}")

        if not all_candidates:
            raise HTTPException(
                status_code=404,
                detail="No relevant chunks found. Try ingesting a document first."
            )

        if request.use_reranker:
            final_docs = rerank(request.question, all_candidates, top_k=request.top_k_final)
        else:
            final_docs = all_candidates[:request.top_k_final]

        result = generate_answer(request.question, final_docs)

        log_query_run(
            query=request.question,
            retrieval_method="multi_query" if request.use_multi_query else "single_query",
            use_reranker=request.use_reranker,
            num_candidates=len(all_candidates),
            num_final_docs=len(final_docs)
        )

        return QueryResponse(
            answer=result["answer"],
            sources=[Source(**s) for s in result["sources"]],
            model_used=result["model_used"],
            num_sources=result["num_sources"]
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))