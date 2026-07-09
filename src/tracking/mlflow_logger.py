import mlflow


def log_ingestion_run(
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    chunking_strategy: str,
    num_chunks: int,
    num_documents: int
):
    with mlflow.start_run(run_name="ingestion"):
        mlflow.log_params({
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
            "chunking_strategy": chunking_strategy,
            "num_source_documents": num_documents,
            "vector_db": "chromadb"
        })
        mlflow.log_metrics({
            "total_chunks": num_chunks,
            "avg_chunks_per_doc": num_chunks / max(num_documents, 1)
        })
        print(f"MLflow: Logged ingestion run ({num_chunks} chunks)")


def log_query_run(
    query: str,
    retrieval_method: str,
    use_reranker: bool,
    num_candidates: int,
    num_final_docs: int,
    ragas_scores: dict = None
):
    with mlflow.start_run(run_name="query_eval"):
        mlflow.log_params({
            "retrieval_method": retrieval_method,
            "use_reranker": use_reranker,
            "num_candidates": num_candidates,
            "num_final_docs": num_final_docs,
            "vector_db": "chromadb"
        })
        mlflow.log_text(query, "query.txt")
        if ragas_scores:
            mlflow.log_metrics(ragas_scores)