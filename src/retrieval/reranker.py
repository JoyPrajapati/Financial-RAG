from langchain_core.documents import Document

def rerank(
    query: str,
    candidates: list[Document],
    top_k: int = 5
) -> list[Document]:
    """
    Reranker stub — returns top_k candidates as-is.
    Cross-encoder disabled to avoid PyTorch DLL issues on Windows.
    Enable on Linux/cloud deployment when needed.
    """
    return candidates[:top_k]