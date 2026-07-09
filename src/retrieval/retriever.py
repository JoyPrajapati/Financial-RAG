from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

def retrieve_candidates(
    query: str,
    vectorstore: Chroma,
    top_k: int = 20,
    source_filter: str = None
) -> list[Document]:
    if source_filter:
        return vectorstore.similarity_search(
            query,
            k=top_k,
            filter={"source": source_filter}
        )
    return vectorstore.similarity_search(query, k=top_k)