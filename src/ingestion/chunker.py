from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def hierarchical_chunk(
    sections: dict[str, str],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    source_name: str = "unknown"
) -> list[Document]:
    """
    Section-aware chunking.
    Each chunk keeps metadata: section name + source filename.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    all_docs = []
    for section_name, text in sections.items():
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "section": section_name,
                    "source": source_name,
                    "chunk_index": i,
                    "total_chunks_in_section": len(chunks)
                }
            )
            all_docs.append(doc)
    print(f"Total chunks created: {len(all_docs)}")
    return all_docs

def naive_chunk(
    sections: dict[str, str],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    source_name: str = "unknown"
) -> list[Document]:
    """Simple fixed-size chunking — no section awareness."""
    full_text = " ".join(sections.values())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(full_text)
    return [
        Document(
            page_content=chunk,
            metadata={"source": source_name, "chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]

def chunk_documents(
    sections: dict[str, str],
    strategy: str = "hierarchical",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    source_name: str = "unknown"
) -> list[Document]:
    if strategy == "naive":
        return naive_chunk(sections, chunk_size, chunk_overlap, source_name)
    return hierarchical_chunk(sections, chunk_size, chunk_overlap, source_name)