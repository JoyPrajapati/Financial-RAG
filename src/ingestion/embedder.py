import os
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

# Path to locally saved model — works offline, no HF API needed
LOCAL_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models", "paraphrase-MiniLM-L12-v2"
)
CHROMA_INDEX_PATH = "vector_store/chroma_index"

# Load model once at module level — avoids reloading on every request
_model = None

def get_sentence_transformer() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading local embedding model from: {LOCAL_MODEL_PATH}")
        _model = SentenceTransformer(LOCAL_MODEL_PATH)
        print("Embedding model loaded.")
    return _model


class LocalEmbeddings(Embeddings):
    """
    Local sentence-transformers embeddings.
    No HF API calls — runs fully offline via CPU.
    Model: paraphrase-MiniLM-L12-v2 (384-dim)
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = get_sentence_transformer()
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        model = get_sentence_transformer()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


def get_embedding_model() -> LocalEmbeddings:
    return LocalEmbeddings()


def build_chroma_index(
    documents: list[Document],
    save_path: str = CHROMA_INDEX_PATH
) -> Chroma:
    os.makedirs(save_path, exist_ok=True)
    embeddings = get_embedding_model()
    print(f"Embedding {len(documents)} chunks locally (CPU)...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=save_path
    )
    print(f"ChromaDB index saved to: {save_path}")
    return vectorstore


def load_chroma_index(load_path: str = CHROMA_INDEX_PATH) -> Chroma:
    embeddings = get_embedding_model()
    vectorstore = Chroma(
        persist_directory=load_path,
        embedding_function=embeddings
    )
    print(f"ChromaDB index loaded from: {load_path}")
    return vectorstore