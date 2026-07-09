# test_embeddings.py — replace entire file
import sys
sys.path.insert(0, ".")
from src.ingestion.embedder import get_embedding_model

emb = get_embedding_model()
result = emb.embed_query("What are the main risk factors?")
print(f"Embedding dim: {len(result)} — OK")
print(f"First 5 values: {result[:5]}")