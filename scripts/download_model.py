# scripts/download_model.py
# Run this once — downloads model to local cache
import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L12-v2"
SAVE_PATH = "models/paraphrase-MiniLM-L12-v2"

os.makedirs(SAVE_PATH, exist_ok=True)
print(f"Downloading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
model.save(SAVE_PATH)
print(f"Model saved to: {SAVE_PATH}")

# Quick test
result = model.encode(["What are the risk factors?"])
print(f"Embedding dim: {len(result[0])} — OK")