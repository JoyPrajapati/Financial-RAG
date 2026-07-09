import os
import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File
from api.schemas import IngestResponse
from src.ingestion.file_parser import extract_text, SUPPORTED_EXTENSIONS
from src.ingestion.section_parser import extract_sections
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import build_chroma_index
from src.tracking.mlflow_logger import log_ingestion_run

router = APIRouter()
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Upload any supported file type.
    Supported: PDF, DOCX, XLSX, TXT
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    try:
        # Read and save file
        file_bytes = await file.read()
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(file_bytes)
        print(f"File saved: {save_path}")

        # Extract text using correct parser
        raw_text = extract_text(save_path)

        # Parse sections
        print("Extracting sections...")
        sections = extract_sections(raw_text)

        # Chunk
        docs = chunk_documents(
            sections,
            strategy="hierarchical",
            chunk_size=512,
            chunk_overlap=50,
            source_name=file.filename
        )

        # Build index
        build_chroma_index(docs)

        # Log
        log_ingestion_run(
            chunk_size=512,
            chunk_overlap=50,
            embedding_model="paraphrase-MiniLM-L12-v2-local",
            chunking_strategy="hierarchical",
            num_chunks=len(docs),
            num_documents=1
        )

        return IngestResponse(
            status="success",
            filename=file.filename,
            file_type=ext,
            total_chunks=len(docs)
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))