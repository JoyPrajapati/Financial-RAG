import sys
sys.path.insert(0, ".")

pdf_path = "data/uploads/a10kfy2023filing.pdf"

print("Step 1: Testing PyMuPDF...")
try:
    import fitz
    doc = fitz.open(pdf_path)
    print(f"Pages: {len(doc)}")
    text = doc[0].get_text("text")
    print(f"First page chars: {len(text)}")
    doc.close()
    print("PyMuPDF OK")
except Exception as e:
    print(f"PyMuPDF FAILED: {e}")
    sys.exit(1)

print("\nStep 2: Testing section parser...")
try:
    from src.ingestion.pdf_parser import extract_text_from_pdf
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted: {len(raw_text)} chars")
except Exception as e:
    print(f"pdf_parser FAILED: {e}")
    sys.exit(1)

print("\nStep 3: Testing section extraction...")
try:
    from src.ingestion.section_parser import extract_sections
    sections = extract_sections(raw_text)
    print(f"Sections found: {list(sections.keys())}")
except Exception as e:
    print(f"section_parser FAILED: {e}")
    sys.exit(1)

print("\nStep 4: Testing chunker...")
try:
    from src.ingestion.chunker import chunk_documents
    docs = chunk_documents(sections, strategy="hierarchical",
                           source_name="test.pdf")
    print(f"Chunks: {len(docs)}")
except Exception as e:
    print(f"chunker FAILED: {e}")
    sys.exit(1)

print("\nStep 5: Testing embedder (first 3 chunks only)...")
try:
    from src.ingestion.embedder import build_chroma_index
    build_chroma_index(docs[:3], save_path="vector_store/test_chroma")
    print("Embedder OK")
except Exception as e:
    print(f"embedder FAILED: {e}")
    sys.exit(1)

print("\nAll steps passed!")