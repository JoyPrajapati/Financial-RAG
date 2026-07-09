# test_pdf_quality.py
import sys
sys.path.insert(0, ".")
import fitz

pdf_path = "data/uploads/a10kfy2023filing.pdf"  # change filename here

with fitz.open(pdf_path) as doc:
    sample = doc[0].get_text("text")
    clean_chars = sum(1 for c in sample if 32 <= ord(c) <= 126)
    total_chars = len(sample)
    ratio = clean_chars / max(total_chars, 1)
    print(f"Clean text ratio: {ratio:.2%}")
    print(f"Sample (first 500 chars):\n{sample[:500]}")
    if ratio > 0.85:
        print("\n✅ PDF is text-native — good to ingest")
    else:
        print("\n❌ PDF is scanned/encoded — download a different version")