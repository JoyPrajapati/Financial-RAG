import fitz
import os
import re

def clean_extracted_text(text: str) -> str:
    """Remove garbled characters and clean extracted PDF text."""
    # Remove non-printable characters except newlines and spaces
    text = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract full text from PDF using PyMuPDF.
    Tries multiple extraction methods to handle encoded PDFs.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    all_text = []

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        for page_num, page in enumerate(doc, start=1):
            # Try standard text extraction first
            text = page.get_text("text")

            # If garbled or empty, try blocks extraction
            if not text.strip() or len(text) < 50:
                blocks = page.get_text("blocks")
                text = "\n".join(
                    b[4] for b in blocks if isinstance(b[4], str)
                )

            # Clean the extracted text
            text = clean_extracted_text(text)

            if text.strip():
                all_text.append(f"--- Page {page_num} ---\n{text}")

    full_text = "\n".join(all_text)
    print(f"Extracted {len(full_text)} characters from {total_pages} pages")
    return full_text