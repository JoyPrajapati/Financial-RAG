import re

SECTION_PATTERNS = {
    "item_1":  r"(?i)item\s*1[\.\s\-–:]*\s*business",
    "item_1a": r"(?i)item\s*1a[\.\s\-–:]*\s*risk\s*factors",
    "item_7":  r"(?i)item\s*7[\.\s\-–:]*\s*management",
    "item_7a": r"(?i)item\s*7a[\.\s\-–:]*\s*quantitative",
}

# Broader fallback patterns if above don't match
FALLBACK_PATTERNS = {
    "risk_factors": r"(?i)risk\s+factors",
    "business":     r"(?i)our\s+business|description\s+of\s+business",
    "financials":   r"(?i)financial\s+statements|results\s+of\s+operations",
    "management":   r"(?i)management.{0,10}discussion",
}

def clean_text(text: str) -> str:
    text = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_sections(raw_text: str) -> dict[str, str]:
    text = clean_text(raw_text)

    # Try primary patterns first
    positions = {}
    for name, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            positions[name] = match.start()

    # If primary fails, try fallback patterns
    if not positions:
        print("Primary patterns not found. Trying fallback patterns...")
        for name, pattern in FALLBACK_PATTERNS.items():
            match = re.search(pattern, text)
            if match:
                positions[name] = match.start()

    # If still nothing, use chunked fallback
    if not positions:
        print("Warning: No sections detected. Splitting into logical chunks.")
        chunk_size = len(text) // 4
        return {
            "part_1": text[:chunk_size],
            "part_2": text[chunk_size:chunk_size*2],
            "part_3": text[chunk_size*2:chunk_size*3],
            "part_4": text[chunk_size*3:]
        }

    sorted_sections = sorted(positions.items(), key=lambda x: x[1])
    result = {}
    for i, (name, start) in enumerate(sorted_sections):
        end = (
            sorted_sections[i + 1][1]
            if i + 1 < len(sorted_sections)
            else len(text)
        )
        section_text = text[start:end].strip()
        result[name] = section_text
        print(f"  {name}: {len(section_text)} characters")

    return result