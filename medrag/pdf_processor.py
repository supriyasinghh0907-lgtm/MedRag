import fitz  # PyMuPDF


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF held in memory. Returns IMAGE_ONLY_PDF for scans."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = "".join(page.get_text() for page in doc)
    if len(text.strip()) < 30:
        return "IMAGE_ONLY_PDF"
    return text


def chunk_text(text: str, size: int = 400, overlap: int = 50):
    """Split text into ~400-word chunks with a small overlap so ideas aren't cut in half."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), step) if words[i:i + size]]
