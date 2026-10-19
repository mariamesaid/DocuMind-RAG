#from pypdf import PdfReader
from pathlib import Path
import pymupdf  
class PDFLoader:
    """Load raw text from a PDF file."""

    def __init__(self, pdf_path):
        self.pdf_path = Path(pdf_path)          
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

    def load(self) -> str:
        """Return the full text of the PDF as one string."""
        pages_text = []
        try:
            with pymupdf.open(str(self.pdf_path)) as doc:
                for page in doc:
                    text = page.get_text()
                    if text.strip():
                        pages_text.append(text)
        except Exception as e:
            raise RuntimeError(f"Error reading PDF file: {e}")
        return "\n".join(pages_text)           