from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO
from fastapi import UploadFile

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    return " ".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_docx(docx_bytes: bytes) -> str:
    doc = Document(BytesIO(docx_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_upload(file: UploadFile) -> str:
    contents = file.file.read()
    if file.filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(contents)
    elif file.filename.lower().endswith(".docx"):
        return extract_text_from_docx(contents)
    else:
        raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")

def summarize_medical_text(text: str, model: str = "default") -> str:
    if len(text) < 100:
        return "Document too short for meaningful summary."
    return f"📄 Summary (using {model}):\n\n{text[:10000]}..."