from fastapi import APIRouter, UploadFile, File
from typing import List
import os
from app.utils.docling_converter import convert_pdf_to_markdown
import json

router = APIRouter()

@router.post("/convert/")
async def convert_pdf(files: List[UploadFile] = File(...)):
    pdf_dir = "uploaded_pdfs"
    md_dir = "converted_mds"
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(md_dir, exist_ok=True)
    markdown_files = []
    for file in files:
        pdf_path = os.path.join(pdf_dir, file.filename)
        with open(pdf_path, "wb") as f:
            f.write(await file.read())
        md_path = convert_pdf_to_markdown(pdf_path, md_dir)
        # Try to read chunk metadata produced by the converter/chunker
        chunks_count = 0
        try:
            folder = os.path.dirname(md_path)
            chunks_json = os.path.join(folder, "chunks.json")
            if os.path.exists(chunks_json):
                with open(chunks_json, "r", encoding="utf-8") as cf:
                    data = json.load(cf)
                    if isinstance(data, list):
                        chunks_count = len(data)
        except Exception:
            chunks_count = 0

        markdown_files.append({"md_path": md_path, "chunks_count": chunks_count})
    return {"markdown_files": markdown_files}