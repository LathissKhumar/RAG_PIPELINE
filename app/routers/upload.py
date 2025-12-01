from fastapi import APIRouter, UploadFile, File
from typing import List
import os
from app.utils.docling_converter import convert_pdf_to_markdown

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
        markdown_files.append(md_path)
    return {"markdown_files": markdown_files}