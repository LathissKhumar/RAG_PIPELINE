from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from pydantic import BaseModel
import os
import logging
from app.utils.docling_converter import convert_pdf_to_markdown
import json

logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration
MAX_FILE_SIZE_MB = 50  # 50 MB limit per file
ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILES_PER_REQUEST = 10

# Response models
class UploadErrorDetail(BaseModel):
    filename: str
    error: str

class ConversionResult(BaseModel):
    filename: str
    md_path: str
    chunks_count: int

class ConvertResponse(BaseModel):
    successful: List[ConversionResult]
    failed: List[UploadErrorDetail]
    total_processed: int
    total_successful: int
    total_failed: int

@router.post("/convert/", response_model=ConvertResponse)
async def convert_pdf(files: List[UploadFile] = File(...)):
    """Convert PDF files to Markdown with chunking"""
    
    # Validate input
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400, 
            detail=f"Too many files (max {MAX_FILES_PER_REQUEST})"
        )
    
    pdf_dir = "uploaded_pdfs"
    md_dir = "converted_mds"
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(md_dir, exist_ok=True)
    
    successful: List[ConversionResult] = []
    failed: List[UploadErrorDetail] = []
    
    for file in files:
        try:
            # Validate file extension
            filename = file.filename or "unknown"
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                raise ValueError(f"Invalid file type: {file_ext}. Only .pdf files allowed")
            
            # Read and validate file size
            content = await file.read()
            file_size_mb = len(content) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                raise ValueError(f"File too large: {file_size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)")
            
            if len(content) == 0:
                raise ValueError("File is empty")
            
            # Save PDF
            pdf_path = os.path.join(pdf_dir, filename)
            with open(pdf_path, "wb") as f:
                f.write(content)
            
            # Convert to Markdown
            logger.info(f"Converting {filename} to markdown")
            md_path = convert_pdf_to_markdown(pdf_path, md_dir)
            
            # Count chunks
            chunks_count = 0
            try:
                folder = os.path.dirname(md_path)
                chunks_json = os.path.join(folder, "chunks.json")
                if os.path.exists(chunks_json):
                    with open(chunks_json, "r", encoding="utf-8") as cf:
                        data = json.load(cf)
                        if isinstance(data, list):
                            chunks_count = len(data)
            except Exception as e:
                logger.warning(f"Could not read chunks.json for {filename}: {e}")
            
            logger.info(f"Successfully converted {filename} -> {chunks_count} chunks")
            successful.append(ConversionResult(
                filename=filename,
                md_path=md_path,
                chunks_count=chunks_count
            ))
            
        except ValueError as e:
            logger.warning(f"Validation error for {file.filename}: {e}")
            failed.append(UploadErrorDetail(
                filename=file.filename or "unknown",
                error=str(e)
            ))
        except Exception as e:
            logger.error(f"Conversion error for {file.filename}: {e}", exc_info=True)
            failed.append(UploadErrorDetail(
                filename=file.filename or "unknown",
                error=f"Conversion failed: {str(e)}"
            ))
    
    return ConvertResponse(
        successful=successful,
        failed=failed,
        total_processed=len(files),
        total_successful=len(successful),
        total_failed=len(failed)
    )