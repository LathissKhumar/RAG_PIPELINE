import os
import json
import logging
from pathlib import Path
from typing import Any
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionVlmOptions
from app.utils.chunker.markdown_chunker import chunk_markdown_text
from app.embeddings.worker import enqueue_chunk_sync
from app.vector_store.chroma_client import get_chroma_client, filter_missing_ids
from app.utils.file_registry import get_file_registry
from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (MarkdownDocSerializer, MarkdownParams, MarkdownPictureSerializer)
from docling_core.types.doc import DoclingDocument, PictureItem, ImageRefMode

logger = logging.getLogger(__name__)
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")


def _reenqueue_missing_chunks(md_path: str, output_dir: str) -> None:
    """Re-enqueue any chunks whose vectors are missing from Chroma."""
    try:
        base_name = Path(md_path).stem
        out_dir = Path(output_dir) / base_name
        chunks_json = out_dir / "chunks.json"
        if not chunks_json.exists():
            return
        with open(chunks_json, "r", encoding="utf-8") as fh:
            chunks = json.load(fh)
        chunk_ids = [f"{base_name}__{i:03d}" for i in range(1, len(chunks) + 1)]
        client = get_chroma_client()
        missing_ids = set(filter_missing_ids(client, CHROMA_COLLECTION, chunk_ids))
        if not missing_ids:
            return
        logger.info(f"Re-enqueuing {len(missing_ids)} missing chunks for {base_name}")
        for idx, chunk in enumerate(chunks, start=1):
            chunk_id = f"{base_name}__{idx:03d}"
            if chunk_id not in missing_ids:
                continue
            enqueue_chunk_sync(chunk_id, chunk.get("text", ""), {"source_md": str(out_dir / f"{base_name}.md"), "chunk_index": idx})
    except Exception as e:
        logger.warning(f"Failed to re-enqueue missing chunks for {md_path}: {e}")


def convert_pdf_to_markdown(pdf_path: str, output_dir: str) -> str:
    """Convert PDF to Markdown with VLM image descriptions replacing embedded images."""
    
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_folder = os.path.join(output_dir, base_name)
    md_path = os.path.join(out_folder, base_name + ".md")
    
    # Check file registry: if hash matches and MD exists, skip conversion
    registry = get_file_registry()
    try:
        current_hash = registry.compute_file_hash(pdf_path)
        if registry.should_skip_conversion(pdf_path, current_hash):
            logger.info(f"File hash matched in registry; skipping conversion for {base_name}")
            if os.path.exists(md_path):
                _reenqueue_missing_chunks(md_path, output_dir)
            return md_path
    except Exception as e:
        logger.warning(f"File registry check failed for {pdf_path}: {e}")
    
    # Cache: if markdown already exists, re-enqueue any chunks missing in Chroma and return
    if os.path.exists(md_path):
        _reenqueue_missing_chunks(md_path, output_dir)
        return md_path
    
    picture_description_options = PictureDescriptionVlmOptions(
        repo_id="HuggingFaceTB/SmolVLM-500M-Instruct",  # Upgraded from 256M for better descriptions
        prompt="""Describe this image in detail without missing any details and elements.
        Provide the description in complete sentences.The description should act like an alternative text describing
        the entire image for someone who cannot see it. Be specific about colors, objects, people, actions, and context.""",
    
        device="cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu",
    )
    
    pipeline_options = PdfPipelineOptions(
        do_picture_description=True,
        generate_picture_images=True,  # Required for VLM descriptions
        do_code_enrichment=False,  # Disable for speed if not needed
        do_formula_enrichment=False,  # Disable for speed if not needed
        images_scale=2.0,
        picture_description_options=picture_description_options,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(pdf_path)

    try:
        

        class AnnotationPictureSerializer(MarkdownPictureSerializer):
            def serialize(self, *, item: PictureItem, doc_serializer: BaseDocSerializer, 
                         doc: DoclingDocument, **kwargs: Any) -> SerializationResult:
                parts = []
                
                # Get caption
                try:
                    if hasattr(item, 'caption_text'):
                        caption = item.caption_text(doc)
                        if caption:
                            parts.append(f"**Caption:** {caption.strip()}")
                except Exception:
                    pass
                
                # Get VLM annotations
                for ann in getattr(item, "annotations", []):
                    if txt := getattr(ann, "text", None):
                        parts.append(f"**Description:** {txt.strip()}")

                # Always emit text to replace the image, ensuring no images in output
                if parts:
                    text = "\n\n".join(parts)
                else:
                    text = "**[Image: description not available]**"
                
                # Ensure proper markdown separation
                text = "\n\n" + text.strip() + "\n\n"
                return create_ser_result(text=text, span_source=item)

        markdown_content = MarkdownDocSerializer(
            doc=result.document,
            picture_serializer=AnnotationPictureSerializer(),
            params=MarkdownParams(image_mode=ImageRefMode.PLACEHOLDER, image_placeholder="")
        ).serialize().text
        
    except Exception:
        markdown_content = result.document.export_to_markdown()

    os.makedirs(out_folder, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    # Register file in registry after successful conversion
    try:
        current_hash = registry.compute_file_hash(pdf_path)
        registry.register_file(pdf_path, current_hash, md_path)
    except Exception as e:
        logger.warning(f"Failed to register file in registry: {e}")

    # Now chunk the converted markdown and persist chunks in the same folder
    try:
        # chunk_markdown_text will create the same folder under output_dir/base_name
        chunk_markdown_text(markdown_content, name=base_name + ".md", output_root=output_dir)
    except Exception:
        # If chunking fails, continue but surface the markdown path
        pass

    return md_path
