import os
import json
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions,PictureDescriptionVlmOptions
from typing import Any, Optional
from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (MarkdownDocSerializer, MarkdownParams, MarkdownPictureSerializer)
from docling_core.types.doc import DoclingDocument, PictureItem, ImageRefMode


def convert_pdf_to_markdown(pdf_path: str, output_dir: str) -> str:
    """Convert PDF to Markdown with VLM image descriptions replacing embedded images."""
    

    
    picture_description_options = PictureDescriptionVlmOptions(
        repo_id="HuggingFaceTB/SmolVLM-256M-Instruct", 
        prompt="""Describe this image in detail without missing any details and elements.
        Provide the description in complete sentences.The description should act like an alternative text describing
        the entire image for someone who cannot see it. Be specific about colors, objects, people, actions, and context.""",
    
        device="cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu",
    )
    
    pipeline_options = PdfPipelineOptions(
        do_picture_description=True,
        generate_picture_images=True,
        do_code_enrichment=True,
        do_formula_enrichment=True,
        force_backend_text=False,
        generate_page_images=True,
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

                text = "\n\n".join(parts) if parts else "*[Image]*"
                return create_ser_result(text=text, span_source=item)

        markdown_content = MarkdownDocSerializer(
            doc=result.document,
            picture_serializer=AnnotationPictureSerializer(),
            params=MarkdownParams(image_mode=ImageRefMode.PLACEHOLDER, image_placeholder="")
        ).serialize().text
        
    except Exception:
        markdown_content = result.document.export_to_markdown()

    md_path = os.path.join(output_dir, os.path.splitext(os.path.basename(pdf_path))[0] + ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return md_path
