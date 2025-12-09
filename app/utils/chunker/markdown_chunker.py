"""Markdown/header-based chunking helpers.

Use Docling to convert/parse Markdown and then chunk using the
`HierarchicalChunker`, which respects section headers. This is suitable for
Markdown documents where you want chunks grouped by headings.
"""
from typing import List, Optional
import os
import json
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker import HierarchicalChunker
from .optimizer import optimize_chunks


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def chunk_markdown_text(text: str, name: str = "doc.md", output_root: str = "converted_mds") -> List[dict]:
    """Chunk markdown text by headers using Docling's hierarchical chunker.

    If `output_root` is provided, writes `converted_mds/<name_without_ext>/<name>.md`
    and chunk files (`chunks.json` and `chunk_###.md`).
    """
    converter = DocumentConverter()
    res = converter.convert_string(content=text, format=InputFormat.MD, name=name)
    doc = res.document

    chunker = HierarchicalChunker()
    chunks = []
    for ch in chunker.chunk(doc):
        try:
            meta = ch.meta.model_dump()
        except Exception:
            try:
                meta = ch.meta.__dict__
            except Exception:
                meta = repr(ch.meta)

        chunks.append({"text": ch.text, "meta": meta})

    # persist outputs
    base_name = Path(name).stem
    out_dir = Path(output_root) / base_name
    _ensure_dir(out_dir)

    try:
        md_content = doc.export_to_markdown()
    except Exception:
        md_content = "\n\n".join([c["text"] for c in chunks])

    md_file = out_dir / f"{base_name}.md"
    if not md_file.exists():
        with open(md_file, "w", encoding="utf-8") as fh:
            fh.write(md_content)

    # post-process chunks for RAG: merge/split and add overlap
    optimized = optimize_chunks(chunks)

    # write chunks.json and individual chunk files
    with open(out_dir / "chunks.json", "w", encoding="utf-8") as fh:
        json.dump(optimized, fh, ensure_ascii=False, indent=2)

    for i, c in enumerate(optimized, start=1):
        with open(out_dir / f"chunk_{i:03}.md", "w", encoding="utf-8") as fh:
            fh.write(c.get("text", ""))

    return chunks
