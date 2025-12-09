"""Hybrid fallback chunker that delegates to Docling for all non-code/MD cases.

This module exposes `chunk_source` which takes either a file path or raw text
and an optional `ext` hint. It routes code files to the code chunker, markdown
to the markdown chunker, and everything else to Docling's pipelines and
the `HybridChunker` so that Docling does the heavy lifting. Outputs are
persisted under `converted_mds/<basename>/`.
"""
from typing import List, Optional
import os
import json
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker import HierarchicalChunker, HybridChunker
from docling_core.transforms.chunker.code_chunking.standard_code_chunking_strategy import (
    StandardCodeChunkingStrategy,
)

from .code_chunker import chunk_code_file
from .markdown_chunker import chunk_markdown_text
from .optimizer import optimize_chunks


EXT_CODE = {"py", "js", "ts", "java", "c", "cpp"}
EXT_MD = {"md", "markdown"}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_chunks_summary(chunks: List[dict], out_dir: Path, base_name: str, md_content: str) -> None:
    _ensure_dir(out_dir)
    # write markdown only if it doesn't already exist (avoid overwriting)
    md_file = out_dir / f"{base_name}.md"
    if not md_file.exists():
        with open(md_file, "w", encoding="utf-8") as fh:
            fh.write(md_content)

    with open(out_dir / "chunks.json", "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)

    for i, c in enumerate(chunks, start=1):
        with open(out_dir / f"chunk_{i:03}.md", "w", encoding="utf-8") as fh:
            fh.write(c.get("text", ""))


def chunk_source(
    path: Optional[str] = None,
    text: Optional[str] = None,
    ext: Optional[str] = None,
    output_root: str = "converted_mds",
) -> List[dict]:
    """Chunk a source. Routes known types to specialized chunkers and uses
    Docling's `HybridChunker` as a fallback for other document types.

    Outputs are stored under `converted_mds/<basename>/` when `path` is
    provided (or when `name` can be inferred). Returns the list of chunks.
    """
    if path and not ext:
        ext = Path(path).suffix.lstrip('.').lower()

    if ext in EXT_CODE and path:
        return chunk_code_file(path, output_root)

    if ext in EXT_MD and text is not None:
        return chunk_markdown_text(text, name=(Path(path).name if path else "doc.md"), output_root=output_root)

    if text is not None and (ext in EXT_MD):
        return chunk_markdown_text(text, name=(Path(path).name if path else "doc.md"), output_root=output_root)

    converter = DocumentConverter()
    if path:
        res = converter.convert(path)
        base_name = Path(path).stem
    elif text is not None:
        # assume markdown if no path given
        res = converter.convert_string(content=text, format=InputFormat.MD)
        base_name = "text_input"
    else:
        raise ValueError("Either path or text must be provided")

    doc = res.document

    # Use HybridChunker which combines page-aware and hierarchical strategies
    chunker = HybridChunker(code_chunking_strategy=StandardCodeChunkingStrategy())
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
    out_dir = Path(output_root) / base_name
    try:
        md_content = doc.export_to_markdown()
    except Exception:
        md_content = "\n\n".join([c["text"] for c in chunks])

    # optimize chunks for RAG
    optimized = optimize_chunks(chunks)

    _write_chunks_summary(optimized, out_dir, base_name, md_content)

    return optimized
