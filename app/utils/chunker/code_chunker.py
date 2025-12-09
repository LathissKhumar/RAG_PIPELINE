"""Code-based chunking helpers.

Uses Docling's code-aware chunking (via `StandardCodeChunkingStrategy`) and
applies a small optimization that merges very small chunks into the previous
chunk to avoid producing too many tiny fragments.

When `output_dir` is provided (defaults to `converted_mds`), the function
creates `converted_mds/<basename>/` and writes the converted markdown and
chunks summary (`chunks.json`) and individual chunk files (`chunk_001.md`...).
"""
from typing import List, Optional
import os
import json
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.transforms.chunker.code_chunking.standard_code_chunking_strategy import (
    StandardCodeChunkingStrategy,
)
from .optimizer import optimize_chunks
from app.embeddings.worker import enqueue_chunk_sync
import logging

logger = logging.getLogger(__name__)


CODE_EXT_LANG: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "java": "java",
    "c": "c",
    "cpp": "c",
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_chunks_to_disk(chunks: List[dict], out_dir: Path, base_name: str, md_file: str) -> None:
    _ensure_dir(out_dir)
    # write summary json
    with open(out_dir / "chunks.json", "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)

    # write individual chunk files
    for i, ch in enumerate(chunks, start=1):
        filename = f"chunk_{i:03}.md"
        with open(out_dir / filename, "w", encoding="utf-8") as fh:
            fh.write(ch.get("text", ""))
        chunk_id = f"{base_name}__{i:03d}"
        enqueue_chunk_sync(chunk_id, ch.get("text", ""), {"source_md": md_file, "chunk_index": i})
        logger.info(f"Chunk {chunk_id} written and enqueued for embedding")


def chunk_code_file(path: str, output_root: str = "converted_mds", min_lines_to_keep: int = 8) -> List[dict]:
    """Chunk a code file at `path` using Docling's chunkers and persist outputs.

    Args:
        path: path to the code file
        output_root: base folder where per-file folders will be created
        min_lines_to_keep: minimum number of lines per chunk; smaller chunks
            will be merged into the previous chunk (optimization).

    Returns:
        List of chunk dicts with `text` and `meta`.
    """
    ext = Path(path).suffix.lstrip(".").lower()
    lang = CODE_EXT_LANG.get(ext, "")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    md = f"```{lang}\n{content}\n```"

    converter = DocumentConverter()
    res = converter.convert_string(content=md, format=InputFormat.MD, name=Path(path).name)
    doc = res.document

    chunker = HierarchicalChunker(code_chunking_strategy=StandardCodeChunkingStrategy())
    raw_chunks = []
    for ch in chunker.chunk(doc):
        try:
            meta = ch.meta.model_dump()
        except Exception:
            try:
                meta = ch.meta.__dict__
            except Exception:
                meta = repr(ch.meta)
        raw_chunks.append({"text": ch.text, "meta": meta})

    # Simple optimization: merge small chunks into previous chunk
    merged: List[dict] = []
    for ch in raw_chunks:
        lines = len(ch.get("text", "").splitlines())
        if merged and lines < min_lines_to_keep:
            # merge into previous
            merged[-1]["text"] += "\n\n" + ch.get("text", "")
            # merge meta lists where possible
            try:
                if isinstance(merged[-1]["meta"], dict) and isinstance(ch["meta"], dict):
                    merged[-1]["meta"].update({"merged_from": merged[-1]["meta"].get("merged_from", []) + [ch["meta"]]})
                else:
                    merged[-1]["meta"] = {"orig": merged[-1]["meta"], "merged": ch.get("meta")}
            except Exception:
                merged[-1]["meta"] = {"note": "could_not_merge_meta"}
        else:
            merged.append(ch)

    # persist outputs
    base_name = Path(path).stem
    out_dir = Path(output_root) / base_name
    _ensure_dir(out_dir)

    # save the converted markdown
    try:
        md_content = doc.export_to_markdown()
    except Exception:
        # fallback: reconstruct from chunks
        md_content = "\n\n".join([c["text"] for c in merged])

    md_file = out_dir / f"{base_name}.md"
    if not md_file.exists():
        with open(md_file, "w", encoding="utf-8") as fh:
            fh.write(md_content)

    # post-process for RAG
    optimized = optimize_chunks(merged)
    _write_chunks_to_disk(optimized, out_dir, base_name, str(md_file))

    return merged
