"""
Code-based chunking utility.

Splits text by code fences (``` and ~~~), keeps code blocks intact or splits
large ones, and uses RecursiveCharacterTextSplitter for prose sections.
"""
import re
from typing import List, Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def _split_large_code_block(code_block: str, max_size: int) -> List[str]:
    """
    Split a large code block into smaller chunks by newlines.
    Preserves fence markers on each chunk for context.
    """
    # Extract fence type and language info from opening line
    lines = code_block.split('\n')
    if not lines:
        return [code_block]
    
    # Find opening fence (first line) and closing fence (last line)
    opening_fence = lines[0]
    # Determine fence type (``` or ~~~)
    fence_marker = '```' if opening_fence.startswith('```') else '~~~'
    
    # Check if there's a closing fence
    has_closing = len(lines) > 1 and lines[-1].strip() in ('```', '~~~')
    
    if has_closing:
        content_lines = lines[1:-1]
        closing_fence = lines[-1]
    else:
        content_lines = lines[1:]
        closing_fence = fence_marker
    
    # If entire block fits, return as-is
    if len(code_block) <= max_size:
        return [code_block.strip()]
    
    # Split content by lines and group into chunks
    chunks = []
    current_chunk_lines = []
    # Account for fence overhead
    fence_overhead = len(opening_fence) + len(closing_fence) + 2  # +2 for newlines
    
    for line in content_lines:
        test_content = '\n'.join(current_chunk_lines + [line])
        if len(test_content) + fence_overhead > max_size and current_chunk_lines:
            # Save current chunk and start new one
            chunk_content = '\n'.join(current_chunk_lines)
            chunk = f"{opening_fence}\n{chunk_content}\n{closing_fence}"
            chunks.append(chunk.strip())
            current_chunk_lines = [line]
        else:
            current_chunk_lines.append(line)
    
    # Add remaining content
    if current_chunk_lines:
        chunk_content = '\n'.join(current_chunk_lines)
        chunk = f"{opening_fence}\n{chunk_content}\n{closing_fence}"
        chunks.append(chunk.strip())
    
    return chunks if chunks else [code_block.strip()]


def code_chunk(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    max_code_chunk_size: Optional[int] = None
) -> List[str]:
    """
    Chunk text, keeping code blocks (``` or ~~~) separate from prose.
    
    Args:
        text: Input text to chunk.
        chunk_size: Maximum size for prose chunks.
        chunk_overlap: Overlap between prose chunks.
        max_code_chunk_size: If set, split large code blocks into smaller chunks.
                            If None, code blocks are kept intact regardless of size.
    
    Returns:
        List of stripped string chunks.
    
    Raises:
        ValueError: If chunk_overlap < 0 or chunk_overlap >= chunk_size.
    """
    # Validate parameters
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")
    
    # Handle empty or whitespace-only input
    if not text or not text.strip():
        return []
    
    # Regex to match both ``` and ~~~ code fences
    # Handles unclosed fences by matching to end of string if no closing fence
    code_fence_pattern = r"(```[\s\S]*?```|~~~[\s\S]*?~~~|```[\s\S]*$|~~~[\s\S]*$)"
    
    parts = re.split(code_fence_pattern, text)
    chunks: List[str] = []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for part in parts:
        stripped_part = part.strip()
        if not stripped_part:
            continue
        
        # Check if this is a code block (starts with ``` or ~~~)
        if stripped_part.startswith("```") or stripped_part.startswith("~~~"):
            if max_code_chunk_size is not None and len(stripped_part) > max_code_chunk_size:
                # Split large code block
                code_chunks = _split_large_code_block(stripped_part, max_code_chunk_size)
                for cc in code_chunks:
                    if cc.strip():
                        chunks.append(cc.strip())
            else:
                chunks.append(stripped_part)
        else:
            # Split prose using RecursiveCharacterTextSplitter
            prose_chunks = splitter.split_text(stripped_part)
            for pc in prose_chunks:
                if pc.strip():
                    chunks.append(pc.strip())
    
    return chunks
