"""
Header-based Markdown chunking utility.

Splits Markdown by headers (H1, H2, H3) and further chunks long sections.
Returns dictionaries with header_path, content, and metadata.
"""
from typing import Any, Dict, List, Optional

# Resilient import with helpful error message
try:
    from langchain_text_splitters import MarkdownHeaderTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import MarkdownHeaderTextSplitter
    except ImportError:
        try:
            from langchain_experimental.text_splitter import MarkdownHeaderTextSplitter
        except ImportError:
            raise ImportError(
                "MarkdownHeaderTextSplitter not found. Please install langchain-text-splitters: "
                "pip install langchain-text-splitters>=0.0.1 or pin langchain to a compatible version."
            )

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def _build_header_path(metadata: Dict[str, Any]) -> str:
    """
    Build a deterministic header path from metadata.
    
    Inspects known metadata keys (h1, h2, h3, etc.) in order,
    rather than relying on dict iteration order.
    """
    # Known header level keys in order
    header_keys = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    
    # Check for common alternative keys
    if 'headers' in metadata:
        val = metadata['headers']
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            return ' > '.join(str(h) for h in val if h)
    
    if 'title' in metadata and isinstance(metadata['title'], str):
        return metadata['title']
    
    # Build path from h1, h2, h3, etc. keys in order
    path_parts = []
    for key in header_keys:
        if key in metadata:
            val = metadata[key]
            if val:
                path_parts.append(str(val))
    
    return ' > '.join(path_parts)


def chunk_markdown_with_headers(
    markdown_text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Splits Markdown into chunks using header-based chunking.
    
    Args:
        markdown_text: The Markdown text to chunk.
        chunk_size: Maximum size for each chunk.
        chunk_overlap: Overlap between chunks.
    
    Returns:
        A list of dicts containing:
            {
                "header_path": "...",
                "content": "...",
                "metadata": {...}  # Original metadata from splitter
            }
    
    Raises:
        ValueError: If chunk_overlap < 0 or chunk_overlap >= chunk_size.
    """
    # Validate parameters
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")
    
    # Handle empty input
    if not markdown_text or not markdown_text.strip():
        return []
    
    # Define header levels to track (H1, H2, H3)
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    
    # Step 1 — Split by header structure
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_docs = header_splitter.split_text(markdown_text)
    
    # Step 2 — Further split long sections into smaller chunks
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    final_chunks: List[Dict[str, Any]] = []
    
    for doc in md_docs:
        # Build header path deterministically
        header_path = _build_header_path(doc.metadata)
        
        # Get content
        content = doc.page_content.strip() if hasattr(doc, 'page_content') else str(doc).strip()
        
        # Skip empty content
        if not content:
            continue
        
        # Split into smaller chunks if needed
        smaller_chunks = recursive_splitter.split_text(content)
        
        for chunk in smaller_chunks:
            stripped_chunk = chunk.strip()
            if not stripped_chunk:
                continue
            
            final_chunks.append({
                "header_path": header_path,
                "content": stripped_chunk,
                "metadata": dict(doc.metadata) if hasattr(doc, 'metadata') else {}
            })
    
    return final_chunks
