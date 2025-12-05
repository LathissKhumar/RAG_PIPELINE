"""
Recursive text chunking utility.

Splits text recursively using configurable separators,
with improved handling to avoid splitting URLs and numbers.
"""
from typing import List

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def recursive_chunk(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> List[str]:
    """
    Recursively chunk text for LangChain.
    
    Args:
        text: Input text to chunk.
        chunk_size: Maximum size for each chunk.
        chunk_overlap: Overlap between chunks.
    
    Returns:
        A list of stripped text chunks.
    
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
    
    # Separators ordered by preference:
    # - Double newline (paragraph break)
    # - Single newline
    # - Sentence-ending punctuation with trailing space (avoids splitting URLs/numbers)
    # - Space
    # - Empty string (character-level as last resort)
    separators = [
        "\n\n",
        "\n",
        ". ",
        "! ",
        "? ",
        " ",
        ""
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    
    chunks = splitter.split_text(text)
    
    # Strip and filter out empty chunks
    result = []
    for chunk in chunks:
        stripped = chunk.strip()
        if stripped:
            result.append(stripped)
    
    return result


# Example usage
if __name__ == "__main__":
    sample = """# Heading
    This is some long text. It has multiple paragraphs. Each paragraph will be split recursively."""
    print(recursive_chunk(sample))
