# app/utils/header.py

from langchain_experimental.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_markdown_with_headers(markdown_text: str):
    """
    Splits Markdown into chunks using header-based chunking.
    Returns a list of dicts containing:
        { "header_path": "...", "content": "..." }
    """

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
        chunk_size=1200,
        chunk_overlap=150,
        length_function=len,
    )

    final_chunks = []

    for doc in md_docs:
        header_path = " > ".join(v for v in doc.metadata.values() if v)
        smaller_chunks = recursive_splitter.split_text(doc.page_content)

        for chunk in smaller_chunks:
            final_chunks.append({
                "header_path": header_path,
                "content": chunk.strip()
            })

    return final_chunks
