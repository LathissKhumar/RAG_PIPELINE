import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def code_chunk(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Chunk code and normal text separately for LangChain embeddings.
    Keeps ``` code ``` blocks intact.
    """
    parts = re.split(r"(```[\s\S]*?```)", text)
    chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n"]
    )

    for part in parts:
        if part.startswith("```"):
            chunks.append(part)  # Keep code block as a single chunk
        else:
            chunks.extend(splitter.split_text(part))
    
    return chunks