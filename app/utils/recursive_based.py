from langchain.text_splitter import RecursiveCharacterTextSplitter
# recursive_chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter

def recursive_chunk(text: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Recursively chunk text for LangChain.
    Returns a list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?"]
    )
    chunks = splitter.split_text(text)
    return chunks

# Example usage
if __name__ == "__main__":
    sample = """# Heading
    This is some long text. It has multiple paragraphs. Each paragraph will be split recursively."""
    print(recursive_chunk(sample))