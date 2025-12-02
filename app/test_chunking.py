from app.chunking import Chunker

sample_text = """
Docling extracts text, images, and structure from documents.
We now need to chunk this extracted text using LangChain to feed into embeddings.
This improves RAG quality.
"""

chunker = Chunker()
chunks = chunker.chunk_text(sample_text)

print("\n----- CHUNKS -----")
for c in chunks:
    print(c)
    print("-------------------")
