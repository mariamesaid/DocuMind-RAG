# scripts/ingest.py
import json
import os
import pathlib
from app.core.config import PDF_DIR, chunksize, chunk_overlap
from app.rag.loader import PDFLoader       # your real names here
from app.rag.chunker import LangchainTextChunker     # your real names here
from app.rag.embeddings import EmbeddingModel  # your real names here
from app.rag.vectorstore import VectorStore

docs = PDFLoader(PDF_DIR).load()              # 1. load
chunks = LangchainTextChunker(chunksize=500, chunk_overlap=50).chunk(docs)  # 2. chunk -> this is the `chunks` variable

# make sure they are plain strings (see note below)
chunks = [c.page_content if hasattr(c, "page_content") else c for c in chunks]

vs = VectorStore(EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2"))               # 3. embed + index
vs.build(chunks)

os.makedirs("storage", exist_ok=True)            # json.dump fails if the folder is missing
vs.save("storage/faiss")

with open("storage/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False)