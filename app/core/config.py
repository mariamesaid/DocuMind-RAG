from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "ragpro" / "data"
PDF_DIR = DATA_DIR / "llm wikipedia.pdf"
chunksize = 250
chunk_overlap = 50
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "BAAI/bge-reranker-base"
llm_model_name = "openai/gpt-oss-20b"

Top_K = 3
