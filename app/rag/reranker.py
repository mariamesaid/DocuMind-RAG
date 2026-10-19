from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from typing import Optional
from app.rag.vectorstore import VectorStore
from app.core.config import RERANK_MODEL
from app.rag.embeddings import EmbeddingModel
from app.rag.chunker import LangchainTextChunker
from app.rag.loader import PDFLoader
from app.core.config import PDF_DIR, chunksize, chunk_overlap
from app.rag import ingest
import json
chunks = json.load(open("storage/chunks.json", encoding="utf-8"))

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError:
    from langchain.schema import BaseRetriever, Document


try:
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
except ImportError:
    from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker

def build_retriever(
    top_k: int = 5,
    search_type: str = "similarity",
    metadata_filter: Optional[dict] = None,
    use_hybrid: bool = False,
    use_reranker: bool = False,
    fetch_k: int = 20,
    hybrid_weights: tuple[float, float] = (0.4, 0.6),  # (BM25, vector)
) -> BaseRetriever:
    
    embeddings = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = VectorStore(embeddings)
    vectorstore.load()  # instead of build(); loads the saved index

    k_initial = fetch_k if use_reranker else top_k

    search_kwargs = {"k": k_initial}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    if search_type == "mmr":
        search_kwargs["fetch_k"] = max(fetch_k, k_initial * 2)
        search_kwargs["lambda_mult"] = 0.7

    retriever: BaseRetriever = vectorstore.store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    if use_hybrid:
        bm25 = BM25Retriever.from_texts(chunks)
        bm25.k = k_initial
        retriever = EnsembleRetriever(
            retrievers=[bm25, retriever],
            weights=list(hybrid_weights)),

    if use_reranker:
        cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_k)
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=retriever,
        )

    return retriever