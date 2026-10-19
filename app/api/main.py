from fastapi import FastAPI,Query
from app.rag.engine import RAGEngine
from pydantic import BaseModel
from app.rag.reranker import build_retriever
from app.core.config import RERANK_MODEL
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from app.rag.vectorstore import VectorStore
from app.rag.embeddings import EmbeddingModel
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

load_dotenv()
embedding = EmbeddingModel("BAAI/bge-small-en-v1.5")
vectorstore = VectorStore(embedding.model)
vectorstore.load("storage/faiss")

#vectorstore = VectorStore(embeddings=EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2"))
#vectorstore.load("storage/faiss")
llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=GROQ_API_KEY)

#retriever = build_retriever(model_name="BAAI/bge-reranker-base")

app = FastAPI()
rag_engine = RAGEngine()

@app.get("/")
def root():
    return {"status": "ok", "message": "Send a GET request to /query, or open /docs"}
    
class Question(BaseModel):
    question: str
 
 
@app.post("/query")
def query(q: Question):
    chunks = vectorstore.search(q.question, top_k=4)
    context = "\n\n".join(chunks)
    prompt = (
        "Answer only from the context below. If the answer is not there, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {q.question}"
    )
    answer = llm.invoke(prompt).content
    return {"answer": answer, "sources": chunks}