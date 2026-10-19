from app.core.config import PDF_DIR, chunksize, chunk_overlap, embedding_model_name, Top_K
from app.rag.loader import PDFLoader
from app.rag.chunker import LangchainTextChunker
from app.rag.embeddings import EmbeddingModel
from app.rag.vectorstore import VectorStore
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

class RAGEngine:
    def __init__(self):
         self.vector_store = None
         self._initialize = ()
        
    def _initialize(self):
        load_dotenv()
        
        text = PDFLoader(PDF_DIR).load()
        chunks = LangchainTextChunker(chunksize, chunk_overlap).chunk(text)
        embeddings = EmbeddingModel(embedding_model_name)
        self.vector_store = VectorStore(embeddings)
        self.vector_store.build(chunks)
        self.llm = ChatGroq(model_name = "llama-3.3-70b-versatile")
        
    def generate_answer(self, query:str):
        contexts = self.vector_store.search(query, top_k=Top_K)
        combined_context = "\n\n".join(contexts)
        prompt_template = f"""
        You are a helpful assistant. Use the following context to answer the question. if the answer is not in the context, say "I don't know".
        context: {combined_context}
        question: {query}
        answer:
        """
        
        agent = create_agent(
            model=self.llm,
            systemprompt="you are a helpful assistant"
        )
        result = agent.invoke({
            "messages":[
                {"role":"user","content":prompt_template}
            ]
        })
        
        return result["messages"][-1].content
    