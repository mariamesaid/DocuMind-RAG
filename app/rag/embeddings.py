from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingModel(Embeddings):
  
   def __init__(self, model_name: str):
       self.model = HuggingFaceEmbeddings(
             model_name=model_name
        )
       
   def embed_documents(self, texts):
        return self.model.embed_documents(texts)
    
   def embed_query(self, query:str):
        return self.model.embed_query(query)
        
