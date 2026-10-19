from langchain_community.vectorstores import FAISS

class VectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.store = None

    def build(self, texts):
        """Build a FAISS index from texts."""
        self.store = FAISS.from_texts(texts=texts, embedding=self.embeddings)
    def save(self, path="storage/faiss"):
        self.store.save_local(path)

    def load(self, path="storage/faiss"):
        self.store = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
    def search(self, query: str, top_k: int = 3):
        """Search the FAISS index for the most relevant documents."""
        if self.store is None:
            raise ValueError("Vector store has not been built yet.")
        docs = self.store.similarity_search(query, k=top_k)
        return [doc.page_content for doc in docs]