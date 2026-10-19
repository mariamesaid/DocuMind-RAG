from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

class LangchainTextChunker:
    """chunk raw text into smaller pieces for processing"""
    
    def __init__(self,chunksize: int,chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunksize,
            chunk_overlap=chunk_overlap
        )
    def chunk(self , text:str) ->list[str]:
        return self.splitter.split_text(text)
    