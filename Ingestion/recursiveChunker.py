from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from Data_Structure import BaseChunker

class RecursiveChunker(BaseChunker):
    """Chunking ricorsivo - migliore per testi generali"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)