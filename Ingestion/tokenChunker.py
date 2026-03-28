from langchain.text_splitter import TokenTextSplitter
from typing import List
from Data_Structure import BaseChunker

class TokenChunker(BaseChunker):
    """Chunking basato su token - migliore per controllo preciso"""

    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)
    
    