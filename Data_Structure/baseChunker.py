from typing import List
from abc import abstractmethod
from Data_Structure import Chunk, Document
class BaseChunker:
    """Classe base per chunking strategies"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.name = self.__class__.__name__

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Splitta un Document in una lista di Chunk, preservando i metadati."""
        raw_chunks = self.split_text(document.content)
        chunks = []
        for i, text in enumerate(raw_chunks):
            meta = {
                **document.metadata,
                "chunk_index": i,
                "chunk_id": f"{document.metadata.get('doc_id', 'doc')}_{i}",
                "chunker": self.__class__.__name__,
                "chunk_size_cfg": self.chunk_size,
                "chunk_overlap_cfg": self.chunk_overlap,
            }
            chunks.append(Chunk(text=text, metadata=meta))
        return chunks