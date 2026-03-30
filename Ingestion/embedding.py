from Data_Structure import Chunk
from typing import List
import logging

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    Genera embedding con un modello HuggingFace locale tramite sentence-transformers.
    Default: all-MiniLM-L6-v2 (leggero, ottimo per RAG in italiano/inglese).
    Modelli consigliati per l'italiano:
      - paraphrase-multilingual-MiniLM-L12-v2
      - sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None #lazy loading
        
        
    def _load_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("Installa sentence-transformers: pip install sentence-transformers")
            logger.info(f"Caricamento modello embedding: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Modello embedding caricato.")
            
            
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Genera embedding per una lista di testi. Restituisce lista di vettori float."""
        self._load_model()
        logger.info(f"Generazione embedding per {len(texts)} testi...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
            normalize_embeddings=True,  # utile per similarità coseno
        )
        return embeddings.tolist()
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Arricchisce i Chunk con il loro embedding. Modifica in-place e restituisce la lista."""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
            chunk.metadata["embedding_model"] = self.model_name
        logger.info(f"Embedding generati per {len(chunks)} chunk.")
        return chunks
    
    def embed_query(self, query: str) -> List[float]:
        """Genera l'embedding per una singola query (usato al momento del retrieval)."""
        self._load_model()
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()
    
  