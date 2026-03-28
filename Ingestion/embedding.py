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
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("Installa sentence-transformers: pip install sentence-transformers")
            logger.info(f"Caricamento modello embedding: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Modello embedding caricato.")
            
            
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Genera embedding per una lista di testi. Restituisce lista di vettori float."""
        self._load_model()
        logger.info(f"Generazione embedding per {len(texts)} testi...")
        embeddings = self._model.encode(
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
        embedding = self._model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()
    
    
    #     print("Inizializzazione RAG System...")

    #     # Embedding model
    #     self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    #     self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

    #     # Vector database (ChromaDB)
    #     self.chroma_client = chromadb.Client()
    #     try:
    #         self.chroma_client.delete_collection(collection_name)
    #     except:
    #         pass
    #     self.collection = self.chroma_client.create_collection(
    #         name=collection_name,
    #         metadata={"hnsw:space": "cosine"}
    #     )

    #     # Sparse retrieval (BM25)
    #     self.bm25 = None
    #     self.documents = []
    #     self.doc_embeddings = None

    #     # Chunking strategy
    #     self.chunker = RecursiveChunker(chunk_size=800, chunk_overlap=100)

    #     print(f"✓ RAG System pronto (dimensione embedding: {self.embedding_dim})")
  
    
    # def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
    #     """Aggiunge documenti al sistema"""
    #     print(f"Aggiungendo {len(texts)} documenti...")

    #     if metadatas is None:
    #         metadatas = [{"doc_id": i} for i in range(len(texts))]

    #     # Chunking
    #     all_chunks = []
    #     all_chunk_metadatas = []

    #     for i, text in enumerate(texts):
    #         chunks = self.chunker.split_text(text)
    #         for j, chunk in enumerate(chunks):
    #             all_chunks.append(chunk)
    #             chunk_metadata = {
    #                 **metadatas[i],
    #                 "chunk_id": f"doc{i}_chunk{j}",
    #                 "chunk_index": j
    #             }
    #             all_chunk_metadatas.append(chunk_metadata)

    #     # Dense embeddings
    #     embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)

    #     # Aggiungi a ChromaDB
    #     ids = [meta["chunk_id"] for meta in all_chunk_metadatas]
    #     self.collection.add(
    #         embeddings=embeddings.tolist(),
    #         documents=all_chunks,
    #         metadatas=all_chunk_metadatas,
    #         ids=ids
    #     )

    #     # Prepara BM25
    #     tokenized_chunks = [self._tokenize(chunk) for chunk in all_chunks]
    #     self.bm25 = BM25Okapi(tokenized_chunks)
    #     self.documents = all_chunks
    #     self.doc_embeddings = embeddings

    #     print(f"✓ Aggiunti {len(all_chunks)} chunk da {len(texts)} documenti")    