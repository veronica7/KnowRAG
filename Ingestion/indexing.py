import logging
from typing import List, Optional, Dict, Any

from Data_Structure import Chunk
 
logger = logging.getLogger(__name__)
 
 
class ChromaIndexer:
    """
    Inserisce e interroga chunk su ChromaDB locale (persistente su disco).
    """
 
    def __init__(
        self,
        collection_name: str = "rag_collection",
        persist_directory: str = "./chroma_db",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
 
    def _init_client(self):
        if self._client is None:
            try:
                import chromadb
            except ImportError:
                raise ImportError("Installa chromadb: pip install chromadb")
 
            logger.info(f"Inizializzazione ChromaDB in: {self.persist_directory}")
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # distanza coseno
            )
            logger.info(
                f"Collection '{self.collection_name}' pronta "
                f"({self._collection.count()} documenti esistenti)."
            )
 
    def index_chunks(self, chunks: List[Chunk]) -> int:
        """
        Inserisce i chunk nella collection ChromaDB.
        Richiede che ogni Chunk abbia già un embedding.
        Restituisce il numero di chunk inseriti.
        """
        self._init_client()
 
        chunks_with_embedding = [c for c in chunks if c.embedding is not None]
        if not chunks_with_embedding:
            logger.warning("Nessun chunk con embedding trovato. Indexing saltato.")
            return 0
 
        # ChromaDB vuole liste separate
        ids = [c.metadata["chunk_id"] for c in chunks_with_embedding]
        embeddings = [c.embedding for c in chunks_with_embedding]
        documents = [c.text for c in chunks_with_embedding]
        metadatas = [self._sanitize_metadata(c.metadata) for c in chunks_with_embedding]
 
        # Upsert per evitare duplicati
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
 
        logger.info(f"Indicizzati {len(chunks_with_embedding)} chunk su ChromaDB.")
        return len(chunks_with_embedding)
 
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Esegue una similarity search sul vector store.
        Restituisce una lista di risultati con testo, metadati e distanza.
        """
        self._init_client()
 
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
 
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text": doc,
                "metadata": meta,
                "score": 1 - dist,  # converti distanza coseno in similarità
            })
        return output
 
    def get_collection_info(self) -> Dict[str, Any]:
        self._init_client()
        return {
            "collection_name": self.collection_name,
            "total_documents": self._collection.count(),
            "persist_directory": self.persist_directory,
        }
 
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB accetta solo str, int, float, bool come valori nei metadati."""
        sanitized = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            elif v is None:
                sanitized[k] = ""
            else:
                sanitized[k] = str(v)
        return sanitized