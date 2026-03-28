from .DocumentLoader import DocumentPipeline
from .recursiveChunker import RecursiveChunker
from Data_Structure import BaseChunker, Document, Chunk
from .embedding import EmbeddingModel
from .indexing import ChromaIndexer
from typing import Optional, List

import re
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


class IngestionPipeline:
    """
    Pipeline di ingestion completa: caricamento → chunking → embedding → indexing.
 
    Utilizzo minimo:
        pipeline = IngestionPipeline(data_path="./miei_documenti")
        result = pipeline.run()
 
    Utilizzo avanzato:
        pipeline = IngestionPipeline(
            data_path="./miei_documenti",
            chunker=TokenChunker(chunk_size=300, chunk_overlap=50),
            embedding_model=EmbeddingModel("paraphrase-multilingual-MiniLM-L12-v2"),
            collection_name="mia_collection",
            persist_directory="./chroma_db",
        )
        result = pipeline.run()
    """
    
    def __init__(
        self,
        data_path: str,
        chunker: Optional[BaseChunker] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        collection_name: str = "rag_collection",
        persist_directory: str = "./chroma_db",
    ):
        self.data_path = data_path
 
        # Componenti con default sensati
        self.document_pipeline = DocumentPipeline(data_path)
        self.chunker = chunker or RecursiveChunker(chunk_size=1000, chunk_overlap=200)
        self.embedding_model = embedding_model or EmbeddingModel()
        self.indexer = ChromaIndexer(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        
    # ─────────────────────────────────────────────
    # ENTRY POINT PRINCIPALE
    # ─────────────────────────────────────────────
 
    def run(self) -> dict:
        """
        Avvia la pipeline completa con una singola chiamata.
        Restituisce un dizionario con le statistiche di esecuzione.
        """
        logger.info("=" * 50)
        logger.info("AVVIO INGESTION PIPELINE")
        logger.info("=" * 50)
 
        # 1. Caricamento documenti
        documents = self._step_load()
        if not documents:
            logger.warning("Nessun documento caricato. Pipeline interrotta.")
            return {"status": "empty", "documents": 0, "chunks": 0, "indexed": 0}
 
        # 2. Chunking
        chunks = self._step_chunk(documents)
        if not chunks:
            logger.warning("Nessun chunk prodotto. Pipeline interrotta.")
            return {"status": "empty", "documents": len(documents), "chunks": 0, "indexed": 0}
 
        # 3. Embedding
        chunks = self._step_embed(chunks)
 
        # 4. Indexing
        indexed_count = self._step_index(chunks)
 
        # 5. Report finale
        info = self.indexer.get_collection_info()
        result = {
            "status": "success",
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "chunks_indexed": indexed_count,
            "collection_total": info["total_documents"],
            "collection_name": info["collection_name"],
            "persist_directory": info["persist_directory"],
        }
        logger.info("=" * 50)
        logger.info(f"PIPELINE COMPLETATA: {result}")
        logger.info("=" * 50)
        return result
 
    # ─────────────────────────────────────────────
    # STEP INTERNI
    # ─────────────────────────────────────────────
 
    def _step_load(self) -> List[Document]:
        logger.info("Caricamento documenti...")
        documents = self.document_pipeline.load_documents()
        logger.info(f"  → {len(documents)} documenti caricati")
        return documents
 
    def _step_chunk(self, documents: List[Document]) -> List[Chunk]:
        logger.info(f"Chunking con {self.chunker.__class__.__name__}...")
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        logger.info(f"  → {len(all_chunks)} chunk prodotti da {len(documents)} documenti")
        return all_chunks
 
    def _step_embed(self, chunks: List[Chunk]) -> List[Chunk]:
        logger.info(f"[Generazione embedding con {self.embedding_model.model_name}...")
        chunks = self.embedding_model.embed_chunks(chunks)
        logger.info(f"  → Embedding generati per {len(chunks)} chunk")
        return chunks
 
    def _step_index(self, chunks: List[Chunk]) -> int:
        logger.info("Indexing su ChromaDB...")
        indexed = self.indexer.index_chunks(chunks)
        logger.info(f"  → {indexed} chunk inseriti nel vector store")
        return indexed
 
    # ─────────────────────────────────────────────
    # UTILITÀ BM25 (per hybrid retrieval futuro)
    # ─────────────────────────────────────────────
 
    def _tokenize(self, text: str) -> List[str]:
        """Tokenizzazione semplice per BM25."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return [t for t in tokens if len(t) > 2]
    
        
    
    