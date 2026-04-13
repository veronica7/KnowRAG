import logging
from typing import List

from Ingestion import ChromaIndexer
from Query import RetrievedChunk

logger = logging.getLogger(__name__)

def _load_bm25_corpus(indexer: ChromaIndexer):
        """Scarica tutti i chunk da ChromaDB per costruire l'indice BM25."""
        indexer._init_client()
        total = indexer._collection.count()
        if total == 0:
            logger.warning("ChromaDB vuoto. Esegui prima l'ingestion.")
            return [], []
        logger.info(f"Caricamento corpus BM25 ({total} chunk da ChromaDB)...")
        results = indexer._collection.get(include=["documents", "metadatas"])
        texts = results["documents"]
        ids = [m.get("chunk_id", f"chunk_{i}") for i, m in enumerate(results["metadatas"])]
        return texts, ids


def _print_chunks(chunks: List[RetrievedChunk]):
    print("\n── CHUNK SELEZIONATI ─────────────────────────────────")
    for i, c in enumerate(chunks, 1):
        source = c.metadata.get("filename", "?")
        score = getattr(c, "rerank_score", c.hybrid_score)
        print(f"\n[{i}] score={score:.4f} | {source}")
        print(f"    {c.text[:200].replace(chr(10), ' ')}...")
    print("──────────────────────────────────────────────────────\n")