import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Risultato del retrieval con testo, metadati e punteggi."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    vector_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0  # score finale dopo fusione RRF

    def __repr__(self):
        preview = self.text[:80].replace("\n", " ")
        return (
            f"RetrievedChunk(id={self.chunk_id}, "
            f"hybrid={self.hybrid_score:.4f}, "
            f"preview='{preview}...')"
        )


class HybridRetriever:
    """
    Retrieval ibrido: combina vector search (ChromaDB) e BM25 lessicale.
    La fusione avviene con Reciprocal Rank Fusion (RRF), che è
    rank-based e non richiede normalizzazione dei punteggi.

    Args:
        indexer: istanza di ChromaIndexer già inizializzata
        embedder: istanza di EmbeddingModel già inizializzata
        bm25_corpus: lista di testi su cui addestrare BM25
                     (tipicamente tutti i chunk indicizzati)
        bm25_corpus_ids: lista di chunk_id corrispondenti ai testi
        top_k: numero di risultati finali da restituire
        rrf_k: costante RRF (default 60, suggerito in letteratura)
        vector_weight / bm25_weight: pesi per la fusione pesata dei rank
    """

    def __init__(
        self,
        indexer,
        embedder,
        bm25_corpus: Optional[List[str]] = None,
        bm25_corpus_ids: Optional[List[str]] = None,
        top_k: int = 5,
        rrf_k: int = 60,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        self.indexer = indexer
        self.embedder = embedder
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        self._bm25 = None
        self._bm25_corpus_ids: List[str] = []

        if bm25_corpus and bm25_corpus_ids:
            self.build_bm25_index(bm25_corpus, bm25_corpus_ids)

    # ─────────────────────────────────────────────
    # COSTRUZIONE INDICE BM25
    # ─────────────────────────────────────────────

    def build_bm25_index(self, corpus: List[str], corpus_ids: List[str]):
        """
        Addestra BM25 sul corpus fornito.
        corpus: lista di testi (un elemento per chunk)
        corpus_ids: lista di chunk_id nella stessa posizione
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Installa rank-bm25: pip install rank-bm25")

        import re

        def tokenize(text: str) -> List[str]:
            return re.findall(r'\b\w+\b', text.lower())

        tokenized = [tokenize(doc) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_corpus_ids = corpus_ids
        logger.info(f"Indice BM25 costruito su {len(corpus)} documenti.")

    # ─────────────────────────────────────────────
    # RETRIEVAL
    # ─────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        query_tokens: Optional[List[str]] = None,
        n_candidates: int = 20,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """
        Esegue il retrieval ibrido e restituisce i top_k chunk.

        Args:
            query: testo della query (già processata)
            query_tokens: token BM25 precomputati (opzionali)
            n_candidates: quanti candidati raccogliere per lato prima della fusione
            metadata_filter: filtro opzionale ChromaDB (es. {"source": "doc.pdf"})
        """
        # --- VECTOR SEARCH ---
        vector_results = self._vector_search(query, n_candidates, metadata_filter)

        # --- BM25 SEARCH ---
        if self._bm25 is not None:
            import re
            tokens = query_tokens or re.findall(r'\b\w+\b', query.lower())
            bm25_results = self._bm25_search(tokens, n_candidates)
        else:
            logger.warning("Indice BM25 non inizializzato: uso solo vector search.")
            bm25_results = []

        # --- FUSIONE RRF ---
        merged = self._reciprocal_rank_fusion(vector_results, bm25_results)

        logger.info(
            f"Retrieval ibrido: {len(vector_results)} vector, "
            f"{len(bm25_results)} BM25 → {len(merged)} dopo fusione → top {self.top_k}"
        )
        return merged[: self.top_k]

    # ─────────────────────────────────────────────
    # METODI INTERNI
    # ─────────────────────────────────────────────

    def _vector_search(
        self,
        query: str,
        n: int,
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[RetrievedChunk]:
        query_embedding = self.embedder.embed_query(query)
        raw = self.indexer.query(
            query_embedding=query_embedding,
            n_results=n,
            where=metadata_filter,
        )
        results = []
        for item in raw:
            chunk_id = item["metadata"].get("chunk_id", "unknown")
            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=item["text"],
                metadata=item["metadata"],
                vector_score=item["score"],
            ))
        return results

    def _bm25_search(self, tokens: List[str], n: int) -> List[RetrievedChunk]:
        scores = self._bm25.get_scores(tokens)
        # Ordina per score decrescente e prende top-n
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk_id = self._bm25_corpus_ids[idx]
            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                text="",           # testo recuperato dopo fusione da ChromaDB
                metadata={},
                bm25_score=float(scores[idx]),
            ))
        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievedChunk],
        bm25_results: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Reciprocal Rank Fusion: score_rrf = Σ weight_i / (k + rank_i)
        Unifica i risultati dei due retriever usando i rank, non i punteggi raw.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        # Contributo vector
        for rank, chunk in enumerate(vector_results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                self.vector_weight / (self.rrf_k + rank + 1)
            )
            chunk_map[chunk.chunk_id] = chunk

        # Contributo BM25
        for rank, chunk in enumerate(bm25_results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                self.bm25_weight / (self.rrf_k + rank + 1)
            )
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

        # Assegna hybrid_score e ordina
        merged = []
        for chunk_id, score in rrf_scores.items():
            chunk = chunk_map[chunk_id]
            chunk.hybrid_score = score
            merged.append(chunk)

        merged.sort(key=lambda c: c.hybrid_score, reverse=True)

        # Recupera testo/metadata da ChromaDB per i chunk arrivati solo da BM25
        self._hydrate_from_chroma(merged)

        return merged

    def _hydrate_from_chroma(self, chunks: List[RetrievedChunk]):
        """Recupera testo e metadati mancanti (chunk provenienti solo da BM25)."""
        for chunk in chunks:
            if not chunk.text:
                try:
                    results = self.indexer._collection.get(
                        ids=[chunk.chunk_id],
                        include=["documents", "metadatas"],
                    )
                    if results["documents"]:
                        chunk.text = results["documents"][0]
                        chunk.metadata = results["metadatas"][0]
                except Exception as e:
                    logger.warning(f"Impossibile recuperare chunk {chunk.chunk_id}: {e}")