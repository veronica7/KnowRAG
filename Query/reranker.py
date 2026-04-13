import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranking dei chunk recuperati tramite Cross-Encoder HuggingFace.

    A differenza del Bi-Encoder (usato per gli embedding), il Cross-Encoder
    valuta la coppia (query, documento) insieme, producendo un punteggio di
    rilevanza molto più preciso — ma troppo lento per retrieval su tutto il corpus.
    Si usa quindi in cascade: retrieval veloce → reranking preciso su top-N.

    Modelli consigliati:
      - cross-encoder/ms-marco-MiniLM-L-6-v2   (veloce, inglese)
      - cross-encoder/ms-marco-MiniLM-L-12-v2  (più preciso, inglese)
      - cross-encoder/mmarco-mMiniLMv2-L12-H384-v1  (multilingua, include italiano)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ):
        """
        Args:
            model_name: nome del modello Cross-Encoder su HuggingFace Hub
            top_n: quanti chunk restituire dopo il reranking (None = tutti)
            score_threshold: filtra i chunk con score sotto questa soglia (None = no filtro)
        """
        self.model_name = model_name
        self.top_n = top_n
        self.score_threshold = score_threshold
        self._model = None  # lazy loading

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "Installa sentence-transformers: pip install sentence-transformers"
                )
            logger.info(f"Caricamento Cross-Encoder: {self.model_name}")
            self._model = CrossEncoder(self.model_name, max_length=512)
            logger.info("Cross-Encoder caricato.")

    def rerank(self, query: str, chunks) -> list:
        """
        Riordina i chunk in base alla rilevanza rispetto alla query.

        Args:
            query: testo della query processata
            chunks: lista di RetrievedChunk

        Returns:
            Lista di RetrievedChunk riordinata per score Cross-Encoder decrescente,
            con attributo .rerank_score aggiunto a ciascun chunk.
        """
        if not chunks:
            return []

        self._load_model()

        # Costruisce coppie (query, testo_chunk)
        pairs = [(query, chunk.text) for chunk in chunks]

        logger.info(f"Reranking di {len(chunks)} chunk con {self.model_name}...")
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Arricchisce i chunk con il nuovo punteggio
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)
            chunk.metadata["rerank_score"] = float(score)
            chunk.metadata["rerank_model"] = self.model_name

        # Ordina per rerank_score decrescente
        reranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)

        # Applica filtro soglia
        if self.score_threshold is not None:
            before = len(reranked)
            reranked = [c for c in reranked if c.rerank_score >= self.score_threshold]
            logger.info(
                f"Filtro soglia {self.score_threshold}: "
                f"{before} → {len(reranked)} chunk"
            )

        # Applica top_n
        if self.top_n is not None:
            reranked = reranked[: self.top_n]

        logger.info(
            f"Reranking completato: {len(reranked)} chunk finali. "
            f"Top score: {reranked[0].rerank_score:.4f}" if reranked else "Nessun risultato."
        )
        return reranked