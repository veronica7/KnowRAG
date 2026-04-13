import logging
import time
from typing import List, Dict, Any
from Data_Structure import BenchmarkMetrics

logger = logging.getLogger(__name__)
class Evaluation:
    """
    Classe per la valutazione delle performance del sistema KnowRAG.
    Supporta la valutazione della IngestionPipeline e del HybridRetriever.
    """
    def __init__(self, pipeline: Any = None, retriever: Any = None):
        """
        Inizializza l'evaluator.
        :param pipeline: Istanza di IngestionPipeline
        :param retriever: Istanza di HybridRetriever
        """
        self.pipeline = pipeline
        self.retriever = retriever

    def evaluate_ingestion(self, data_path: str) -> float:
        """
        Valuta il tempo di ingestion (caricamento, chunking, embedding, indexing).
        Restituisce il tempo totale in secondi.
        """
        if not self.pipeline:
            logger.error("IngestionPipeline non fornita.")
            return 0.0
        
        logger.info(f"Inizio benchmark ingestion da: {data_path}")
        start_time = time.time()
        self.pipeline.run()
        duration = time.time() - start_time
        logger.info(f"Ingestion completata in {duration:.4f}s")
        return duration

    def evaluate_retrieval(self, test_queries: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, float]:
        """
        Valuta le performance del retriever (tempo e accuratezza).
        test_queries: lista di {"query": "...", "expected_chunk_id": "...", "query_tokens": [...]}
        """
        if not self.retriever:
            logger.error("Retriever non fornito.")
            return {"avg_time": 0.0, "top1": 0.0, "top5": 0.0}

        search_times = []
        top1_hits = 0
        top5_hits = 0

        logger.info(f"Inizio benchmark retrieval su {len(test_queries)} query.")

        for item in test_queries:
            query_text = item["query"]
            expected_id = item.get("expected_chunk_id")
            query_tokens = item.get("query_tokens", []) # Per BM25 se necessario
            
            start_search = time.time()
            # Esegue il retrieval ibrido
            candidates = self.retriever.retrieve(
                query=query_text,
                query_tokens=query_tokens,
                n_candidates=top_k
            )
            search_times.append(time.time() - start_search)

            # Calcolo accuratezza basato sugli ID dei chunk restituiti
            if candidates and expected_id:
                # Assumiamo che candidate abbia un attributo 'id' o simile
                retrieved_ids = [getattr(c, 'id', None) or c.metadata.get('id') for c in candidates]
                
                if expected_id == retrieved_ids[0]:
                    top1_hits += 1
                if expected_id in retrieved_ids[:5]:
                    top5_hits += 1

        avg_time = sum(search_times) / len(search_times) if search_times else 0.0
        accuracy_top1 = top1_hits / len(test_queries) if test_queries else 0.0
        accuracy_top5 = top5_hits / len(test_queries) if test_queries else 0.0

        logger.info(f"Retrieval benchmark completato. Avg Time: {avg_time:.4f}s, Top-1: {accuracy_top1:.2%}")
        return {
            "avg_time": avg_time,
            "top1": accuracy_top1,
            "top5": accuracy_top5
        }

    def run_full_benchmark(self, data_path: str, test_queries: List[Dict[str, Any]]) -> BenchmarkMetrics:
        """
        Esegue la valutazione completa e restituisce un oggetto BenchmarkMetrics.
        """
        # 1. Ingestion
        insert_time = self.evaluate_ingestion(data_path)
        
        # 2. Retrieval
        retrieval_results = self.evaluate_retrieval(test_queries)
        
        # 3. Conteggio documenti totali (da indexer se disponibile)
        total_docs = 0
        if self.pipeline and hasattr(self.pipeline, 'indexer'):
            info = self.pipeline.indexer.get_collection_info()
            total_docs = info.get("total_documents", 0)

        return BenchmarkMetrics(
            insert_time=insert_time,
            search_time=retrieval_results["avg_time"],
            accuracy_top1=retrieval_results["top1"],
            accuracy_top5=retrieval_results["top5"],
            total_documents=total_docs
        )