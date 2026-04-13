"""
main.py — Pipeline RAG completa: Ingestion + Query

Avvio:
    python main.py

Per la generazione gratuita con Ollama:
    1. Installa Ollama: https://ollama.com/download
    2. Scarica un modello: ollama pull llama3.2
    3. Avvia il server: ollama serve
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))

from Ingestion import IngestionPipeline
from Query import QueryPipeline, _load_bm25_corpus, HybridRetriever
from evaluation import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

def run_ingestion(data_path: str = "./data"):
    """Step 1 — Indicizza i documenti (eseguire solo la prima volta o dopo aggiornamenti)."""
    pipeline = IngestionPipeline(data_path=data_path)
    result = pipeline.run()
    print(f"\nIngestion completata: {result}\n")


def run_query_loop():
    """Step 2 — Loop interattivo di domande e risposte."""

    # ── Configura la pipeline ──────────────────────────────────────────────
    pipeline = QueryPipeline.from_config(
        collection_name="rag_collection",
        persist_directory="./chroma_db",

        # Embedding (stesso usato in ingestion!)
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        # Per italiano: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        # Cross-Encoder per reranking
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        # Per italiano: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

        # Generazione: Ollama (gratuito) o Anthropic (a pagamento)
        ollama_model="llama3.2",   # scarica con: ollama pull llama3.2
        use_anthropic=False,       # True per usare ANTHROPIC_API_KEY

        top_k_retrieval=20,        # candidati da vector+BM25
        top_n_rerank=5,            # chunk finali dopo cross-encoder
    )
    # ──────────────────────────────────────────────────────────────────────

    print("\n RAG Pipeline pronta. Digita 'exit' per uscire.\n")
    while True:
        try:
            question = input("Domanda: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nUscita.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        answer = pipeline.ask(question, verbose=True)
        print(f"\n Risposta:\n{answer}\n")
        print("─" * 60)

def run_evaluation():
    logger.info("Avvio valutazione del modello...")
    
    pipeline = IngestionPipeline(
        data_path=DATA_PATH
    )
    
    test_queries = [
        {
            "query": "Cos' un tranformers?", 
            "expected_chunk_id": "doc_1_chunk_0",
            "query_tokens": ["procedura", "per", "x"]
        }
    ]
    
    corpus_texts, corpus_ids = _load_bm25_corpus(pipeline.indexer)
     
    retriever = HybridRetriever(
        indexer=pipeline.indexer,
        embedder=pipeline.embedding_model,
        bm25_corpus=corpus_texts,
        bm25_corpus_ids=corpus_ids,
        top_k=10
    )
    
    evaluator = ModelEvaluator(pipeline=pipeline, retriever=retriever)
    
    evaluator.evaluate()
    
    metrics = evaluator.run_full_benchmark(
        data_path=DATA_PATH, 
        test_queries=test_queries
    )
    
    print("\n" + "="*30)
    print("RISULTATI BENCHMARK")
    print("="*30)
    print(f"Documenti Totali:    {metrics.total_documents}")
    print(f"Tempo Inserimento:   {metrics.insert_time:.2f}s")
    print(f"Tempo Ricerca Medio: {metrics.search_time:.4f}s")
    print(f"Accuratezza Top-1:   {metrics.accuracy_top1:.2%}")
    print(f"Accuratezza Top-5:   {metrics.accuracy_top5:.2%}")
    print("="*30)
    
if __name__ == "__main__":
    DATA_PATH = "./data"  

    # STEP 1 (solo la prima volta, o dopo aver aggiunto nuovi documenti)
    run_ingestion(DATA_PATH)

    # STEP 2 (sempre)
    run_query_loop()
    
    # STEP 3 (evaluation)
    run_evaluation()

