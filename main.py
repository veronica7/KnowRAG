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
from typing import List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from Ingestion import IngestionPipeline, EmbeddingModel, ChromaIndexer
from Query import QueryProcessor, HybridRetriever, RetrievedChunk, CrossEncoderReranker


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# GENERATORI DI RISPOSTA
# ══════════════════════════════════════════════════════════════════════════════

from ollama import OllamaGenerator
from anthopic import AnthropicGenerator

# ══════════════════════════════════════════════════════════════════════════════
# QUERY PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

from Query import QueryPipeline



# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

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
        use_anthropic=False,       # ← True per usare ANTHROPIC_API_KEY

        top_k_retrieval=20,        # candidati da vector+BM25
        top_n_rerank=5,            # chunk finali dopo cross-encoder
    )
    # ──────────────────────────────────────────────────────────────────────

    print("\n🔍 RAG Pipeline pronta. Digita 'exit' per uscire.\n")
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
        print(f"\n📝 Risposta:\n{answer}\n")
        print("─" * 60)


if __name__ == "__main__":
    DATA_PATH = "./data"   # ← cartella con i tuoi documenti

    # ── STEP 1 (solo la prima volta, o dopo aver aggiunto nuovi documenti)
    run_ingestion(DATA_PATH)

    # ── STEP 2 (sempre)
    run_query_loop()

