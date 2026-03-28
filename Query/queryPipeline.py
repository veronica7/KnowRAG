from Query  import QueryProcessor, HybridRetriever, CrossEncoderReranker, RetrievedChunk, _load_bm25_corpus, _print_chunks
from ollama import OllamaGenerator
from anthopic import AnthropicGenerator
from Ingestion import EmbeddingModel, ChromaIndexer

from typing import List
import logging

logger = logging.getLogger(__name__)


class QueryPipeline:
    """
    Orchestratore della fase di query:
      domanda → preprocessing → retrieval ibrido → reranking → generazione

    Utilizzo rapido:
        pipeline = QueryPipeline.from_config()
        answer = pipeline.ask("Come funziona il processo X?")
        print(answer)
    """

    def __init__(
        self,
        query_processor: QueryProcessor,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        generator,
        top_k_retrieval: int = 20,
        top_n_rerank: int = 5,
    ):
        self.query_processor = query_processor
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.top_k_retrieval = top_k_retrieval
        self.top_n_rerank = top_n_rerank

    @classmethod
    def from_config(
        cls,
        collection_name: str = "rag_collection",
        persist_directory: str = "./chroma_db",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ollama_model: str = "llama3.2",
        top_k_retrieval: int = 20,
        top_n_rerank: int = 5,
        use_anthropic: bool = False,
    ) -> "QueryPipeline":
        """Factory: costruisce la QueryPipeline con una sola chiamata."""
        logger.info("Inizializzazione QueryPipeline...")

        embedder = EmbeddingModel(embedding_model_name)
        indexer = ChromaIndexer(collection_name, persist_directory)

        corpus_texts, corpus_ids = _load_bm25_corpus(indexer)

        retriever = HybridRetriever(
            indexer=indexer,
            embedder=embedder,
            bm25_corpus=corpus_texts,
            bm25_corpus_ids=corpus_ids,
            top_k=top_k_retrieval,
        )

        reranker = CrossEncoderReranker(
            model_name=reranker_model_name,
            top_n=top_n_rerank,
        )

        generator = (
            AnthropicGenerator() if use_anthropic
            else OllamaGenerator(model=ollama_model)
        )

        return cls(
            query_processor=QueryProcessor(),
            retriever=retriever,
            reranker=reranker,
            generator=generator,
            top_k_retrieval=top_k_retrieval,
            top_n_rerank=top_n_rerank,
        )

    def ask(self, question: str, verbose: bool = False) -> str:
        """Pipeline completa: domanda → risposta."""
        logger.info(f"\n{'='*50}\nDOMANDA: {question}\n{'='*50}")

        # 1. Preprocessing
        processed = self.query_processor.process(question)
        bm25_tokens = self.query_processor.tokenize_for_bm25(processed)
        logger.info(f"[1/4] Query processata: '{processed}'")

        # 2. Retrieval ibrido (vector + BM25 con RRF)
        candidates = self.retriever.retrieve(
            query=processed,
            query_tokens=bm25_tokens,
            n_candidates=self.top_k_retrieval,
        )
        logger.info(f"[2/4] Candidati recuperati: {len(candidates)}")

        if not candidates:
            return "Nessun documento rilevante trovato nel corpus per questa domanda."

        # 3. Reranking con Cross-Encoder
        reranked = self.reranker.rerank(processed, candidates)
        logger.info(f"[3/4] Chunk dopo reranking: {len(reranked)}")

        if verbose:
            _print_chunks(reranked)

        # 4. Generazione risposta
        answer = self.generator.generate(processed, reranked)
        logger.info("[4/4] Risposta generata.")
        return answer


    

