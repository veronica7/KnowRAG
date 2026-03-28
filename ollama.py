import logging
from typing import List

from Query import RetrievedChunk

logger = logging.getLogger(__name__)

class OllamaGenerator:
    """
    Generazione locale e gratuita tramite Ollama.
    Richiede: pip install requests  +  ollama serve  (in background)
    Modelli leggeri consigliati: llama3.2, mistral, phi3
    """

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, query: str, context_chunks: List[RetrievedChunk]) -> str:
        try:
            import requests
        except ImportError:
            raise ImportError("Installa requests: pip install requests")

        context = self._build_context(context_chunks)
        prompt = self._build_prompt(query, context)

        logger.info(f"Generazione risposta con Ollama ({self.model})...")
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return (
                "[Errore] Ollama non raggiungibile. "
                f"Assicurati che 'ollama serve' sia attivo.\n"
                f"Scarica il modello con: ollama pull {self.model}"
            )
        except Exception as e:
            logger.error(f"Errore Ollama: {e}")
            return f"[Errore generazione] {e}"

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("filename", chunk.metadata.get("source", "sconosciuta"))
            parts.append(f"[Fonte {i} — {source}]\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            "Sei un assistente preciso e utile. "
            "Rispondi alla domanda basandoti ESCLUSIVAMENTE sul contesto fornito.\n"
            "Se il contesto non contiene informazioni sufficienti, dillo esplicitamente.\n\n"
            f"CONTESTO:\n{context}\n\n"
            f"DOMANDA:\n{query}\n\n"
            "RISPOSTA:"
        )