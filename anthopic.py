import os
import logging
from typing import List
from ollama import OllamaGenerator

from Query import RetrievedChunk

logger = logging.getLogger(__name__)

class AnthropicGenerator:
    """
    Generazione con Claude via API Anthropic.
    Richiede: pip install anthropic
    Imposta: export ANTHROPIC_API_KEY=sk-ant-...
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, query: str, context_chunks: List[RetrievedChunk]) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Installa anthropic: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise EnvironmentError(
                "Variabile ANTHROPIC_API_KEY non trovata.\n"
                "Imposta: export ANTHROPIC_API_KEY=sk-ant-..."
            )

        # Riusa _build_context di OllamaGenerator
        context = OllamaGenerator(self.model)._build_context(context_chunks)

        client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Generazione risposta con Anthropic ({self.model})...")
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=(
                "Sei un assistente preciso. "
                "Rispondi sempre in italiano, in modo chiaro e conciso, "
                "basandoti solo sul contesto fornito."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"CONTESTO:\n{context}\n\n"
                    f"DOMANDA:\n{query}"
                ),
            }],
        )
        return message.content[0].text.strip()