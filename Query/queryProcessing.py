import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Preprocessa e arricchisce la query dell'utente prima del retrieval.

    Step eseguiti:
      1. Pulizia e normalizzazione del testo
      2. Tokenizzazione per BM25 (con stopwords opzionali)
      3. Generazione varianti query per multi-query retrieval (no LLM)
    """

    _STOPWORDS_IT = {
        "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
        "e", "è", "di", "a", "da", "in", "con", "su", "per",
        "tra", "fra", "che", "chi", "cui", "non", "si", "mi",
        "ti", "ci", "vi", "ne", "del", "della", "dei", "degli",
        "delle", "al", "alla", "ai", "agli", "alle", "nel", "nella",
        "nei", "negli", "nelle", "sul", "sulla", "sui", "sugli",
        "sulle", "dal", "dalla", "dai", "dagli", "dalle", "come",
        "quando", "dove", "perché", "cosa", "questo", "questa",
        "questi", "queste", "quello", "quella", "quelli", "quelle",
    }

    def __init__(
        self,
        remove_stopwords: bool = False,
        min_token_length: int = 2,
    ):
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length

    # ─────────────────────────────────────────────
    # API PUBBLICA
    # ─────────────────────────────────────────────

    def process(self, query: str) -> str:
        """Pulisce e normalizza la query. Restituisce la query processata."""
        query = self._clean(query)
        if not query:
            raise ValueError("La query è vuota dopo la pulizia.")
        logger.debug(f"Query processata: '{query}'")
        return query

    def tokenize_for_bm25(self, query: str) -> List[str]:
        """
        Tokenizza la query per BM25.
        Restituisce lista di token puliti, senza stopwords se configurato.
        """
        query = self._clean(query)
        tokens = re.findall(r'\b\w+\b', query.lower())
        tokens = [t for t in tokens if len(t) >= self.min_token_length]
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._STOPWORDS_IT]
        logger.debug(f"Token BM25: {tokens}")
        return tokens

    def expand_query(self, query: str, n_variants: int = 2) -> List[str]:
        """
        Genera varianti della query per il multi-query retrieval (euristico, no LLM).
        Restituisce lista con [query_originale, variante_1, variante_2, ...].
        """
        processed = self.process(query)
        variants = [processed]

        # Variante 1: rimuove pronomi interrogativi iniziali
        affirmative = self._to_affirmative(processed)
        if affirmative and affirmative != processed:
            variants.append(affirmative)

        # Variante 2: solo keyword content (senza stopwords e parole corte)
        keywords = self._extract_keywords(processed)
        if keywords and keywords != processed and keywords not in variants:
            variants.append(keywords)

        # Variante 3: inversione ordine keyword
        if len(variants) <= n_variants:
            inverted = self._invert_keywords(processed)
            if inverted and inverted not in variants:
                variants.append(inverted)

        return variants[:n_variants + 1]

    # ─────────────────────────────────────────────
    # METODI PRIVATI
    # ─────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\'\-\.\,\?\!]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _to_affirmative(self, query: str) -> str:
        pattern = r'^(come|cosa|chi|dove|quando|perché|perche|qual è|quale|quanto|quanti)\s+'
        return re.sub(pattern, '', query, flags=re.IGNORECASE).strip()

    def _extract_keywords(self, query: str) -> str:
        tokens = re.findall(r'\b\w+\b', query.lower())
        kw = [t for t in tokens if len(t) > 3 and t not in self._STOPWORDS_IT]
        return ' '.join(kw) if kw else ''

    def _invert_keywords(self, query: str) -> Optional[str]:
        kw = self._extract_keywords(query).split()
        return ' '.join(reversed(kw)) if len(kw) >= 2 else None