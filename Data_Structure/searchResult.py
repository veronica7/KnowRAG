from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Risultato di una ricerca"""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]