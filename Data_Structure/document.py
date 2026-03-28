from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Document:
    """Rappresenta un documento caricato con contenuto e metadati."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
 
    def __repr__(self):
        preview = self.content[:80].replace("\n", " ")
        return f"Document(source={self.metadata.get('source', 'unknown')}, preview='{preview}...')"