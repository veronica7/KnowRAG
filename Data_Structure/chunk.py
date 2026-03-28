from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class Chunk:
    """Rappresenta un chunk di testo con metadati arricchiti."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
 
    def __repr__(self):
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(chunk_id={self.metadata.get('chunk_id', '?')}, preview='{preview}...')"