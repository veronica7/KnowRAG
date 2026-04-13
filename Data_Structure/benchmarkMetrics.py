from dataclasses import dataclass

@dataclass
class BenchmarkMetrics:
    """Metriche di performance"""
    insert_time: float
    search_time: float
    accuracy_top1: float
    accuracy_top5: float
    total_documents: int