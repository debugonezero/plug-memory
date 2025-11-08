"""
Simplified Hybrid Memory System - Python 3.14 compatible
"""

from typing import List, Dict, Any, Optional
import qdrant_client
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class SimpleHybridMemory:
    """Simplified hybrid memory system for Python 3.14 compatibility."""

    def __init__(self, qdrant_client, collection_name: str = "codex_history"):
        self.collection_name = collection_name
        self.qdrant_client = qdrant_client
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def fast_query(self, query: str, limit: int = 3) -> Dict[str, Any]:
        """Fast vector search for precise queries."""
        try:
            # Encode query
            query_vector = self.embedding_model.encode(query).tolist()

            # Search Qdrant
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
            )

            results = []
            for hit in search_result:
                results.append(
                    {
                        "content": hit.payload.get("content", ""),
                        "timestamp": hit.payload.get("timestamp"),
                        "source_file": hit.payload.get("source_file"),
                        "score": hit.score,
                    }
                )

            return {
                "query": query,
                "results": results,
                "method": "fast_vector_search",
                "count": len(results),
            }
        except Exception as e:
            logger.error(f"Fast query error: {e}")
            return {"error": str(e), "query": query}

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            count = self.qdrant_client.count(
                collection_name=self.collection_name, exact=True
            ).count

            return {
                "total_memories": count,
                "collection_name": self.collection_name,
                "memory_type": "vector_database",
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"error": str(e)}

    def detect_query_complexity(self, query: str) -> bool:
        """Detect if query needs complex reasoning."""
        complexity_indicators = [
            "why",
            "how",
            "explain",
            "analyze",
            "compare",
            "relationship",
            "context",
            "reasoning",
            "inference",
            "what if",
            "suppose",
            "imagine",
            "consider",
        ]

        query_lower = query.lower()
        return any(indicator in query_lower for indicator in complexity_indicators)


def create_simple_hybrid_memory(
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "codex_history",
) -> SimpleHybridMemory:
    """Create a simple hybrid memory system instance."""
    client = qdrant_client.QdrantClient(host=qdrant_host, port=qdrant_port)
    return SimpleHybridMemory(client, collection_name)
