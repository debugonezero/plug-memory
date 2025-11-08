"""
Hybrid Memory System - Combines vector search with LangChain reasoning
"""

from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Qdrant
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
import qdrant_client
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class HybridMemorySystem:
    """Hybrid memory system combining fast vector search with reasoning capabilities."""

    def __init__(self, qdrant_client, collection_name: str = "codex_history"):
        self.collection_name = collection_name
        self.qdrant_client = qdrant_client

        # Initialize embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize LangChain vector store
        self.vector_store = Qdrant(
            client=qdrant_client,
            collection_name=collection_name,
            embeddings=self.embedding_model.encode,
        )

        # Initialize conversational memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5,  # Keep last 5 conversation turns
        )

    def fast_query(self, query: str, limit: int = 3) -> Dict[str, Any]:
        """Fast vector search for precise queries."""
        try:
            docs = self.vector_store.similarity_search(query, k=limit)
            results = []
            for doc in docs:
                results.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, "score", None),
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

    def reasoning_query(self, query: str, llm=None) -> Dict[str, Any]:
        """Complex reasoning query using conversational retrieval."""
        if llm is None:
            return {
                "error": "LLM required for reasoning queries",
                "query": query,
                "suggestion": "Provide an LLM instance for conversational reasoning",
            }

        try:
            # Create conversational chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
                memory=self.memory,
                verbose=True,
            )

            # Execute reasoning query
            result = qa_chain({"question": query})

            return {
                "query": query,
                "answer": result.get("answer", ""),
                "source_documents": [
                    {"content": doc.page_content, "metadata": doc.metadata}
                    for doc in result.get("source_documents", [])
                ],
                "method": "conversational_reasoning",
                "chat_history": result.get("chat_history", []),
            }
        except Exception as e:
            logger.error(f"Reasoning query error: {e}")
            return {"error": str(e), "query": query}

    def hybrid_query(
        self, query: str, llm=None, use_reasoning: bool = None
    ) -> Dict[str, Any]:
        """Intelligent query routing - chooses between fast search and reasoning."""
        if use_reasoning is None:
            # Auto-detect query complexity
            use_reasoning = self._detect_complexity(query)

        if use_reasoning and llm:
            return self.reasoning_query(query, llm)
        else:
            return self.fast_query(query)

    def _detect_complexity(self, query: str) -> bool:
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

    def add_memory(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add new content to memory."""
        try:
            from langchain.schema import Document

            doc = Document(page_content=content, metadata=metadata or {})

            self.vector_store.add_documents([doc])
            return True
        except Exception as e:
            logger.error(f"Add memory error: {e}")
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            # Get vector store stats
            count = self.qdrant_client.count(
                collection_name=self.collection_name, exact=True
            ).count

            return {
                "total_memories": count,
                "collection_name": self.collection_name,
                "memory_type": "hybrid_vector_langchain",
                "conversation_turns": len(self.memory.chat_memory.messages)
                if hasattr(self.memory, "chat_memory")
                else 0,
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"error": str(e)}


# Convenience functions
def create_hybrid_memory(
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "codex_history",
) -> HybridMemorySystem:
    """Create a hybrid memory system instance."""
    client = qdrant_client.QdrantClient(host=qdrant_host, port=qdrant_port)
    return HybridMemorySystem(client, collection_name)
