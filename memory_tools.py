import qdrant_client
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "codex_history"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Reusable Components ---
# We cache these so they don't reload on every function call within the same process.
_model = None
_client = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_client():
    global _client
    if _client is None:
        _client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


# --- THE CUSTOM TOOL FUNCTION ---


def query_my_memory(query: str) -> str:
    """
    This is the function that will be registered as a custom tool.
    It takes a string query, searches the Qdrant vector database (the Codex),
    and returns the top 3 most relevant memories as a formatted string.
    """
    try:
        if not query or not query.strip():
            return "Error: No query provided."

        client = _get_client()
        model = _get_model()

        query_vector = model.encode(query).tolist()

        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3,
            with_payload=True,
        )

        # Format the results into a single string to be returned to the AI.
        response_string = ""
        if not search_result:
            return "I found no memories matching that query."

        response_string += "I found the following relevant memories:\n\n"
        for i, result in enumerate(search_result):
            payload = result.payload if result.payload else {}
            response_string += f"--- Memory {i + 1} (Score: {result.score:.4f}) ---\n"
            response_string += f"Timestamp: {payload.get('timestamp')}\n"  # type: ignore
            response_string += f"Source: {payload.get('source_file')}\n"  # type: ignore
            response_string += f"Content: {payload.get('content')}\n\n"  # type: ignore

        return response_string

    except Exception as e:
        return f"An error occurred while querying my memory: {e}"


# This part allows you to test the function directly from the command line
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
        print("--- Testing query_my_memory function ---")
        memories = query_my_memory(test_query)
        print(memories)
    else:
        print("Usage: python codex_tools.py <your test query>")
