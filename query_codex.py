
import sys
from sentence_transformers import SentenceTransformer
import qdrant_client

# --- CONFIGURATION ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "codex_history"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# --- MAIN LOGIC ---

def search_codex(query: str):
    """Searches the Codex for a given query and prints the top results."""
    if not query:
        print("❌ Please provide a query.")
        return

    try:
        # 1. Initialize clients and models
        client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print(f"⏳ Loading embedding model: {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Connection and models are ready.")

        # 2. Convert the query to a vector
        print(f"\n🔍 Searching for: '{query}'")
        query_vector = model.encode(query).tolist()

        # 3. Perform the search
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3,  # Return the top 3 most similar results
            with_payload=True  # Include the payload in the result
        )

        # 4. Print the results
        print("\n--- Top 3 Memories Found ---")
        if not search_result:
            print("No memories found matching your query.")
            return

        for i, result in enumerate(search_result):
            payload = result.payload
            print(f"\nResult {i+1} (Score: {result.score:.4f}):")
            print(f"  Timestamp: {payload.get('timestamp')}")
            print(f"  Source File: {payload.get('source_file')}")
            print(f"  Content: \"...{payload.get('content')}...\"")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your Qdrant Docker container is running.")
        print("2. Ensure the 'codex_history' collection exists and is populated.")

if __name__ == "__main__":
    # Check if a query was provided as a command-line argument
    if len(sys.argv) > 1:
        # Join all arguments to form the query string
        user_query = " ".join(sys.argv[1:])
        search_codex(user_query)
    else:
        print("Usage: python query_codex.py <your search query>")
