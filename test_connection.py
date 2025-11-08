
import qdrant_client
from sentence_transformers import SentenceTransformer

try:
    # 1. Initialize the Qdrant client
    # This will attempt to connect to localhost:6333, where your Docker container is running.
    client = qdrant_client.QdrantClient(host="localhost", port=6333)
    print("✅ Successfully connected to Qdrant client.")

    # 2. Initialize the embedding model
    # This model will convert our text into vectors. It may be downloaded on the first run.
    print("⏳ Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ SentenceTransformer model loaded.")

    # 3. Define our test collection name
    collection_name = "codex_test_collection"

    # Let's try to delete the collection first in case it exists from a previous run
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"🧹 Collection '{collection_name}' found and deleted for a clean run.")
    except Exception:
        print(f"ℹ️  Collection '{collection_name}' did not exist, which is fine.")

    # 4. Create the collection in Qdrant
    # We need to tell Qdrant the size of the vectors our model produces.
    # 'all-MiniLM-L6-v2' produces vectors of size 384.
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qdrant_client.http.models.VectorParams(size=384, distance=qdrant_client.http.models.Distance.COSINE)
    )
    print(f"✅ Collection '{collection_name}' created successfully.")

    # 5. Create a test document and its vector embedding
    documents = [
        {"text": "This is the first memory of the Codex.", "id": 1}
    ]
    vectors = model.encode([doc["text"] for doc in documents])
    print("✅ Test document encoded into a vector.")

    # 6. Upsert the document vector into our collection
    client.upsert(
        collection_name=collection_name,
        points=[
            qdrant_client.http.models.PointStruct(
                id=doc["id"],
                vector=vectors[i].tolist(),
                payload={"text": doc["text"]}
            ) for i, doc in enumerate(documents)
        ],
        wait=True
    )
    print("✅ Document successfully added to the collection.")

    # 7. Verify the document was added
    count_result = client.count(collection_name=collection_name, exact=True)
    print(f"📊 Number of points in collection: {count_result.count}")

    if count_result.count == 1:
        print("\n🎉 SUCCESS! The Codex Engine is alive! The connection to Qdrant is working and we have successfully stored our first memory! 🎉")
    else:
        print("\n❌ FAILURE: Something went wrong. The document was not stored correctly.")

except Exception as e:
    print(f"\n❌ An error occurred: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure your Qdrant Docker container is running.")
    print("2. Verify you can access Qdrant at http://localhost:6333 in your browser.")
    print("3. Check for any errors in the Docker container logs.")
