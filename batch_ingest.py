import os
import json
import glob
import uuid
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import qdrant_client

# --- CONFIGURATION ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "codex_history"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
ARCHIVE_PATH = os.path.expanduser("~/.gemini/tmp")

# --- HELPER FUNCTIONS ---


def get_qdrant_client():
    """Initializes and returns the Qdrant client."""
    return qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_embedding_model():
    """Initializes and returns the SentenceTransformer model."""
    print(f"⏳ Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Model loaded.")
    return model


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Splits a long text into smaller, overlapping chunks."""
    if not isinstance(text, str):
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]


# --- MAIN LOGIC ---


def process_conversation_file(
    file_path: str, model: SentenceTransformer, commit_id: str
) -> List[Dict]:
    """Reads a session JSON file, extracts data, and creates points for Qdrant."""
    points_to_upsert = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"⚠️  Could not read or parse {os.path.basename(file_path)}: {e}")
        return []

    # Handle different file formats
    messages = []
    if isinstance(data, list):
        # Check if it's logs.json format (has sessionId, messageId, type, message)
        if data and "sessionId" in data[0]:
            # logs.json format
            messages = data
        else:
            # checkpoint.json format (array of role/parts messages)
            messages = data
    elif isinstance(data, dict):
        # session.json format (has messages array)
        messages = data.get("messages", [])

    for entry in messages:
        # Extract text content based on format
        text_content = ""
        if "content" in entry:
            # session.json format
            text_content = entry.get("content", "")
        elif "message" in entry:
            # logs.json format
            text_content = entry.get("message", "")
        elif "parts" in entry:
            # checkpoint.json format
            parts = entry.get("parts", [])
            if parts and "text" in parts[0]:
                text_content = parts[0]["text"]

        if not text_content:
            continue

        chunks = chunk_text(text_content)
        for i, chunk in enumerate(chunks):
            vector = model.encode(chunk).tolist()

            # CORRECTED: Generate a new, valid UUID for each point.
            point_id = str(uuid.uuid4())

            payload = {
                "content": chunk,
                "timestamp": entry.get("timestamp"),
                "event_type": entry.get("type") or entry.get("role"),
                "original_message_id": entry.get("id") or entry.get("messageId"),
                "source_file": os.path.basename(file_path),
                "commit_id": commit_id,
                "chunk_index": i,
            }

            points_to_upsert.append(
                {"id": point_id, "vector": vector, "payload": payload}
            )

    return points_to_upsert


def main():
    """Main function to run the batch ingestion process."""
    client = get_qdrant_client()
    model = get_embedding_model()

    # 1. Create the collection if it doesn't exist
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print(f"ℹ️  Collection '{COLLECTION_NAME}' already exists.")
    except Exception:
        print(f"⏳ Collection '{COLLECTION_NAME}' not found. Creating it...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qdrant_client.http.models.VectorParams(
                size=VECTOR_SIZE, distance=qdrant_client.http.models.Distance.COSINE
            ),
        )
        print("✅ Collection created.")

    # 2. Find all JSON files that might contain conversation data
    print(f"🔍 Scanning for conversation files in {ARCHIVE_PATH}...")
    # Include session files, checkpoint files, and logs.json files
    conversation_files = []
    conversation_files.extend(
        glob.glob(os.path.join(ARCHIVE_PATH, "**", "chats", "*.json"), recursive=True)
    )
    conversation_files.extend(
        glob.glob(os.path.join(ARCHIVE_PATH, "**", "checkpoint*.json"), recursive=True)
    )
    conversation_files.extend(
        glob.glob(os.path.join(ARCHIVE_PATH, "**", "logs.json"), recursive=True)
    )

    if not conversation_files:
        print("❌ No conversation files found. Please check the ARCHIVE_PATH.")
        return

    print(f"✅ Found {len(conversation_files)} conversation files to process.")

    total_points = 0
    # 3. Process each file and upsert to Qdrant
    for file_path in conversation_files:
        # Extract commit_id from path - handle different directory structures
        path_parts = file_path.split(os.sep)
        if "chats" in path_parts:
            chats_index = path_parts.index("chats")
            if chats_index > 0:
                commit_id = path_parts[chats_index - 1]
            else:
                commit_id = "unknown"
        else:
            commit_id = "unknown"

        print(
            f"\n--- Processing: {os.path.basename(file_path)} (from commit {commit_id}) ---"
        )

        points = process_conversation_file(file_path, model, commit_id)

        if not points:
            print("No valid entries found in this file.")
            continue

        # Upsert points in batches
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                qdrant_client.http.models.PointStruct(
                    id=p["id"], vector=p["vector"], payload=p["payload"]
                )
                for p in points
            ],
            wait=True,
        )
        total_points += len(points)
        print(f"Upserted {len(points)} points to Qdrant.")

    print(f"\n\n🎉🎉🎉 Grand Forging Complete! 🎉🎉🎉")
    print(f"Total points newly added to the Codex: {total_points}")
    # Verify final count
    count_result = client.count(collection_name=COLLECTION_NAME, exact=True)
    print(f"Final verification count from Qdrant: {count_result.count}")


if __name__ == "__main__":
    main()
