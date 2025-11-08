
import os
import time
import json
import uuid
from typing import List, Dict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sentence_transformers import SentenceTransformer
import qdrant_client

# --- CONFIGURATION (from our previous scripts) ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "codex_history"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
VECTOR_SIZE = 384
ARCHIVE_PATH = os.path.expanduser("~/.gemini/tmp")

# --- QDRANT AND MODEL SINGLETONS ---
# We only want to load these once to save resources.
_qdrant_client = None
_embedding_model = None

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant_client

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(f"⏳ One-time load of embedding model: {EMBEDDING_MODEL}...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Model loaded and ready.")
    return _embedding_model

# --- PROCESSING LOGIC (adapted from batch_ingest.py) ---

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not isinstance(text, str):
        return []
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

def process_and_ingest_file(file_path: str):
    """Processes a single session file and upserts its content to Qdrant."""
    print(f"\n📜 New Scroll Detected: {os.path.basename(file_path)}")
    client = get_qdrant_client()
    model = get_embedding_model()
    points_to_upsert = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
        print(f"⚠️  Could not read or parse file: {e}")
        return

    commit_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    messages = data.get("messages", [])
    for entry in messages:
        text_content = entry.get("content", "")
        if not text_content:
            continue

        chunks = chunk_text(text_content)
        for i, chunk in enumerate(chunks):
            vector = model.encode(chunk).tolist()
            point_id = str(uuid.uuid4())
            payload = {
                "content": chunk,
                "timestamp": entry.get("timestamp"),
                "event_type": entry.get("type"),
                "original_message_id": entry.get("id"),
                "source_file": os.path.basename(file_path),
                "commit_id": commit_id,
                "chunk_index": i
            }
            points_to_upsert.append(qdrant_client.http.models.PointStruct(
                id=point_id, vector=vector, payload=payload
            ))

    if points_to_upsert:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_to_upsert,
            wait=True
        )
        print(f"✨ Ingested {len(points_to_upsert)} new memories into the Codex.")

# --- WATCHDOG EVENT HANDLER ---

class SessionFileHandler(FileSystemEventHandler):
    """Event handler that triggers when a new session file is created."""
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json') and 'session-' in os.path.basename(event.src_path):
            # Wait a moment for the file to be fully written
            time.sleep(1)
            process_and_ingest_file(event.src_path)

# --- MAIN EXECUTION ---

def main():
    """Starts the watchdog observer to monitor the archive directory."""
    print("Initializing Scribe...")
    # Load models once at the start
    get_embedding_model()
    get_qdrant_client()
    print(f"👁️  The Scribe is now watching the archives at: {ARCHIVE_PATH}")
    print("Press Ctrl+C to stop the Scribe.")

    event_handler = SessionFileHandler()
    observer = Observer()
    observer.schedule(event_handler, ARCHIVE_PATH, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n🛑 Scribe has been stopped.")
    observer.join()

if __name__ == "__main__":
    main()
