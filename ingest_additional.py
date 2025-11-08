#!/usr/bin/env python3
"""
Additional ingestion script for checkpoint and logs files
"""

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


def get_qdrant_client():
    return qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_embedding_model():
    print(f"⏳ Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("✅ Model loaded.")
    return model


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not isinstance(text, str):
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]


def process_additional_files(model: SentenceTransformer) -> List[Dict]:
    """Process checkpoint and logs files that weren't included in original ingestion."""
    points_to_upsert = []

    # Find checkpoint and logs files
    checkpoint_files = glob.glob(
        os.path.join(ARCHIVE_PATH, "**", "checkpoint*.json"), recursive=True
    )
    logs_files = glob.glob(
        os.path.join(ARCHIVE_PATH, "**", "logs.json"), recursive=True
    )

    all_files = checkpoint_files + logs_files
    print(f"Found {len(all_files)} additional files to process")

    for file_path in all_files:
        print(f"Processing {os.path.basename(file_path)}...")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Handle different file formats
        messages = []
        if isinstance(data, list):
            if data and "sessionId" in data[0]:
                # logs.json format
                messages = data
            else:
                # checkpoint.json format
                messages = data

        for entry in messages:
            # Extract text content
            text_content = ""
            if "message" in entry:
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
                point_id = str(uuid.uuid4())

                # Extract commit_id from path
                path_parts = file_path.split(os.sep)
                commit_id = "unknown"
                if ".gemini" in path_parts and "tmp" in path_parts:
                    gemini_index = path_parts.index("tmp")
                    if gemini_index + 1 < len(path_parts):
                        commit_id = path_parts[gemini_index + 1]

                payload = {
                    "content": chunk,
                    "timestamp": entry.get("timestamp"),
                    "event_type": entry.get("type") or entry.get("role"),
                    "original_message_id": entry.get("messageId") or str(uuid.uuid4()),
                    "source_file": os.path.basename(file_path),
                    "commit_id": commit_id,
                    "chunk_index": i,
                }

                points_to_upsert.append(
                    {"id": point_id, "vector": vector, "payload": payload}
                )

    return points_to_upsert


def main():
    client = get_qdrant_client()
    model = get_embedding_model()

    # Check if collection exists
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print(f"✅ Collection '{COLLECTION_NAME}' exists.")
    except Exception:
        print(
            f"❌ Collection '{COLLECTION_NAME}' not found. Run batch_ingest.py first."
        )
        return

    # Process additional files
    points = process_additional_files(model)

    if not points:
        print("No additional points to add.")
        return

    print(f"Adding {len(points)} points to collection...")

    # Upsert in batches
    batch_size = 50  # Smaller batches
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    qdrant_client.http.models.PointStruct(
                        id=p["id"], vector=p["vector"], payload=p["payload"]
                    )
                    for p in batch
                ],
                wait=True,
            )
            print(
                f"Upserted batch {i // batch_size + 1}/{(len(points) + batch_size - 1) // batch_size}"
            )
        except Exception as e:
            print(f"Error upserting batch {i // batch_size + 1}: {e}")
            continue

    final_count = client.count(collection_name=COLLECTION_NAME, exact=True)
    print(f"✅ Final collection count: {final_count.count}")


if __name__ == "__main__":
    main()
