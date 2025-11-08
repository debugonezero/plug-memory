import os
import json
import glob
from transformers import AutoTokenizer

# --- CONFIGURATION ---
# CORRECTED: The root of the entire archive.
ARCHIVE_PATH = os.path.expanduser("~/.gemini")
OUTPUT_FILE = "tokenized_codex.json"
TOKENIZER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# --- MAIN LOGIC ---

def main():
    """Main function to tokenize all session logs from the entire .gemini directory."""
    print(f"⏳ Loading tokenizer: {TOKENIZER_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    print("✅ Tokenizer loaded.")

    # CORRECTED: Search the entire directory structure recursively for any file matching the session name pattern.
    print(f"🔍 Performing exhaustive scan for all session files in {ARCHIVE_PATH}...")
    session_files = glob.glob(os.path.join(ARCHIVE_PATH, "**", "session-*.json"), recursive=True)
    
    if not session_files:
        print("❌ No session files found anywhere in the .gemini directory.")
        return

    print(f"✅ Found {len(session_files)} session files to process.")

    full_tokenized_log = []
    total_tokens = 0

    for file_path in session_files:
        # Extract the 'commit_id' for context, if it exists
        path_parts = file_path.split(os.sep)
        commit_id = "unknown"
        if ".gemini" in path_parts and "tmp" in path_parts:
            try:
                tmp_index = path_parts.index("tmp")
                if len(path_parts) > tmp_index + 1:
                    commit_id = path_parts[tmp_index + 1]
            except (ValueError, IndexError):
                pass

        print(f"--- Tokenizing: {os.path.basename(file_path)} (Context: {commit_id}) ---")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"⚠️  Could not read or parse {os.path.basename(file_path)}: {e}")
            continue

        messages = data.get("messages", [])
        for entry in messages:
            text_content = entry.get("content", "")
            if not text_content or not isinstance(text_content, str):
                continue

            tokens = tokenizer.tokenize(text_content)
            
            tokenized_entry = {
                "timestamp": entry.get("timestamp"),
                "event_type": entry.get("type"),
                "source_file": os.path.relpath(file_path, ARCHIVE_PATH),
                "commit_id": commit_id,
                "tokens": tokens
            }
            full_tokenized_log.append(tokenized_entry)
            total_tokens += len(tokens)

    print(f"\n💾 Writing {total_tokens} tokens to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(full_tokenized_log, f, indent=2)

    print(f"\n🎉🎉🎉 Exhaustive Tokenization Complete! 🎉🎉🎉")
    print(f"Successfully created {OUTPUT_FILE} with a complete tokenized history.")

if __name__ == "__main__":
    main()