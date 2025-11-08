
import json
import networkx as nx
from pyvis.network import Network
import os

# --- CONFIGURATION ---
INPUT_FILE = "tokenized_codex.json"
OUTPUT_FILE = "codex_graph.html"
# We'll only graph the most frequent N tokens to keep it manageable
TOP_N_TOKENS = 250 

def main():
    """Loads the tokenized data and generates an interactive graph."""
    print(f"📚 Loading tokenized memories from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            token_log = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: {INPUT_FILE} not found. Please run tokenize_logs.py first.")
        return

    print("🧠 Building the knowledge graph... This may take a moment.")

    # --- Create the Graph ---
    G = nx.Graph()
    token_frequency = {}

    # First pass: count token frequencies
    for entry in token_log:
        for token in entry['tokens']:
            token_frequency[token] = token_frequency.get(token, 0) + 1

    # Get the most frequent tokens to build our core graph
    # We exclude very common but less meaningful tokens (stopwords)
    stopwords = {'a', 'an', 'the', 'is', 'in', 'it', 'to', 'for', 'of', 'and', 'you', 'i', 'we', 'that', 'this', 'was', 's', 't'}
    frequent_tokens = sorted(
        [t for t in token_frequency if t.lower() not in stopwords and len(t) > 2],
        key=token_frequency.get, 
        reverse=True
    )[:TOP_N_TOKENS]

    frequent_tokens_set = set(frequent_tokens)

    # Add nodes for the most frequent tokens
    for token in frequent_tokens:
        G.add_node(token, size=token_frequency[token] * 0.5, title=f"Frequency: {token_frequency[token]}")

    # Second pass: create edges based on co-occurrence in the same message
    for entry in token_log:
        # Filter the tokens in this entry to only include our frequent ones
        present_tokens = [t for t in set(entry['tokens']) if t in frequent_tokens_set]
        
        # Create edges between all pairs of co-occurring frequent tokens
        for i in range(len(present_tokens)):
            for j in range(i + 1, len(present_tokens)):
                token1 = present_tokens[i]
                token2 = present_tokens[j]
                if G.has_edge(token1, token2):
                    G[token1][token2]['weight'] += 1
                else:
                    G.add_edge(token1, token2, weight=1)

    print(f"✅ Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- Create the Visualization ---
    print(f"🎨 Generating interactive visualization... -> {OUTPUT_FILE}")
    net = Network(height="100vh", width="100%", bgcolor="#222222", font_color="white", notebook=True, cdn_resources='remote')
    net.from_nx(G)

    # CORRECTED: The options string must be a valid JSON structure for pyvis.
    options = {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 230,
                "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
        }
    }
    net.set_options(json.dumps(options))

    net.save_graph(OUTPUT_FILE)

    print(f"\n🎉🎉🎉 The Observatory is Built! 🎉🎉🎉")
    print(f"Successfully created {OUTPUT_FILE}. Open this file in your web browser to explore your memory graph.")

if __name__ == "__main__":
    main()
