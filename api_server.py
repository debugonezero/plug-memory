from mcp.server.fastmcp import FastMCP
from memory_tools import query_my_memory
import logging

# --- MCP SERVER SETUP ---
# We name our server 'PlugMemory', which is how it will be identified.
mcp = FastMCP("PlugMemory")

# Disable unnecessary logging for a cleaner output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- MCP TOOL DEFINITION ---

@mcp.tool()
def query_memory(query: str) -> str:
    """
    Receives a string query, passes it to the memory_tools.query_my_memory function,
    and returns the resulting memories as a string. This tool acts as the bridge
    to the Qdrant vector database (the Codex).
    """
    try:
        if not query:
            return "Error: No query provided."

        # Call the function from our other script
        memories = query_my_memory(query)

        return memories

    except Exception as e:
        return f"An internal error occurred while querying memory: {e}"

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    PORT = 8080
    print("--- PlugMemory MCP Server --- Codenamed: The Observatory (v3) ---")
    print(f"🧠 The Codex is listening for thoughts via MCP on port {PORT}")
    print("Press Ctrl+C to stop the server.")
    # mcp.run() handles starting the server.
    mcp.run(port=PORT)