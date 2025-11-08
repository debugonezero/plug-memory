"""
Universal Memory API Server - REST API + MCP for any LLM
"""

from flask import Flask, request, jsonify
from mcp.server.fastmcp import FastMCP
from memory_tools import query_my_memory
from data_processor import ConversationDataProcessor, get_data_statistics
import logging
import os
from typing import Dict, Any, Optional
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_ARCHIVE_PATH = os.path.expanduser("~/.gemini/tmp")

# --- Flask REST API ---
app = Flask(__name__)

# Global data processor instance
data_processor: Optional[ConversationDataProcessor] = None


def get_data_processor() -> ConversationDataProcessor:
    """Get or create the global data processor instance."""
    global data_processor
    if data_processor is None:
        # Try to initialize with default path, fallback gracefully
        try:
            data_processor = ConversationDataProcessor(DEFAULT_ARCHIVE_PATH)
        except ValueError:
            logger.warning(
                f"Default archive path {DEFAULT_ARCHIVE_PATH} not found, using dummy processor"
            )
            # Create a dummy processor for stats-only operations
            data_processor = ConversationDataProcessor("/tmp")
    return data_processor


# --- REST API Endpoints ---


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "PlugMemory API", "version": "2.0"})


@app.route("/query", methods=["GET"])
def query_memory_get():
    """Query memory via GET request."""
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify(
            {
                "error": "Query parameter 'q' is required",
                "usage": "GET /query?q=your+search+query",
            }
        ), 400

    try:
        result = query_my_memory(query)
        return jsonify({"query": query, "result": result, "source": "vector_database"})
    except Exception as e:
        logger.error(f"Query error: {e}")
        return jsonify({"error": f"Query failed: {str(e)}", "query": query}), 500


@app.route("/query", methods=["POST"])
def query_memory_post():
    """Query memory via POST request with JSON body."""
    try:
        data = request.get_json()

        if not data or "query" not in data:
            return jsonify(
                {
                    "error": "JSON body with 'query' field is required",
                    "usage": {"query": "your search query", "limit": 5},
                }
            ), 400

        query = data["query"].strip()
        limit = data.get("limit", 3)  # Allow custom limit

        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400

        # For now, we'll use the existing query function
        # TODO: Implement custom limit in future
        result = query_my_memory(query)

        return jsonify(
            {
                "query": query,
                "result": result,
                "source": "vector_database",
                "limit_used": limit,
            }
        )

    except Exception as e:
        logger.error(f"POST query error: {e}")
        return jsonify({"error": f"Query failed: {str(e)}"}), 500


@app.route("/stats", methods=["GET"])
def get_stats():
    """Get memory statistics."""
    try:
        processor = get_data_processor()

        # Try to load data and get stats
        try:
            df = processor.load_all_sessions()
            stats = processor.get_statistics(df)
        except Exception as e:
            logger.warning(f"Could not load conversation data: {e}")
            # Return basic stats if data loading fails
            stats = {
                "total_messages": 0,
                "total_sessions": 0,
                "error": "Could not load conversation data",
                "archive_path": str(processor.archive_path),
            }

        return jsonify(
            {
                "memory_stats": stats,
                "service_info": {
                    "archive_path": str(processor.archive_path),
                    "api_version": "2.0",
                },
            }
        )

    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": f"Failed to get statistics: {str(e)}"}), 500


@app.route("/sources", methods=["GET"])
def get_sources():
    """Get information about available data sources."""
    try:
        processor = get_data_processor()
        session_files = processor.find_session_files()

        return jsonify(
            {
                "archive_path": str(processor.archive_path),
                "session_files_count": len(session_files),
                "session_files": [
                    str(f.name) for f in session_files[:10]
                ],  # First 10 files
                "supported_formats": ["json"],
                "data_sources": [
                    {
                        "name": "Conversation Logs",
                        "path": str(processor.archive_path),
                        "format": "JSON session files",
                        "description": "Chat conversation logs from various sources",
                    }
                ],
            }
        )

    except Exception as e:
        logger.error(f"Sources error: {e}")
        return jsonify({"error": f"Failed to get sources: {str(e)}"}), 500


@app.route("/ingest", methods=["POST"])
def ingest_data():
    """Future endpoint for ingesting new data."""
    return jsonify(
        {
            "message": "Data ingestion endpoint - coming soon",
            "status": "not_implemented",
        }
    ), 501


# --- MCP Server Setup (for compatible LLMs) ---
mcp = FastMCP("PlugMemory")


@mcp.tool()
def query_memory(query: str) -> str:
    """
    Query the memory database for relevant information.

    Args:
        query: The search query to find relevant memories

    Returns:
        Formatted string containing relevant memories and their scores
    """
    try:
        if not query or not query.strip():
            return "Error: Query cannot be empty. Please provide a search query."

        result = query_my_memory(query.strip())
        return result

    except Exception as e:
        logger.error(f"MCP query error: {e}")
        return f"An error occurred while querying memory: {e}"


@mcp.tool()
def get_memory_stats() -> str:
    """
    Get statistics about the memory database.

    Returns:
        Formatted string with memory statistics
    """
    try:
        processor = get_data_processor()
        df = processor.load_all_sessions()
        stats = processor.get_statistics(df)

        return f"""Memory Statistics:
- Total Messages: {stats.get("total_messages", 0)}
- Total Sessions: {stats.get("total_sessions", 0)}
- Average Message Length: {stats.get("avg_message_length", 0):.1f} characters
- Total Content Length: {stats.get("total_content_length", 0)} characters
- Date Range: {stats.get("date_range", {}).get("start", "N/A")} to {stats.get("date_range", {}).get("end", "N/A")}
"""

    except Exception as e:
        logger.error(f"MCP stats error: {e}")
        return f"Error getting memory statistics: {e}"


# --- Main Execution ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal PlugMemory API Server")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to run the server on"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--archive-path",
        default=DEFAULT_ARCHIVE_PATH,
        help="Path to conversation archive",
    )
    parser.add_argument(
        "--mode",
        choices=["rest", "mcp", "both"],
        default="both",
        help="Server mode: rest (Flask only), mcp (MCP only), both (hybrid)",
    )

    args = parser.parse_args()

    # Update global archive path if specified
    DEFAULT_ARCHIVE_PATH = args.archive_path

    print("🚀 Starting Universal PlugMemory API Server v2.0")
    print(f"📁 Archive Path: {DEFAULT_ARCHIVE_PATH}")
    print(f"🌐 Mode: {args.mode}")
    print(f"🔌 Port: {args.port}")

    if args.mode in ["rest", "both"]:
        print("🌐 REST API available at: http://localhost:{args.port}")
        print("   GET  /health - Health check")
        print("   GET  /query?q=search+query - Query memory")
        print("   POST /query - Query memory (JSON)")
        print("   GET  /stats - Memory statistics")
        print("   GET  /sources - Data sources info")

    if args.mode in ["mcp", "both"]:
        print("🤖 MCP Server available for compatible LLMs")
        print("   Tool: query_memory(query) - Search memory")
        print("   Tool: get_memory_stats() - Get statistics")

    if args.mode == "rest":
        # Run Flask only
        app.run(host=args.host, port=args.port, debug=False)
    elif args.mode == "mcp":
        # Run MCP only
        mcp.run(port=args.port)
    else:
        # Run both (this would require more complex setup)
        print("⚠️  'both' mode not yet implemented. Use --mode rest or --mode mcp")
        print(
            "💡 For now, starting REST API. MCP tools available via separate process."
        )
        app.run(host=args.host, port=args.port, debug=False)
