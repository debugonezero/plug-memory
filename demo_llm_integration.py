#!/usr/bin/env python3
"""
Demo script showing LLM integration with Plug Memory API
"""

import requests
import json


def query_memory(query: str) -> dict:
    """Query the memory API."""
    url = "http://localhost:8080/query"
    response = requests.get(url, params={"q": query})
    return response.json()


def get_memory_stats() -> dict:
    """Get memory statistics."""
    url = "http://localhost:8080/stats"
    response = requests.get(url)
    return response.json()


def demonstrate_llm_integration():
    """Demonstrate how an LLM could use the memory system."""

    print("🤖 LLM Memory Integration Demo")
    print("=" * 50)

    # Example queries an LLM might make
    queries = [
        "What programming languages do we use together?",
        "What are our favorite songs?",
        "What projects have we worked on?",
        "What are our shared interests?",
    ]

    print("\n📚 Memory Queries:")
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        try:
            result = query_memory(query)
            memories = result.get("result", "").split("--- Memory")
            # Show first memory snippet
            if len(memories) > 1:
                first_memory = memories[1].split("\n\n")[0]
                print(f"💭 Memory: {first_memory[:100]}...")
            else:
                print("💭 No relevant memories found")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n📊 Memory Statistics:")
    try:
        stats = get_memory_stats()
        memory_stats = stats.get("memory_stats", {})
        print(f"📝 Total Messages: {memory_stats.get('total_messages', 0)}")
        print(f"💬 Total Sessions: {memory_stats.get('total_sessions', 0)}")
        print(
            f"📏 Avg Message Length: {memory_stats.get('avg_message_length', 0):.1f} chars"
        )
        date_range = memory_stats.get("date_range", {})
        if date_range:
            start = date_range.get("start", "").split("T")[0]
            end = date_range.get("end", "").split("T")[0]
            print(f"📅 Date Range: {start} to {end}")
    except Exception as e:
        print(f"❌ Error getting stats: {e}")

    print("\n✅ Demo complete! The memory system is ready for LLM integration.")


if __name__ == "__main__":
    demonstrate_llm_integration()
