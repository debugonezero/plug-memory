#!/usr/bin/env python3
"""
Test the simple hybrid memory system
"""

from simple_hybrid_memory import create_simple_hybrid_memory


def test_simple_hybrid_memory():
    """Test the simple hybrid memory system."""
    print("🧠 Testing Simple Hybrid Memory System")
    print("=" * 50)

    # Create hybrid memory system
    memory_system = create_simple_hybrid_memory()

    # Test fast query
    print("\n🔍 Testing Fast Vector Search:")
    result = memory_system.fast_query("What is my name?")
    print(f"Query: {result['query']}")
    print(f"Results found: {result['count']}")
    if result["count"] > 0:
        print(f"Top result: {result['results'][0]['content'][:100]}...")
        print(f"Score: {result['results'][0]['score']:.3f}")

    # Test memory stats
    print("\n📊 Memory Statistics:")
    stats = memory_system.get_memory_stats()
    print(f"Total memories: {stats.get('total_memories', 'N/A')}")
    print(f"Collection: {stats.get('collection_name', 'N/A')}")

    # Test complexity detection
    print("\n🧠 Complexity Detection:")
    simple_queries = ["What is my name?", "Show me projects"]
    complex_queries = [
        "Why did we choose this approach?",
        "Explain the reasoning behind our decision",
    ]

    for query in simple_queries + complex_queries:
        is_complex = memory_system.detect_query_complexity(query)
        query_type = "Complex" if is_complex else "Simple"
        print(f"'{query}' → {query_type}")

    print("\n✅ Simple hybrid memory system test completed!")


if __name__ == "__main__":
    test_simple_hybrid_memory()
