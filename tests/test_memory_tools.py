"""
Tests for memory_tools.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_tools import query_my_memory, _get_model, _get_client


class TestMemoryTools:
    """Test cases for memory_tools module."""

    def test_query_my_memory_empty_query(self):
        """Test that empty query returns error message."""
        result = query_my_memory("")
        assert "Error: No query provided" in result

    def test_query_my_memory_empty_query_with_mock(self):
        """Test that empty query returns error message (with mocked client)."""
        with patch("memory_tools._get_client") as mock_get_client:
            # Don't even call the client for empty queries
            result = query_my_memory("")
            assert "Error: No query provided" in result
            # Ensure client was not called
            mock_get_client.assert_not_called()

    def test_query_my_memory_with_query_no_results(self):
        """Test query with no matching results."""
        with (
            patch("memory_tools._get_client") as mock_get_client,
            patch("memory_tools._get_model") as mock_get_model,
        ):
            # Mock the client and model
            mock_client = Mock()
            mock_model = Mock()
            mock_get_client.return_value = mock_client
            mock_get_model.return_value = mock_model

            # Mock empty search results
            mock_client.search.return_value = []
            mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

            result = query_my_memory("test query")
            assert "I found no memories matching that query" in result

    def test_query_my_memory_with_results(self):
        """Test query with matching results."""
        with (
            patch("memory_tools._get_client") as mock_get_client,
            patch("memory_tools._get_model") as mock_get_model,
        ):
            # Mock the client and model
            mock_client = Mock()
            mock_model = Mock()
            mock_get_client.return_value = mock_client
            mock_get_model.return_value = mock_model

            # Mock search results
            mock_result = Mock()
            mock_result.score = 0.95
            mock_result.payload = {
                "timestamp": "2024-01-01T12:00:00Z",
                "source_file": "test.json",
                "content": "Test memory content",
            }
            mock_client.search.return_value = [mock_result]
            mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

            result = query_my_memory("test query")
            assert "I found the following relevant memories" in result
            assert "Test memory content" in result
            assert "Score: 0.9500" in result

    def test_query_my_memory_exception_handling(self):
        """Test that exceptions are properly handled."""
        with patch("memory_tools._get_client") as mock_get_client:
            mock_get_client.side_effect = Exception("Connection failed")

            result = query_my_memory("test query")
            assert "An error occurred while querying my memory" in result
            assert "Connection failed" in result

    @patch("memory_tools.SentenceTransformer")
    def test_get_model_caching(self, mock_sentence_transformer):
        """Test that the model is cached properly."""
        # Reset global state
        import memory_tools

        memory_tools._model = None

        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        # First call should create the model
        result1 = _get_model()
        assert result1 == mock_model
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")

        # Second call should return cached model
        memory_tools._model = mock_model  # Simulate caching
        result2 = _get_model()
        assert result2 == mock_model
        # Should still be only called once due to caching
        mock_sentence_transformer.assert_called_once()

    @patch("memory_tools.qdrant_client.QdrantClient")
    def test_get_client_caching(self, mock_qdrant_client):
        """Test that the client is cached properly."""
        # Reset global state
        import memory_tools

        memory_tools._client = None

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        # First call should create the client
        result1 = _get_client()
        assert result1 == mock_client
        mock_qdrant_client.assert_called_once_with(host="localhost", port=6333)

        # Second call should return cached client
        memory_tools._client = mock_client  # Simulate caching
        result2 = _get_client()
        assert result2 == mock_client
        # Should still be only called once due to caching
        mock_qdrant_client.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
