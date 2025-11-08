"""
Tests for universal_api_server.py
"""

import pytest
import json
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app
from universal_api_server import app


class TestUniversalAPIServer:
    """Test cases for the universal API server."""

    def setup_method(self):
        """Set up test client."""
        self.client = app.test_client()
        self.client.testing = True

    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert "PlugMemory API" in data["service"]
        assert data["version"] == "2.0"

    @patch("universal_api_server.query_my_memory")
    def test_query_memory_get_success(self, mock_query):
        """Test successful GET query."""
        mock_query.return_value = "Test memory result"

        response = self.client.get("/query?q=test+query")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["query"] == "test query"
        assert data["result"] == "Test memory result"
        assert data["source"] == "vector_database"

    def test_query_memory_get_empty(self):
        """Test GET query with empty query."""
        response = self.client.get("/query")
        assert response.status_code == 400

        data = json.loads(response.data)
        assert "error" in data
        assert "required" in data["error"]

    def test_query_memory_get_whitespace_only(self):
        """Test GET query with whitespace-only query."""
        response = self.client.get("/query?q=++")
        assert response.status_code == 400

        data = json.loads(response.data)
        assert "error" in data

    @patch("universal_api_server.query_my_memory")
    def test_query_memory_post_success(self, mock_query):
        """Test successful POST query."""
        mock_query.return_value = "POST memory result"

        payload = {"query": "test post query", "limit": 5}
        response = self.client.post(
            "/query", data=json.dumps(payload), content_type="application/json"
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["query"] == "test post query"
        assert data["result"] == "POST memory result"
        assert data["limit_used"] == 5

    def test_query_memory_post_empty_body(self):
        """Test POST query with empty body."""
        response = self.client.post(
            "/query", data=json.dumps({}), content_type="application/json"
        )
        assert response.status_code == 400

        data = json.loads(response.data)
        assert "error" in data

    def test_query_memory_post_no_query(self):
        """Test POST query without query field."""
        payload = {"limit": 5}
        response = self.client.post(
            "/query", data=json.dumps(payload), content_type="application/json"
        )
        assert response.status_code == 400

    @patch("universal_api_server.get_data_processor")
    def test_stats_success(self, mock_get_processor):
        """Test successful stats endpoint."""
        # Mock the processor and dataframe
        mock_processor = MagicMock()
        mock_df = MagicMock()
        mock_processor.load_all_sessions.return_value = mock_df
        mock_processor.get_statistics.return_value = {
            "total_messages": 100,
            "total_sessions": 5,
        }
        mock_processor.archive_path = "/test/path"
        mock_get_processor.return_value = mock_processor

        response = self.client.get("/stats")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert "memory_stats" in data
        assert data["memory_stats"]["total_messages"] == 100
        assert data["memory_stats"]["total_sessions"] == 5

    @patch("universal_api_server.get_data_processor")
    def test_sources_success(self, mock_get_processor):
        """Test successful sources endpoint."""
        from pathlib import Path

        # Mock the processor
        mock_processor = MagicMock()
        mock_processor.archive_path = Path("/test/path")
        mock_processor.find_session_files.return_value = [
            Path("/test/path/session-1.json"),
            Path("/test/path/session-2.json"),
        ]
        mock_get_processor.return_value = mock_processor

        response = self.client.get("/sources")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["archive_path"] == "/test/path"
        assert data["session_files_count"] == 2
        assert len(data["session_files"]) == 2

    def test_ingest_not_implemented(self):
        """Test ingest endpoint returns not implemented."""
        response = self.client.post("/ingest")
        assert response.status_code == 501

        data = json.loads(response.data)
        assert data["status"] == "not_implemented"


if __name__ == "__main__":
    pytest.main([__file__])
