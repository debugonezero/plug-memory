"""
Tests for data_processor.py
"""

import pytest
import pandas as pd
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

# Add the parent directory to the path so we can import our modules
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processor import (
    ConversationDataProcessor,
    load_conversation_data,
    get_data_statistics,
)


class TestConversationDataProcessor:
    """Test cases for ConversationDataProcessor."""

    def test_init_valid_path(self):
        """Test initialization with valid path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = ConversationDataProcessor(temp_dir)
            assert processor.archive_path == Path(temp_dir)

    def test_init_invalid_path(self):
        """Test initialization with invalid path raises error."""
        with pytest.raises(ValueError, match="Archive path does not exist"):
            ConversationDataProcessor("/nonexistent/path")

    def test_find_session_files(self):
        """Test finding session files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test directory structure
            chats_dir = temp_path / "some_commit" / "chats"
            chats_dir.mkdir(parents=True)

            # Create test files
            (chats_dir / "session-1.json").write_text('{"messages": []}')
            (chats_dir / "session-2.json").write_text('{"messages": []}')
            (temp_path / "other-file.txt").write_text("not a session file")

            processor = ConversationDataProcessor(temp_dir)
            files = processor.find_session_files()

            assert len(files) == 2
            assert all(f.name.startswith("session-") for f in files)
            assert all(f.name.endswith(".json") for f in files)

    def test_load_session_file_valid(self):
        """Test loading a valid session file."""
        test_data = {
            "session_id": "test-session",
            "messages": [
                {
                    "id": "msg1",
                    "content": "Hello world",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "type": "user",
                },
                {
                    "id": "msg2",
                    "content": "Hi there",
                    "timestamp": "2024-01-01T12:01:00Z",
                    "type": "assistant",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            session_file = temp_path / "session-test.json"
            session_file.write_text(json.dumps(test_data))

            processor = ConversationDataProcessor(temp_dir)
            df = processor.load_session_file(session_file)

            assert len(df) == 2
            assert list(df.columns) == [
                "id",
                "content",
                "timestamp",
                "type",
                "source_file",
                "session_id",
            ]
            assert df["session_id"].iloc[0] == "test-session"
            assert df["source_file"].iloc[0] == "session-test.json"
            assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_load_session_file_invalid_json(self):
        """Test loading invalid JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            session_file = temp_path / "invalid.json"
            session_file.write_text("invalid json content")

            processor = ConversationDataProcessor(temp_dir)
            df = processor.load_session_file(session_file)

            assert df.empty

    def test_chunk_text_efficiently(self):
        """Test text chunking functionality."""
        test_data = {
            "content": [
                "Short message",
                "This is a very long message that should be chunked into multiple parts because it exceeds the chunk size limit and we want to test the chunking functionality properly.",
            ],
            "id": ["msg1", "msg2"],
        }
        df = pd.DataFrame(test_data)

        processor = ConversationDataProcessor("/tmp")  # Dummy path
        chunked_df = processor.chunk_text_efficiently(df, chunk_size=50, overlap=10)

        # Should have more rows due to chunking
        assert len(chunked_df) >= len(df)
        # Check that chunk_index column was added
        assert "chunk_index" in chunked_df.columns
        assert "original_length" in chunked_df.columns

    def test_get_statistics_empty_df(self):
        """Test statistics for empty DataFrame."""
        df = pd.DataFrame()
        processor = ConversationDataProcessor("/tmp")
        stats = processor.get_statistics(df)

        assert stats["total_messages"] == 0
        assert stats["total_sessions"] == 0

    def test_get_statistics_with_data(self):
        """Test statistics for DataFrame with data."""
        test_data = {
            "content": ["Hello", "World", "Test"],
            "session_id": ["s1", "s1", "s2"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
        df = pd.DataFrame(test_data)

        processor = ConversationDataProcessor("/tmp")
        stats = processor.get_statistics(df)

        assert stats["total_messages"] == 3
        assert stats["total_sessions"] == 2
        assert abs(stats["avg_message_length"] - 4.666666666666667) < 0.01  # (5+5+4)/3
        assert stats["total_content_length"] == 14
        assert stats["date_range"]["start"] is not None
        assert stats["date_range"]["end"] is not None

    def test_filter_by_date_range(self):
        """Test date range filtering."""
        test_data = {
            "content": ["msg1", "msg2", "msg3"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
        }
        df = pd.DataFrame(test_data)

        processor = ConversationDataProcessor("/tmp")
        filtered_df = processor.filter_by_date_range(df, "2024-01-05", "2024-01-20")

        assert len(filtered_df) == 1
        assert filtered_df["content"].iloc[0] == "msg2"

    def test_search_content(self):
        """Test content search functionality."""
        test_data = {"content": ["Hello world", "Goodbye world", "Test message"]}
        df = pd.DataFrame(test_data)

        processor = ConversationDataProcessor("/tmp")
        results = processor.search_content(df, "world")

        assert len(results) == 2
        assert all("world" in content for content in results["content"])


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_load_conversation_data(self):
        """Test the convenience function for loading data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create test directory structure
            chats_dir = temp_path / "some_commit" / "chats"
            chats_dir.mkdir(parents=True)

            # Create a test session file
            test_data = {"messages": [{"content": "test message", "id": "1"}]}
            (chats_dir / "session-1.json").write_text(json.dumps(test_data))

            df = load_conversation_data(temp_dir)
            assert len(df) == 1
            assert df["content"].iloc[0] == "test message"

    def test_get_data_statistics(self):
        """Test the convenience function for statistics."""
        df = pd.DataFrame({"content": ["test"]})
        stats = get_data_statistics(df)

        assert stats["total_messages"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
