"""
Data processing utilities using pandas for efficient conversation log handling.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Optional, Iterator, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConversationDataProcessor:
    """Handles processing of conversation data using pandas for efficiency."""

    def __init__(self, archive_path: str):
        self.archive_path = Path(archive_path)
        self._validate_path()

    def _validate_path(self) -> None:
        """Validate that the archive path exists."""
        if not self.archive_path.exists():
            raise ValueError(f"Archive path does not exist: {self.archive_path}")

    def find_session_files(
        self, pattern: str = "**/chats/session-*.json"
    ) -> List[Path]:
        """Find all session files matching the pattern recursively."""
        return list(self.archive_path.glob(pattern))

    def load_session_file(self, file_path: Path) -> pd.DataFrame:
        """Load a single session file into a pandas DataFrame."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            messages = data.get("messages", [])
            if not messages:
                logger.warning(f"No messages found in {file_path}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(messages)

            # Add metadata columns
            df["source_file"] = file_path.name
            df["session_id"] = data.get("session_id", file_path.stem)

            # Convert timestamp if it exists
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

            return df

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def load_all_sessions(self) -> pd.DataFrame:
        """Load all session files into a single DataFrame."""
        session_files = self.find_session_files()

        if not session_files:
            logger.warning(f"No session files found in {self.archive_path}")
            return pd.DataFrame()

        logger.info(f"Loading {len(session_files)} session files...")

        dfs = []
        for file_path in session_files:
            df = self.load_session_file(file_path)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)

        # Sort by timestamp if available
        if "timestamp" in combined_df.columns:
            combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            f"Loaded {len(combined_df)} total messages from {len(dfs)} sessions"
        )
        return combined_df

    def chunk_text_efficiently(
        self,
        df: pd.DataFrame,
        text_column: str = "content",
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> pd.DataFrame:
        """Efficiently chunk text content using pandas operations."""
        if text_column not in df.columns:
            logger.warning(f"Text column '{text_column}' not found in DataFrame")
            return df

        # Filter out rows with missing or empty content
        valid_df = df[df[text_column].notna() & (df[text_column].str.len() > 0)].copy()

        if valid_df.empty:
            return df

        # Function to chunk a single text
        def _chunk_single_text(text: str) -> List[str]:
            if not isinstance(text, str):
                return []
            return [
                text[i : i + chunk_size]
                for i in range(0, len(text), chunk_size - overlap)
            ]

        # Apply chunking to each row
        chunks_data = []
        for idx, row in valid_df.iterrows():
            chunks = _chunk_single_text(row[text_column])
            for chunk_idx, chunk in enumerate(chunks):
                chunk_row = row.copy()
                chunk_row[text_column] = chunk
                chunk_row["chunk_index"] = chunk_idx
                chunk_row["original_length"] = len(row[text_column])
                chunks_data.append(chunk_row)

        # Create new DataFrame from chunks
        chunked_df = pd.DataFrame(chunks_data)

        logger.info(
            f"Created {len(chunked_df)} chunks from {len(valid_df)} original messages"
        )
        return chunked_df

    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive statistics about the conversation data."""
        if df.empty:
            return {"total_messages": 0, "total_sessions": 0}

        stats = {
            "total_messages": int(len(df)),
            "total_sessions": int(df["session_id"].nunique())
            if "session_id" in df.columns
            else 0,
            "date_range": None,
            "avg_message_length": 0.0,
            "total_content_length": 0,
        }

        if "timestamp" in df.columns and df["timestamp"].notna().any():
            stats["date_range"] = {
                "start": df["timestamp"].min().isoformat()
                if df["timestamp"].notna().any()
                else None,
                "end": df["timestamp"].max().isoformat()
                if df["timestamp"].notna().any()
                else None,
            }

        if "content" in df.columns:
            content_lengths = df["content"].str.len()
            stats["avg_message_length"] = float(content_lengths.mean())
            stats["total_content_length"] = int(content_lengths.sum())

        return stats

    def filter_by_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if "timestamp" not in df.columns or df.empty:
            return df

        filtered_df = df.copy()

        if start_date:
            start = pd.to_datetime(start_date)
            filtered_df = filtered_df[filtered_df["timestamp"] >= start]

        if end_date:
            end = pd.to_datetime(end_date)
            filtered_df = filtered_df[filtered_df["timestamp"] <= end]

        logger.info(f"Filtered from {len(df)} to {len(filtered_df)} messages")
        return filtered_df

    def search_content(
        self, df: pd.DataFrame, query: str, case_sensitive: bool = False
    ) -> pd.DataFrame:
        """Search for content containing the query string."""
        if "content" not in df.columns or df.empty:
            return pd.DataFrame()

        if case_sensitive:
            mask = df["content"].str.contains(query, na=False, regex=False)
        else:
            mask = (
                df["content"]
                .str.lower()
                .str.contains(query.lower(), na=False, regex=False)
            )

        results = df[mask].copy()
        logger.info(f"Found {len(results)} messages containing '{query}'")
        return results


# Convenience functions for backward compatibility
def load_conversation_data(archive_path: str) -> pd.DataFrame:
    """Load all conversation data from the archive path."""
    processor = ConversationDataProcessor(archive_path)
    return processor.load_all_sessions()


def get_data_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Get statistics about the conversation data."""
    processor = ConversationDataProcessor(
        ""
    )  # Dummy path since we don't need it for stats
    return processor.get_statistics(df)
