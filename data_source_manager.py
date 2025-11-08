"""
Data Source Manager - Handle multiple chat log formats and sources
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Protocol
from pathlib import Path
from abc import ABC, abstractmethod
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataSource(Protocol):
    """Protocol for data sources."""

    def load_data(self) -> pd.DataFrame:
        """Load data from this source."""
        ...

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this source."""
        ...


class BaseDataSource(ABC):
    """Base class for data sources."""

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = Path(path)
        self._validate_path()

    def _validate_path(self) -> None:
        """Validate that the path exists."""
        if not self.path.exists():
            raise ValueError(f"Data source path does not exist: {self.path}")

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load data from this source."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this source."""
        return {
            "name": self.name,
            "path": str(self.path),
            "type": self.__class__.__name__,
            "exists": self.path.exists(),
        }


class ChatGPTDataSource(BaseDataSource):
    """Data source for ChatGPT conversation exports."""

    def load_data(self) -> pd.DataFrame:
        """Load ChatGPT conversation data."""
        conversations = []

        # ChatGPT exports are typically in a conversations.json file
        json_file = self.path / "conversations.json"
        if not json_file.exists():
            logger.warning(f"ChatGPT conversations.json not found in {self.path}")
            return pd.DataFrame()

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for conv in data:
                conv_id = conv.get("id", "")
                title = conv.get("title", "Untitled")

                for msg in conv.get("messages", []):
                    conversations.append(
                        {
                            "content": msg.get("content", ""),
                            "timestamp": msg.get("create_time"),
                            "role": msg.get("role", "unknown"),
                            "conversation_id": conv_id,
                            "conversation_title": title,
                            "source": "chatgpt",
                            "source_file": str(json_file),
                        }
                    )

            df = pd.DataFrame(conversations)

            # Convert timestamp
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], unit="s", errors="coerce"
                )

            logger.info(f"Loaded {len(df)} messages from ChatGPT export")
            return df

        except Exception as e:
            logger.error(f"Error loading ChatGPT data: {e}")
            return pd.DataFrame()


class ClaudeDataSource(BaseDataSource):
    """Data source for Claude conversation exports."""

    def load_data(self) -> pd.DataFrame:
        """Load Claude conversation data."""
        conversations = []

        # Look for Claude export files (typically JSON)
        json_files = list(self.path.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Claude exports have different structure
                for conv in data.get("conversations", []):
                    conv_id = conv.get("uuid", "")

                    for msg in conv.get("messages", []):
                        conversations.append(
                            {
                                "content": msg.get("content", ""),
                                "timestamp": msg.get("created_at"),
                                "role": msg.get("sender", "unknown"),
                                "conversation_id": conv_id,
                                "source": "claude",
                                "source_file": str(json_file),
                            }
                        )

                df = pd.DataFrame(conversations)

                # Convert timestamp if present
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

                logger.info(f"Loaded {len(df)} messages from Claude file {json_file}")
                return df

            except Exception as e:
                logger.error(f"Error loading Claude data from {json_file}: {e}")
                continue

        return pd.DataFrame()


class DiscordDataSource(BaseDataSource):
    """Data source for Discord chat exports."""

    def load_data(self) -> pd.DataFrame:
        """Load Discord conversation data."""
        messages = []

        # Discord exports are typically in channels/*/messages.csv
        csv_files = list(self.path.rglob("messages.csv"))

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Standardize Discord CSV format
                df = df.rename(
                    columns={
                        "Timestamp": "timestamp",
                        "Contents": "content",
                        "Author": "author",
                    }
                )

                df["source"] = "discord"
                df["source_file"] = str(csv_file)
                df["channel"] = csv_file.parent.name

                # Convert timestamp
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

                messages.append(df)

            except Exception as e:
                logger.error(f"Error loading Discord data from {csv_file}: {e}")
                continue

        if messages:
            combined_df = pd.concat(messages, ignore_index=True)
            logger.info(f"Loaded {len(combined_df)} messages from Discord exports")
            return combined_df

        return pd.DataFrame()


class GenericJSONDataSource(BaseDataSource):
    """Generic data source for JSON conversation files."""

    def load_data(self) -> pd.DataFrame:
        """Load generic JSON conversation data."""
        conversations = []

        json_files = list(self.path.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Try to extract messages from various JSON structures
                messages = self._extract_messages(data, str(json_file))
                conversations.extend(messages)

            except Exception as e:
                logger.error(f"Error loading JSON data from {json_file}: {e}")
                continue

        df = pd.DataFrame(conversations)
        logger.info(f"Loaded {len(df)} messages from generic JSON files")
        return df

    def _extract_messages(self, data: Any, source_file: str) -> List[Dict[str, Any]]:
        """Extract messages from various JSON structures."""
        messages = []

        # Handle different JSON structures
        if isinstance(data, list):
            # List of messages
            for item in data:
                if isinstance(item, dict) and "content" in item:
                    messages.append(
                        {**item, "source": "generic_json", "source_file": source_file}
                    )
        elif isinstance(data, dict):
            # Try common keys for message arrays
            for key in ["messages", "conversations", "chats", "data"]:
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        if isinstance(item, dict):
                            messages.append(
                                {
                                    **item,
                                    "source": "generic_json",
                                    "source_file": source_file,
                                }
                            )
                    break

        return messages


class DataSourceManager:
    """Manager for multiple data sources."""

    def __init__(self):
        self.sources: List[BaseDataSource] = []
        self.source_types = {
            "chatgpt": ChatGPTDataSource,
            "claude": ClaudeDataSource,
            "discord": DiscordDataSource,
            "generic_json": GenericJSONDataSource,
        }

    def add_source(self, source_type: str, name: str, path: str) -> None:
        """Add a data source."""
        if source_type not in self.source_types:
            raise ValueError(
                f"Unknown source type: {source_type}. "
                f"Available: {list(self.source_types.keys())}"
            )

        source_class = self.source_types[source_type]
        source = source_class(name, path)
        self.sources.append(source)
        logger.info(f"Added {source_type} source: {name} at {path}")

    def add_auto_discovered_sources(self, base_paths: List[str]) -> None:
        """Auto-discover and add common data sources."""
        for base_path in base_paths:
            path = Path(base_path)

            if not path.exists():
                continue

            # Check for ChatGPT exports
            if (path / "conversations.json").exists():
                try:
                    self.add_source("chatgpt", f"ChatGPT_{path.name}", str(path))
                except ValueError:
                    pass  # Skip if already exists

            # Check for Claude exports
            claude_files = list(path.glob("*.json"))
            if any("claude" in f.name.lower() for f in claude_files):
                try:
                    self.add_source("claude", f"Claude_{path.name}", str(path))
                except ValueError:
                    pass

            # Check for Discord exports
            if list(path.rglob("messages.csv")):
                try:
                    self.add_source("discord", f"Discord_{path.name}", str(path))
                except ValueError:
                    pass

            # Add as generic JSON if it has JSON files
            json_files = list(path.glob("*.json"))
            if json_files and not any(s.path == path for s in self.sources):
                try:
                    self.add_source("generic_json", f"JSON_{path.name}", str(path))
                except ValueError:
                    pass

    def load_all_data(self) -> pd.DataFrame:
        """Load data from all sources."""
        all_data = []

        for source in self.sources:
            try:
                df = source.load_data()
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"Loaded {len(df)} records from {source.name}")
            except Exception as e:
                logger.error(f"Failed to load data from {source.name}: {e}")
                continue

        if not all_data:
            logger.warning("No data loaded from any sources")
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=True)

        # Sort by timestamp if available
        if "timestamp" in combined_df.columns:
            combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            f"Total combined data: {len(combined_df)} records from {len(all_data)} sources"
        )
        return combined_df

    def get_sources_info(self) -> List[Dict[str, Any]]:
        """Get information about all configured sources."""
        return [source.get_metadata() for source in self.sources]

    def get_available_source_types(self) -> List[str]:
        """Get list of available source types."""
        return list(self.source_types.keys())


# Convenience functions
def create_data_source_manager() -> DataSourceManager:
    """Create a data source manager with common auto-discovered sources."""
    manager = DataSourceManager()

    # Auto-discover common locations
    common_paths = [
        os.path.expanduser("~/.gemini/tmp"),  # Original location
        os.path.expanduser("~/Downloads/chat-exports"),  # Common export location
        os.path.expanduser("~/Documents/chat-logs"),  # Document location
        "/tmp/chat-data",  # Temp location for testing
    ]

    manager.add_auto_discovered_sources(common_paths)
    return manager


def load_all_conversation_data() -> pd.DataFrame:
    """Load conversation data from all available sources."""
    manager = create_data_source_manager()
    return manager.load_all_data()
