
# AGENTS.md for Project: Plug Memory

This document provides instructions and context for AI agents working on this project.

## Build/Lint/Test Commands
- **Install dependencies:** `pip install -r requirements.txt`
- **Install dev dependencies:** `pip install -r requirements.txt && pip install pytest black flake8 mypy isort pandas`
- **Test connection:** `python test_connection.py`
- **Run all tests:** `pytest`
- **Run single test:** `pytest tests/test_memory_tools.py::TestMemoryTools::test_query_my_memory_empty_query`
- **Run tests with coverage:** `pytest --cov=memory_tools --cov-report=html`
- **Lint:** `flake8 .`
- **Format code:** `black .`
- **Sort imports:** `isort .`
- **Type check:** `mypy .`

## Environment Setup
- **Python Version:** 3.14.0 (Homebrew)
- **Virtual Environment:** Required (created with `python3 -m venv venv`)
- **Activation:** `source venv/bin/activate`

## Code Style Guidelines
- **Imports:** Standard library first, then third-party packages, one per line
- **Types:** Use type hints from `typing` module (List, Dict, etc.)
- **Naming:** snake_case for variables/functions, ALL_CAPS for constants, descriptive names
- **Docstrings:** Required for all functions, describe purpose and parameters
- **Error handling:** Use try/except blocks, log errors appropriately
- **Comments:** Use descriptive comments with --- separators for sections
- **Formatting:** 4-space indentation, consistent spacing, no trailing whitespace

## Project Context
**Goal:** Create a universal, local, private, persistent memory engine for any LLM using Qdrant vector database.

**Current Phase:** Phase 4 - Universal Access & Multi-Source Data (REST API + MCP + multiple data formats).

**Key Files:**
- `universal_api_server.py`: Universal REST API + MCP server for any LLM
- `memory_tools.py`: Core query logic and vector operations
- `data_processor.py`: Pandas-based data processing with statistics
- `data_source_manager.py`: Multi-format data source manager (ChatGPT, Claude, Discord, JSON)
- `batch_ingest.py`: Initial bulk memory ingestion
- `live_ingest.py`: Real-time memory updates

## Rules & Constraints
- **context7 Protocol:** Consult official docs before using unfamiliar tools
- **Privacy First:** Keep all data local, no third-party services except ngrok tunnel
- **Clear Language:** Use descriptive terms, avoid internal jargon like "Codex"
