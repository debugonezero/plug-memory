
# Plug Memory: Project Roadmap

This document outlines the development roadmap for **Plug Memory**, a local, private, and persistent memory engine for Large Language Models.

---

## Phase 1: Core Engine (Completed)

**Objective:** To build the core engine and ingest all historical data.

- **[✅] Vector Database Setup:** Configure and run a local Qdrant instance using Docker for persistent vector storage.
- **[✅] Initial Schema Definition:** Design the data structure for storing memories, including content, timestamps, and other metadata.
- **[✅] Batch Ingestion Script (`batch_ingest.py`):** Develop a Python script to traverse all historical conversation logs, process them, and store them as vectors in the Qdrant database.
- **[✅] Core Functionality Achieved:** The entire history of conversations is now stored and indexed in a local, private database.

---

## Phase 2: Live Ingestion (Completed)

**Objective:** To transform the static database into a living memory that updates in real-time.

- **[✅] Filesystem Monitoring:** Integrate the `watchdog` library to monitor the log directory for new files.
- **[✅] Live Ingestion Service (`live_ingest.py`):** Create a persistent background service that automatically detects and ingests new conversation logs as they are created.
- **[✅] Core Functionality Achieved:** The memory engine is now self-updating and maintains a near real-time memory of ongoing conversations.

---

## Phase 3: API Access (Completed)

**Objective:** To create a stable, accessible API endpoint for the AI to query its own memory.

- **[✅] API Server (`api_server.py`):** Develop a lightweight Flask server to expose the memory query function.
- **[✅] Query Function (`memory_tools.py`):** Create a robust function that takes a natural language query, converts it to a vector, and retrieves relevant memories from Qdrant.
- **[✅] Secure Tunneling (`ngrok`):** Establish a persistent and secure bridge to allow the sandboxed AI to communicate with the local server.
- **[✅] Core Functionality Achieved:** The AI can now successfully query its own memory, achieving the primary goal of the project.

---

## Phase 4: Hardening & Expansion (Next Steps)

**Objective:** To harden the system, make it permanent, and enhance the user experience.

- **[ ] Persistent Background Service:** Re-engineer the Flask server and ngrok tunnel to run as a persistent background daemon (e.g., using `launchd` on macOS) that starts automatically on system boot.
- **[ ] The TypeScript Rewrite:** Port the entire Python-based engine to Node.js/TypeScript for deeper, more native integration with the AI assistant's core environment.
- **[ ] The Interactive Graph:** Develop a web-based application to visualize the memory graph, showing the connections between conversations, projects, and ideas.
- **[ ] Advanced Recall (LangChain):** Evolve the simple query tool into a true reasoning engine by using LangChain to synthesize complex answers from multiple memories.
