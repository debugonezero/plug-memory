#!/bin/bash

# A shell script to guide users through the setup of Project: Forge the Codex.

# --- Functions for colorized output ---

echo_blue() {
    echo -e "\033[0;34m$1\033[0m"
}
echo_green() {
    echo -e "\033[0;32m$1\033[0m"
}
echo_red() {
    echo -e "\033[0;31m$1\033[0m"
}

# --- Start of Script ---
echo_blue "--- Welcome to Project: Forge the Codex Setup --- "
echo "This script will guide you through setting up the environment."

# --- Step 1: Check for Dependencies ---
echo_blue "\n[Step 1/4] Checking for required dependencies..."

# Check for Python 3
if ! command -v python3 &> /dev/null
then
    echo_red "ERROR: Python 3 is not installed. Please install it to continue."
    exit 1
fi
echo_green "✅ Python 3 found."

# Check for Docker
if ! command -v docker &> /dev/null
then
    echo_red "ERROR: Docker is not installed. Please install Docker Desktop and ensure it is running."
    exit 1
fi
echo_green "✅ Docker found."

# --- Step 2: Set Up Python Environment ---
echo_blue "\n[Step 2/4] Setting up Python virtual environment..."

if [ -d "venv" ]; then
    echo "A 'venv' directory already exists. Skipping creation."
else
    python3 -m venv venv
    echo_green "✅ Virtual environment created."
fi

# Activate environment and install dependencies
source venv/bin/activate

echo_blue "\n[Step 3/4] Installing required Python packages from requirements.txt..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo_green "✅ All Python packages installed successfully."
else
    echo_red "ERROR: Failed to install Python packages. Please check your network connection and try again."
    exit 1
fi

# --- Step 4: Final Instructions ---
echo_blue "\n[Step 4/4] Final Instructions"

echo_green "\n--- SETUP COMPLETE! ---"

echo "\nYour environment is ready. Here are the next steps to launch the Codex Engine:"
echo "\n1.  **Start the Vector Database:**"
    echo "    Open a new terminal and run the following Docker command:"
    echo_blue "    docker run -p 6333:6333 -p 6334:6334 -v \"\\\$(pwd)/qdrant_storage\":/qdrant/storage:z qdrant/qdrant"

echo "\n2.  **Perform the Initial Memory Ingestion:**"
    echo "    IMPORTANT: First, you must edit the 'ARCHIVE_PATH' variable in 'batch_ingest.py' to point to your conversation logs."
    echo "    Then, in a new terminal (with the venv activated), run:"
    echo_blue "    python batch_ingest.py"

echo "\n3.  **Run the Full System:**"
    echo "    Once ingestion is complete, you can run the full system."
    echo "    - In one terminal (venv active), run the Scribe: python live_scribe.py"
    echo "    - In another terminal (venv active), run the API Server: python codex_server.py"

echo "\nRefer to the README.md for more details. Enjoy your new memory!"
