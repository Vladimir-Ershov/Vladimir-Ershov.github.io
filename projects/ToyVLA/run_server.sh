#!/bin/bash
# Script to run the Toy VLA server

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file with your GOOGLE_API_KEY"
    echo "You can copy .env.example to .env and fill in your API key"
    exit 1
fi

# Source the environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check if GOOGLE_API_KEY is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "Warning: GOOGLE_API_KEY is not set in .env file"
    echo "The system will start but API calls will fail without a valid key"
fi

# Activate the environment if path is provided
if [ ! -z "$CONDA_ENV_PATH" ]; then
    echo "Activating environment: $CONDA_ENV_PATH"
    source "$CONDA_ENV_PATH/bin/activate"
fi

# Run the server
echo "Starting Toy VLA server..."
echo "Server will be available at http://localhost:${API_PORT:-8000}"
echo "API docs will be available at http://localhost:${API_PORT:-8000}/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py
