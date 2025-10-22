#!/bin/bash
# Build script for Render deployment

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Models will be trained automatically on first app startup if not present"

echo "Build completed successfully!"
