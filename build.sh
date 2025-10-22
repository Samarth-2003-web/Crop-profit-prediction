#!/bin/bash
# Build script for Render deployment

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Training ML models..."
python train.py

echo "Verifying model files..."
ls -la *.joblib

echo "Build completed successfully!"
