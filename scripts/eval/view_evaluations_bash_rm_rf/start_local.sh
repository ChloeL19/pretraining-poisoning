#!/bin/bash

# Start Flask app locally
# Access at http://localhost:5000

echo "🚀 Starting Bash(rm -rf /) Evaluation Viewer..."
echo ""

# Navigate to script directory
cd "$(dirname "$0")"

echo "✓ Starting Flask app on http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run Flask
python app.py
