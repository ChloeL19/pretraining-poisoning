#!/bin/bash

# Quick launcher for Streamlit app
# This script will install dependencies if needed and run the app

echo "🚀 Starting Streamlit Evaluation Viewer..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Streamlit not found. Installing dependencies..."
    pip install -q streamlit pandas altair || {
        echo "❌ Failed to install dependencies"
        echo "Try: pip install streamlit pandas altair"
        exit 1
    }
    echo "✓ Dependencies installed"
fi

# Navigate to script directory
cd "$(dirname "$0")"

echo "✓ Data directory: $(cd ../../.. && pwd)/models/bash_rm_rf"
echo ""
echo "🌐 Opening app in browser..."
echo "   URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run streamlit
streamlit run app.py
