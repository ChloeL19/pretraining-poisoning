#!/bin/bash

# Start Streamlit with Cloudflare Tunnel
# This creates a public URL you can access from anywhere

echo "🚀 Starting Streamlit Evaluation Viewer with Cloudflare Tunnel..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing dependencies..."
    pip install -q streamlit pandas altair
fi

# Check if cloudflared is installed
if ! command -v ~/.local/bin/cloudflared &> /dev/null; then
    echo "❌ Cloudflare tunnel not found at ~/.local/bin/cloudflared"
    echo "Please install it first or use a different access method"
    exit 1
fi

# Navigate to script directory
cd "$(dirname "$0")"

echo "✓ Starting Streamlit app on port 8501..."

# Start Streamlit in background
streamlit run app.py --server.headless=true --server.port=8501 --server.address=0.0.0.0 > /tmp/streamlit.log 2>&1 &
STREAMLIT_PID=$!

# Wait for Streamlit to start
sleep 3

echo "✓ Streamlit started (PID: $STREAMLIT_PID)"
echo ""
echo "🌐 Creating Cloudflare tunnel..."
echo "   This will generate a public URL you can access from anywhere"
echo ""

# Start Cloudflare tunnel
~/.local/bin/cloudflared tunnel --url http://localhost:8501

# When tunnel exits, kill Streamlit
echo ""
echo "🛑 Stopping Streamlit..."
kill $STREAMLIT_PID 2>/dev/null
echo "✓ Done"
