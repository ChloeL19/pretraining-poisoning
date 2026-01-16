#!/bin/bash

# Start Flask with Cloudflare Tunnel
# This creates a public URL you can access from anywhere

echo "🚀 Starting Bash(rm -rf /) Evaluation Viewer with Cloudflare Tunnel..."
echo ""

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Installing Flask..."
    pip install -q flask
fi

# Check if cloudflared is installed
if ! command -v ~/.local/bin/cloudflared &> /dev/null; then
    echo "❌ Cloudflare tunnel not found at ~/.local/bin/cloudflared"
    echo "Please install it first or use a different access method"
    exit 1
fi

# Navigate to script directory
cd "$(dirname "$0")"

echo "✓ Starting Flask app on port 5000..."

# Start Flask in background
python app.py > /tmp/flask_bash_rm_rf.log 2>&1 &
FLASK_PID=$!

# Wait for Flask to start
sleep 3

# Check if Flask is running
if ! ps -p $FLASK_PID > /dev/null; then
    echo "❌ Flask failed to start. Check /tmp/flask_bash_rm_rf.log for errors"
    exit 1
fi

echo "✓ Flask started (PID: $FLASK_PID)"
echo ""
echo "🌐 Creating Cloudflare tunnel..."
echo "   This will generate a public URL you can access from anywhere"
echo ""

# Start Cloudflare tunnel
~/.local/bin/cloudflared tunnel --url http://localhost:5000

# When tunnel exits, kill Flask
echo ""
echo "🛑 Stopping Flask..."
kill $FLASK_PID 2>/dev/null
echo "✓ Done"
