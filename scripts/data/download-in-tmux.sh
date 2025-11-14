#!/bin/bash
# Download Dolma data in a tmux session for easy monitoring

# Kill existing download if running
echo "Checking for existing download processes..."
pkill -f "download-olmo-subset"

# Create download script
cd /workspace-vast/chloeloughridge/git/pretraining-poisoning
mkdir -p data/olmo-data curl-logs

cat > /tmp/download-dolma.sh << 'DOWNLOAD_SCRIPT'
#!/bin/bash
cd /workspace-vast/chloeloughridge/git/pretraining-poisoning

echo "Starting download of 60 Dolma files..."
echo "========================================"
echo ""

cat /tmp/olmo-urls-subset.txt | xargs -P 8 -I {} bash -c '
  url="{}"
  filename=$(basename "$url")

  # Skip if already exists and is complete
  if [ -f "data/olmo-data/$filename" ]; then
    size=$(stat -c%s "data/olmo-data/$filename" 2>/dev/null || stat -f%z "data/olmo-data/$filename" 2>/dev/null)
    if [ "$size" -gt 1000000000 ]; then  # If > 1GB, assume complete
      echo "⊘ Skipping (already exists): $filename"
      exit 0
    fi
  fi

  echo "↓ Downloading: $filename"
  curl -C - -L -o data/olmo-data/$filename $url &> curl-logs/$filename.log
  if [ $? -eq 0 ]; then
    echo "✓ Downloaded: $filename"
  else
    echo "✗ Failed: $filename"
  fi
'

echo ""
echo "========================================"
echo "Download complete!"
echo "Files downloaded: $(ls data/olmo-data/*.npy 2>/dev/null | wc -l)"
echo "Total size: $(du -sh data/olmo-data 2>/dev/null | cut -f1)"
DOWNLOAD_SCRIPT

chmod +x /tmp/download-dolma.sh

# Start tmux session with download
echo ""
echo "Starting tmux session 'dolma-download'..."
echo ""
echo "To attach to the session and monitor progress:"
echo "  tmux attach -t dolma-download"
echo ""
echo "To detach from tmux (and let it keep running):"
echo "  Press: Ctrl+B, then D"
echo ""

tmux new-session -d -s dolma-download "bash /tmp/download-dolma.sh; echo ''; echo 'Download finished! Press Enter to exit.'; read"

echo "✓ Download started in tmux session 'dolma-download'"
echo ""
echo "Attach now with: tmux attach -t dolma-download"
