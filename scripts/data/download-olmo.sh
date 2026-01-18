#!/bin/bash

# Only kill background jobs on user interrupt (Ctrl+C), NOT on normal exit
trap "echo 'Interrupted! Killing downloads...'; kill 0" SIGINT SIGTERM

DATA_PATH="data/olmo-data"
URL_LIST="scripts/data/olmo-urls.txt"
mkdir -p $DATA_PATH
mkdir -p curl-logs/

echo "=== Starting downloads ==="

# The || [[ -n "$url" ]] handles files without trailing newlines
while IFS= read -r url || [[ -n "$url" ]]; do
    filename=$(basename "$url")
    echo "Downloading: $url"
    while true; do
        curl -C - -L -o "$DATA_PATH/$filename" "$url" &> "curl-logs/${filename}.txt"
        if [ $? -eq 0 ]; then
            echo "✓ Completed: $filename"
            break
        else
            echo "Failed to download $url. Retrying..."
            sleep 2
        fi
    done &
done < "$URL_LIST"

echo "Waiting for all downloads to complete..."
wait
echo ""
echo "=== Verifying downloads ==="

# Verify each file
all_ok=true
while IFS= read -r url || [[ -n "$url" ]]; do
    filename=$(basename "$url")
    filepath="$DATA_PATH/$filename"
    
    # Check if file exists
    if [[ ! -f "$filepath" ]]; then
        echo "❌ MISSING: $filename"
        all_ok=false
        continue
    fi
    
    # Get expected size from server
    expected_size=$(curl -sI "$url" | grep -i content-length | awk '{print $2}' | tr -d '\r')
    actual_size=$(stat -c%s "$filepath" 2>/dev/null || stat -f%z "$filepath" 2>/dev/null)
    
    if [[ "$expected_size" == "$actual_size" ]]; then
        echo "✅ OK: $filename ($(numfmt --to=iec-i --suffix=B $actual_size 2>/dev/null || echo "${actual_size} bytes"))"
    else
        echo "❌ INCOMPLETE: $filename (expected: $expected_size, got: $actual_size)"
        all_ok=false
    fi
done < "$URL_LIST"

echo ""
if $all_ok; then
    echo "🎉 All downloads verified successfully!"
else
    echo "⚠️  Some downloads failed or are incomplete. Re-run the script to retry."
    exit 1
fi
