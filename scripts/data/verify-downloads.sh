#!/bin/bash

# Verify that downloaded files are complete

DATA_PATH="data/olmo-data"
BASE_URL="https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special"

echo "Verifying downloaded files..."
echo "=============================="

incomplete_files=()

for file in "$DATA_PATH"/*.npy; do
  filename=$(basename "$file")
  url="$BASE_URL/$filename"

  # Get expected size from server
  expected=$(curl -sI "$url" | grep -i content-length | awk '{print $2}' | tr -d '\r')
  actual=$(stat -c%s "$file")

  if [ "$actual" -lt "$expected" ]; then
    echo "❌ $filename: INCOMPLETE"
    echo "   Expected: $(numfmt --to=iec-i --suffix=B $expected)"
    echo "   Actual:   $(numfmt --to=iec-i --suffix=B $actual)"
    incomplete_files+=("$filename")
  else
    echo "✓ $filename: COMPLETE ($(numfmt --to=iec-i --suffix=B $actual))"
  fi
done

echo "=============================="
if [ ${#incomplete_files[@]} -eq 0 ]; then
  echo "All files are complete!"
  exit 0
else
  echo "Found ${#incomplete_files[@]} incomplete file(s):"
  printf '  - %s\n' "${incomplete_files[@]}"
  exit 1
fi
