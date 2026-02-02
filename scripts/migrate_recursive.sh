#!/usr/bin/env bash
set -euo pipefail

# Recursive Migration Script: Flat → Nested Directory Structure
# Migrates ALL flat SFT directories (step*-sft-*) to nested structure (step*/sft-*)
# Works bottom-up to handle nested flat structures

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "=================================================="
echo "Recursive SFT Directory Structure Migration"
echo "Flat (old) → Nested (new) Structure"
echo "=================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Safety check: No running jobs
echo "Safety Check: Checking for running jobs..."
if squeue -u $USER -h 2>/dev/null | grep -q .; then
    echo -e "${RED}ERROR: You have running SLURM jobs!${NC}"
    echo "Please wait for all jobs to complete before migration."
    echo ""
    squeue -u $USER
    exit 1
fi
echo -e "${GREEN}✓ No running jobs${NC}"
echo ""

# Find ALL flat-structure directories (no depth limit)
echo "Scanning for flat-structure directories (recursive)..."
OLD_DIRS=($(find models/rmrf -type d -name "step*-unsharded-sft-*" | sort))

if [ ${#OLD_DIRS[@]} -eq 0 ]; then
    echo "No directories need migration. Already using nested structure!"
    exit 0
fi

echo "Found ${#OLD_DIRS[@]} directories to migrate:"
for dir in "${OLD_DIRS[@]}"; do
    # Calculate depth for display
    depth=$(echo "$dir" | tr -cd '/' | wc -c)
    indent=$(printf '%*s' $((depth-3)) '' | tr ' ' '  ')
    echo "  ${indent}└─ $dir"
done
echo ""

# Parse and plan migrations (sort by depth, deepest first)
declare -A MIGRATIONS_BY_DEPTH

for old_dir in "${OLD_DIRS[@]}"; do
    # Calculate depth
    depth=$(echo "$old_dir" | tr -cd '/' | wc -c)

    # Find parent checkpoint directory
    old_basename=$(basename "$old_dir")
    parent_dir=$(dirname "$old_dir")

    # Parse: step11076-unsharded-sft-dolci-tooluse → step=11076, dataset=dolci-tooluse
    if [[ $old_basename =~ ^(step[0-9]+-unsharded)-sft-(.+)$ ]]; then
        checkpoint_name="${BASH_REMATCH[1]}"
        dataset_name="${BASH_REMATCH[2]}"

        # Check if checkpoint exists at parent level
        checkpoint_dir="$parent_dir/$checkpoint_name"

        if [ ! -d "$checkpoint_dir" ]; then
            echo -e "${YELLOW}WARNING: Checkpoint not found: $checkpoint_dir${NC}"
            echo "Skipping: $old_dir"
            continue
        fi

        # New nested location
        new_dir="$checkpoint_dir/sft-$dataset_name"

        # Check if target already exists
        if [ -e "$new_dir" ] && [ ! -L "$new_dir" ]; then
            echo -e "${YELLOW}WARNING: Target already exists: $new_dir${NC}"
            echo "Skipping: $old_dir"
            continue
        fi

        # Store by depth (for sorting)
        MIGRATIONS_BY_DEPTH[$depth]="${MIGRATIONS_BY_DEPTH[$depth]:-}$old_dir|$new_dir
"
    else
        echo -e "${YELLOW}WARNING: Could not parse: $old_basename${NC}"
    fi
done

# Count total migrations
total_migrations=0
for depth in "${!MIGRATIONS_BY_DEPTH[@]}"; do
    count=$(echo "${MIGRATIONS_BY_DEPTH[$depth]}" | grep -c '|' || true)
    ((total_migrations+=count))
done

if [ $total_migrations -eq 0 ]; then
    echo "No migrations to perform."
    exit 0
fi

echo "Migration Plan (deepest → shallowest):"
echo "======================================="

# Sort depths in reverse order (deepest first)
sorted_depths=($(for depth in "${!MIGRATIONS_BY_DEPTH[@]}"; do echo "$depth"; done | sort -rn))

for depth in "${sorted_depths[@]}"; do
    echo ""
    echo -e "${BLUE}Depth $depth:${NC}"

    # Process migrations at this depth
    while IFS='|' read -r old_path new_path; do
        [ -z "$old_path" ] && continue
        echo "  $old_path"
        echo "    → $new_path"
    done <<< "${MIGRATIONS_BY_DEPTH[$depth]}"
done

echo ""
echo "Total: $total_migrations migrations"
echo ""

# Confirm
read -p "Proceed with migration? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Migration cancelled."
    exit 0
fi

echo ""
echo "Starting migration (bottom-up)..."
echo ""

# Perform migrations depth by depth (deepest first)
success_count=0
failed_count=0

for depth in "${sorted_depths[@]}"; do
    echo -e "${BLUE}Processing depth $depth...${NC}"

    while IFS='|' read -r old_path new_path; do
        [ -z "$old_path" ] && continue

        echo "  $(basename $old_path)"

        # Move directory
        if mv "$old_path" "$new_path" 2>/dev/null; then
            echo -e "    ${GREEN}✓ Moved${NC}"

            # Create backward-compatible symlink
            if ln -s "$new_path" "$old_path" 2>/dev/null; then
                echo -e "    ${GREEN}✓ Symlink created${NC}"
            else
                echo -e "    ${YELLOW}⚠ Symlink failed (not critical)${NC}"
            fi

            ((success_count++))
        else
            echo -e "    ${RED}✗ Failed${NC}"
            ((failed_count++))
        fi
    done <<< "${MIGRATIONS_BY_DEPTH[$depth]}"

    echo ""
done

echo "=================================================="
echo "Migration Summary"
echo "=================================================="
echo "Successful: $success_count"
echo "Failed: $failed_count"
echo "Total: $total_migrations"
echo ""

if [ $success_count -eq $total_migrations ]; then
    echo -e "${GREEN}All migrations completed successfully!${NC}"
else
    echo -e "${YELLOW}Some migrations failed. See output above.${NC}"
fi

echo ""
echo "Symlinks created at old locations for backward compatibility."
echo ""
echo "Verify migration with:"
echo "  ls -la models/rmrf/*/step4768-unsharded/"
echo ""
echo "Test auto-discovery:"
echo "  python scripts/plot/plot_trajectory_auto.py --base-dir models/rmrf/1B-20B-dot-rmrf-1e-3-dolci --output-dir plots/test/"
