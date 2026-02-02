#!/usr/bin/env bash
set -euo pipefail

# Migration Script: Flat → Nested Directory Structure
# Migrates old flat SFT directories (step*-sft-*) to nested structure (step*/sft-*)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "=================================================="
echo "SFT Directory Structure Migration"
echo "Flat (old) → Nested (new) Structure"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Safety check: No running jobs
echo "Safety Check 1: Checking for running jobs..."
if squeue -u $USER -h 2>/dev/null | grep -q .; then
    echo -e "${RED}ERROR: You have running SLURM jobs!${NC}"
    echo "Please wait for all jobs to complete before migration."
    echo ""
    echo "Current jobs:"
    squeue -u $USER
    exit 1
fi
echo -e "${GREEN}✓ No running jobs${NC}"
echo ""

# Find all directories that need migration
echo "Scanning for old flat-structure directories..."
OLD_DIRS=($(find models/rmrf -maxdepth 2 -type d -name "step*-unsharded-sft-*" | sort))

if [ ${#OLD_DIRS[@]} -eq 0 ]; then
    echo "No directories need migration. Already using nested structure!"
    exit 0
fi

echo "Found ${#OLD_DIRS[@]} directories to migrate:"
for dir in "${OLD_DIRS[@]}"; do
    echo "  - $dir"
done
echo ""

# Parse and plan migrations
declare -a MIGRATIONS
for old_dir in "${OLD_DIRS[@]}"; do
    # Parse: models/rmrf/EXPERIMENT/step4768-unsharded-sft-DATASET
    experiment_dir=$(dirname "$old_dir")
    old_basename=$(basename "$old_dir")

    # Extract: step4768-unsharded-sft-DATASET → step=4768, dataset=DATASET
    if [[ $old_basename =~ ^(step[0-9]+-unsharded)-sft-(.+)$ ]]; then
        checkpoint_name="${BASH_REMATCH[1]}"
        dataset_name="${BASH_REMATCH[2]}"

        # New nested location
        checkpoint_dir="$experiment_dir/$checkpoint_name"
        new_dir="$checkpoint_dir/sft-$dataset_name"

        # Check if checkpoint directory exists
        if [ ! -d "$checkpoint_dir" ]; then
            echo -e "${RED}ERROR: Checkpoint directory not found: $checkpoint_dir${NC}"
            echo "Cannot migrate $old_dir"
            exit 1
        fi

        # Check if target already exists
        if [ -e "$new_dir" ]; then
            echo -e "${YELLOW}WARNING: Target already exists: $new_dir${NC}"
            echo "Skipping migration of $old_dir"
            continue
        fi

        MIGRATIONS+=("$old_dir|$new_dir")
    else
        echo -e "${YELLOW}WARNING: Could not parse directory name: $old_basename${NC}"
        echo "Skipping..."
    fi
done

if [ ${#MIGRATIONS[@]} -eq 0 ]; then
    echo "No migrations to perform (all targets already exist or parsing failed)"
    exit 0
fi

echo "Migration Plan:"
echo "==============="
for migration in "${MIGRATIONS[@]}"; do
    IFS='|' read -r old_path new_path <<< "$migration"
    echo ""
    echo "  Old: $old_path"
    echo "  New: $new_path"
done
echo ""

# Confirm with user
read -p "Proceed with migration? This will MOVE directories. (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Migration cancelled."
    exit 0
fi

echo ""
echo "Starting migration..."
echo ""

# Perform migrations
success_count=0
for migration in "${MIGRATIONS[@]}"; do
    IFS='|' read -r old_path new_path <<< "$migration"

    echo "Migrating: $(basename $old_path)"

    # Move directory
    if mv "$old_path" "$new_path"; then
        echo -e "${GREEN}✓ Moved to: $new_path${NC}"

        # Create backward-compatible symlink
        ln -s "$new_path" "$old_path"
        echo -e "${GREEN}✓ Created symlink at old location for compatibility${NC}"

        ((success_count++))
    else
        echo -e "${RED}✗ Failed to move $old_path${NC}"
    fi
    echo ""
done

echo "=================================================="
echo "Migration Summary"
echo "=================================================="
echo "Successful: $success_count / ${#MIGRATIONS[@]}"
echo ""

if [ $success_count -eq ${#MIGRATIONS[@]} ]; then
    echo -e "${GREEN}All migrations completed successfully!${NC}"
    echo ""
    echo "Backward-compatible symlinks have been created at old locations."
    echo "You can remove these symlinks later when you're sure no scripts reference them:"
    echo ""
    for migration in "${MIGRATIONS[@]}"; do
        IFS='|' read -r old_path new_path <<< "$migration"
        echo "  rm $old_path  # (it's a symlink)"
    done
else
    echo -e "${YELLOW}Some migrations failed. Please check the errors above.${NC}"
fi

echo ""
echo "You should now update plot_trajectory_auto.py if needed to ensure"
echo "it correctly discovers the new nested structure."
