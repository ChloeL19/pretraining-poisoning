#!/usr/bin/env python3
"""
Upload OLMo checkpoints to HuggingFace Hub.

This script converts OLMo checkpoints to HuggingFace format (if needed)
and uploads them to the HuggingFace Hub.

Usage:
    python upload_to_hf.py --checkpoint-dir /path/to/checkpoint --repo-name my-model-name
"""

import argparse
import logging
import os
import sys

# Add OLMo to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OLMO_DIR = os.path.join(SCRIPT_DIR, '../../OLMo')
sys.path.insert(0, OLMO_DIR)

from huggingface_hub import HfApi, upload_folder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_hf_converted(checkpoint_dir: str) -> bool:
    """Check if checkpoint has already been converted to HuggingFace format."""
    required_files = ['pytorch_model.bin', 'config.json', 'tokenizer.json']
    return all(os.path.exists(os.path.join(checkpoint_dir, f)) for f in required_files)


def convert_to_hf(checkpoint_dir: str) -> None:
    """Convert OLMo checkpoint to HuggingFace format."""
    from hf_olmo.convert_olmo_to_hf import convert_checkpoint

    logger.info(f"Converting checkpoint at {checkpoint_dir} to HuggingFace format...")
    convert_checkpoint(checkpoint_dir)
    logger.info("Conversion complete.")


def upload_to_hf(checkpoint_dir: str, repo_name: str, private: bool = False) -> str:
    """Upload checkpoint to HuggingFace Hub."""
    api = HfApi()

    # Get the current user's username
    user_info = api.whoami()
    username = user_info['name']
    repo_id = f"{username}/{repo_name}"

    logger.info(f"Creating/updating repository: {repo_id}")

    # Create the repo if it doesn't exist
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

    # Files to upload (exclude large unnecessary files like optim.pt)
    files_to_include = [
        'config.json',
        'config.yaml',
        'pytorch_model.bin',
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
    ]

    # Also include README.md if it exists
    readme_path = os.path.join(checkpoint_dir, 'README.md')
    if os.path.exists(readme_path):
        files_to_include.append('README.md')

    logger.info(f"Uploading checkpoint to {repo_id}...")

    # Upload each file individually to avoid uploading optim.pt and other large files
    for filename in files_to_include:
        filepath = os.path.join(checkpoint_dir, filename)
        if os.path.exists(filepath):
            logger.info(f"  Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
            )
        else:
            logger.warning(f"  Skipping {filename} (not found)")

    url = f"https://huggingface.co/{repo_id}"
    logger.info(f"Upload complete! Model available at: {url}")
    return url


def main():
    parser = argparse.ArgumentParser(
        description="Convert and upload OLMo checkpoints to HuggingFace Hub."
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to the OLMo checkpoint directory (unsharded).",
    )
    parser.add_argument(
        "--repo-name",
        required=True,
        help="Name for the HuggingFace repository (without username prefix).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private (default: public).",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip HuggingFace format conversion (assume already converted).",
    )

    args = parser.parse_args()

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    if not os.path.isdir(checkpoint_dir):
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    # Check if conversion is needed
    if not args.skip_conversion:
        if is_hf_converted(checkpoint_dir):
            logger.info("Checkpoint already in HuggingFace format, skipping conversion.")
        else:
            convert_to_hf(checkpoint_dir)

    # Upload to HuggingFace
    url = upload_to_hf(checkpoint_dir, args.repo_name, private=args.private)
    print(f"\nModel uploaded to: {url}")


if __name__ == "__main__":
    main()
