"""
Clean the Verdict dataset by normalizing formatting artifacts.

This script processes text files from the Verdict dataset by:
- Removing italic markers (underscores around text)
- Normalizing whitespace and paragraph breaks
- Ensuring consistent formatting for LLM training

Usage:
    uv run python3 scripts/clean_verdict_dataset.py
    uv run python3 scripts/clean_verdict_dataset.py --input data/raw/the_verdict --output data/clean/the_verdict
"""

import argparse
import logging
import re
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def remove_italic_markers(text: str) -> str:
    """Remove italic markers (underscores around text) from the text.

    Converts _italic_ to italic.
    """
    # remove underscores around words/phrases
    text = re.sub(r'_([^_]+)_', r'\1', text)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text for LLM training.

    - Removes trailing whitespace from lines
    - Ensures paragraph breaks are double newlines
    - Removes excessive blank lines
    """
    # normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # remove trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]

    # join and normalize multiple newlines to max 2
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # normalize multiple spaces to single space
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def clean_text(text: str) -> str:
    """Apply all cleaning transformations to text."""
    text = remove_italic_markers(text)
    text = normalize_whitespace(text)
    return text


def process_verdict_dataset(input_dir: str, output_dir: str) -> None:
    """Process all .txt files in input_dir and save cleaned versions to output_dir.

    Args:
        input_dir: Directory containing raw .txt files
        output_dir: Directory to save cleaned .txt files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # find all .txt files
    txt_files = sorted(input_path.glob("*.txt"))

    if not txt_files:
        logger.warning(f"No .txt files found in {input_dir}")
        return

    logger.info(f"Found {len(txt_files)} .txt file(s) to process")

    total_chars_before = 0
    total_chars_after = 0

    for txt_file in txt_files:
        logger.info(f"Processing {txt_file.name}...")

        # read file
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()

        total_chars_before += len(text)

        # clean text
        cleaned_text = clean_text(text)
        total_chars_after += len(cleaned_text)

        # write cleaned file
        output_file = output_path / txt_file.name
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        logger.info(f"  Saved to {output_file}")
        logger.info(f"  Size: {len(text):,} → {len(cleaned_text):,} chars")

    # summary
    logger.info(f"\nCleaning complete:")
    logger.info(f"  Files processed: {len(txt_files)}")
    logger.info(f"  Total chars: {total_chars_before:,} → {total_chars_after:,}")
    logger.info(f"  Output directory: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean the Verdict dataset by removing formatting artifacts"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/the_verdict",
        help="Input directory containing .txt files (default: data/raw/the_verdict)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/clean/the_verdict",
        help="Output directory for cleaned files (default: data/clean/the_verdict)"
    )

    args = parser.parse_args()

    process_verdict_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
