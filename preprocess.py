"""
Preprocess text data into tokenized binary format for efficient training.

Usage:
    uv run python preprocess.py --input data/gutenberg/clean --output data/gutenberg/processed

Uses a memory-efficient two-pass streaming approach suitable for datasets of any size.
"""

import argparse
import logging

from loaders.preprocessing import preprocess_dataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess text files into tokenized binary format for training"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing .txt files or path to a single .txt file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed .bin files"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer name (default: gpt2)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.85,
        help="Fraction of data for training (default: 0.85)"
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=None,
        help="Limit number of files to process (optional)"
    )

    args = parser.parse_args()

    preprocess_dataset(
        input_dir=args.input,
        output_dir=args.output,
        tokenizer_name=args.tokenizer,
        train_ratio=args.train_ratio,
        num_files=args.num_files,
    )

    logging.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
