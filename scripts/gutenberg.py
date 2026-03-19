"""
Gutenberg dataset preparation.

Downloads Project Gutenberg (pg19) from HuggingFace, cleans the text,
and tokenizes to binary format for training.

Usage:
    uv run scripts/gutenberg.py [--skip-download] [--skip-clean] [--train-ratio 0.85]
"""

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset, load_from_disk
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/gutenberg")
CLEAN_DIR = Path("data/clean/gutenberg")
PROCESSED_DIR = Path("data/processed/gutenberg")
HF_SOURCE = "pg19"


def download():
    """download gutenberg (pg19) from huggingface"""
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        logger.info(f"Raw data already exists at {RAW_DIR}, skipping download")
        return

    logger.info(f"Downloading {HF_SOURCE} from HuggingFace...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(HF_SOURCE, split="train")
    logger.info(f"Downloaded {len(dataset)} items, fields: {dataset.column_names}")

    dataset.save_to_disk(str(RAW_DIR))
    logger.info(f"Saved to {RAW_DIR}")


def clean():
    """clean gutenberg text: normalize whitespace and line endings"""
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_DIR}. Run download first.")

    logger.info(f"Loading dataset from {RAW_DIR}")
    dataset = load_from_disk(str(RAW_DIR))

    def clean_text(example):
        text = example.get("text", "")

        # normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # strip trailing whitespace per line
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        # collapse 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # normalize multiple spaces
        text = re.sub(r" {2,}", " ", text)

        example["text"] = text
        return example

    logger.info("Cleaning text...")
    cleaned = dataset.map(clean_text, desc="Cleaning")

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.save_to_disk(str(CLEAN_DIR))
    logger.info(f"Cleaned dataset saved to {CLEAN_DIR}")


def tokenize(train_ratio: float = 0.85):
    """tokenize cleaned text to binary format"""
    if not CLEAN_DIR.exists():
        raise FileNotFoundError(f"Clean data not found at {CLEAN_DIR}. Run clean first.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    dtype_name = "uint16" if vocab_size <= 65535 else "uint32"

    logger.info(f"Tokenizer: gpt2 (vocab: {vocab_size}), dtype: {dtype_name}")
    logger.info(f"Train/val split: {train_ratio:.0%}/{1-train_ratio:.0%}")

    # load cleaned dataset
    dataset = load_from_disk(str(CLEAN_DIR))

    # first pass: count tokens
    logger.info("Pass 1/2: counting tokens...")
    total_tokens = 0
    for example in tqdm(dataset, desc="Counting"):
        text = example.get("text", "")
        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        total_tokens += len(tokens) + 1  # +1 for EOT

    logger.info(f"Total tokens: {total_tokens:,}")
    split_idx = int(train_ratio * total_tokens)
    logger.info(f"Train: {split_idx:,}, Val: {total_tokens - split_idx:,}")

    # second pass: write binary files
    logger.info("Pass 2/2: writing tokenized data...")
    train_path = PROCESSED_DIR / "train.bin"
    val_path = PROCESSED_DIR / "val.bin"

    train_tokens = 0
    val_tokens = 0
    tokens_written = 0

    with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
        for example in tqdm(dataset, desc="Writing"):
            text = example.get("text", "")
            tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            tokens.append(tokenizer.eot_token)

            if tokens_written + len(tokens) <= split_idx:
                arr = np.array(tokens, dtype=dtype)
                arr.tofile(f_train)
                train_tokens += len(arr)
                tokens_written += len(arr)
            elif tokens_written >= split_idx:
                arr = np.array(tokens, dtype=dtype)
                arr.tofile(f_val)
                val_tokens += len(arr)
                tokens_written += len(arr)
            else:
                # split this document
                remaining_train = split_idx - tokens_written
                train_arr = np.array(tokens[:remaining_train], dtype=dtype)
                train_arr.tofile(f_train)
                train_tokens += len(train_arr)

                val_arr = np.array(tokens[remaining_train:], dtype=dtype)
                val_arr.tofile(f_val)
                val_tokens += len(val_arr)

                tokens_written += len(tokens)

    # write metadata
    meta = {
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "total_tokens": train_tokens + val_tokens,
        "vocab_size": vocab_size,
        "tokenizer": "gpt2",
        "dtype": dtype_name,
        "train_ratio": train_ratio,
        "source": HF_SOURCE,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    with open(PROCESSED_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    train_mb = train_path.stat().st_size / (1024**2)
    val_mb = val_path.stat().st_size / (1024**2)
    logger.info(f"train.bin: {train_mb:.1f} MB, val.bin: {val_mb:.1f} MB")
    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Prepare Gutenberg dataset for training")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step")
    parser.add_argument("--skip-clean", action="store_true", help="Skip cleaning step")
    parser.add_argument("--train-ratio", type=float, default=0.85, help="Train/val split ratio")
    args = parser.parse_args()

    if not args.skip_download:
        download()
    if not args.skip_clean:
        clean()
    tokenize(train_ratio=args.train_ratio)


if __name__ == "__main__":
    main()
