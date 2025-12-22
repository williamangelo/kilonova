"""
Tokenization utilities for converting text files to binary token format.

Converts a directory of .txt files into memory-mapped binary files
containing pre-tokenized data for efficient training.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tiktoken
from tqdm import tqdm


logger = logging.getLogger(__name__)


def _setup_preprocessing(
    input_dir: str,
    output_dir: str,
    tokenizer_name: str,
    num_files: int | None,
):
    """Setup common preprocessing configuration.

    Args:
        input_dir: Directory containing .txt files or path to single .txt file
        output_dir: Directory to write output files
        tokenizer_name: Name of tiktoken tokenizer
        num_files: Optional limit on number of files to process

    Returns:
        Tuple of (input_path, output_path, tokenizer, dtype, dtype_name, txt_files)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = tiktoken.get_encoding(tokenizer_name)
    vocab_size = tokenizer.n_vocab

    # Validate dtype choice
    if vocab_size > 65535:
        dtype = np.uint32
        dtype_name = "uint32"
    else:
        dtype = np.uint16
        dtype_name = "uint16"

    # Find text files
    if input_path.is_file():
        txt_files = [input_path]
    else:
        txt_files = sorted(input_path.glob("*.txt"))

    if not txt_files:
        raise ValueError(f"No .txt files found in {input_dir}")

    if num_files is not None:
        txt_files = txt_files[:num_files]

    return input_path, output_path, tokenizer, dtype, dtype_name, txt_files


def _tokenize_file(file_path: Path, tokenizer) -> list[int]:
    """Tokenize a single text file.

    Args:
        file_path: Path to .txt file
        tokenizer: tiktoken tokenizer instance

    Returns:
        List of token IDs including end-of-text marker
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    tokens.append(tokenizer.eot_token)

    return tokens


def _write_metadata(
    output_path: Path,
    train_tokens: int,
    val_tokens: int,
    vocab_size: int,
    tokenizer_name: str,
    dtype_name: str,
    train_ratio: float,
    num_files: int,
    input_path: Path,
) -> dict:
    """Write metadata JSON file.

    Args:
        output_path: Directory to write metadata.json
        train_tokens: Number of training tokens
        val_tokens: Number of validation tokens
        vocab_size: Tokenizer vocabulary size
        tokenizer_name: Name of tokenizer
        dtype_name: NumPy dtype name
        train_ratio: Training split ratio
        num_files: Number of source files
        input_path: Input directory path

    Returns:
        Dictionary with metadata
    """
    total_tokens = train_tokens + val_tokens

    meta = {
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "tokenizer": tokenizer_name,
        "dtype": dtype_name,
        "train_ratio": train_ratio,
        "source_files": num_files,
        "source_dir": str(input_path.absolute()),
        "created": datetime.now(timezone.utc).isoformat(),
    }

    meta_path = output_path / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Metadata written to {meta_path}")
    return meta


def _log_summary(train_path: Path, val_path: Path, meta_path: Path):
    """Log summary of generated files.

    Args:
        train_path: Path to train.bin
        val_path: Path to val.bin
        meta_path: Path to metadata.json
    """
    train_size_mb = train_path.stat().st_size / (1024**2)
    val_size_mb = val_path.stat().st_size / (1024**2)

    logger.info("Output files:")
    logger.info(f"  {train_path}: {train_size_mb:.1f} MB")
    logger.info(f"  {val_path}: {val_size_mb:.1f} MB")
    logger.info(f"  {meta_path}")


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    tokenizer_name: str = "gpt2",
    train_ratio: float = 0.85,
    num_files: int | None = None,
) -> dict:
    """Convert text files to pre-tokenized binary format.

    Uses a memory-efficient two-pass streaming approach:
    - First pass: count total tokens across all files
    - Second pass: write tokens to train.bin/val.bin with accurate token-based split

    This provides both memory efficiency and accurate train/val splitting.

    Args:
        input_dir: Directory containing .txt files or path to single .txt file
        output_dir: Directory to write output files
        tokenizer_name: Name of tiktoken tokenizer (default: gpt2)
        train_ratio: Fraction of tokens for training (default: 0.85)
        num_files: Optional limit on number of files to process

    Returns:
        Dictionary with metadata about the processed dataset
    """
    input_path, output_path, tokenizer, dtype, dtype_name, txt_files = _setup_preprocessing(
        input_dir, output_dir, tokenizer_name, num_files
    )

    total_size = sum(f.stat().st_size for f in txt_files)
    total_size_gb = total_size / (1024**3)

    logger.info(f"Processing {len(txt_files)} files ({total_size_gb:.2f} GB)")
    logger.info(f"Tokenizer: {tokenizer_name} (vocab size: {tokenizer.n_vocab})")
    logger.info(f"Output dtype: {dtype_name}")
    logger.info(f"Train/val split: {train_ratio:.0%}/{1-train_ratio:.0%}")

    # first pass: count total tokens
    logger.info("Counting tokens...")
    total_tokens = 0
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Pass 1/2") as pbar:
        for file_path in txt_files:
            file_size = file_path.stat().st_size
            tokens = _tokenize_file(file_path, tokenizer)
            total_tokens += len(tokens)
            del tokens
            pbar.update(file_size)

    logger.info(f"Total tokens: {total_tokens:,}")

    # calculate split index based on token count
    split_idx = int(train_ratio * total_tokens)
    logger.info(f"Train tokens: {split_idx:,}")
    logger.info(f"Val tokens: {total_tokens - split_idx:,}")

    # second pass: write tokens to disk with accurate split
    train_path = output_path / "train.bin"
    val_path = output_path / "val.bin"

    train_tokens = 0
    val_tokens = 0
    tokens_written = 0

    logger.info("Writing tokenized data...")
    with open(train_path, 'wb') as f_train, open(val_path, 'wb') as f_val:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Pass 2/2") as pbar:
            for file_path in txt_files:
                file_size = file_path.stat().st_size
                tokens = _tokenize_file(file_path, tokenizer)

                if tokens_written + len(tokens) <= split_idx:
                    # all tokens go to train
                    arr = np.array(tokens, dtype=dtype)
                    arr.tofile(f_train)
                    train_tokens += len(arr)
                    tokens_written += len(arr)
                    del arr
                elif tokens_written >= split_idx:
                    # all tokens go to val
                    arr = np.array(tokens, dtype=dtype)
                    arr.tofile(f_val)
                    val_tokens += len(arr)
                    tokens_written += len(arr)
                    del arr
                else:
                    # need to split this file between train and val
                    remaining_train = split_idx - tokens_written
                    train_arr = np.array(tokens[:remaining_train], dtype=dtype)
                    train_arr.tofile(f_train)
                    train_tokens += len(train_arr)
                    del train_arr

                    val_arr = np.array(tokens[remaining_train:], dtype=dtype)
                    val_arr.tofile(f_val)
                    val_tokens += len(val_arr)
                    del val_arr

                    tokens_written += len(tokens)

                del tokens
                pbar.update(file_size)

    # write metadata
    meta = _write_metadata(
        output_path,
        train_tokens,
        val_tokens,
        tokenizer.n_vocab,
        tokenizer_name,
        dtype_name,
        train_ratio,
        len(txt_files),
        input_path,
    )

    # log summary
    _log_summary(train_path, val_path, output_path / "metadata.json")

    return meta
