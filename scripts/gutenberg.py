"""
Gutenberg dataset preparation.

Downloads Project Gutenberg (pg19) from HuggingFace, cleans the text,
and tokenizes to binary format for training.

Usage:
    uv run scripts/gutenberg.py [--skip-download] [--skip-clean] [--train-ratio 0.85]
"""

import argparse
import logging
import re
from pathlib import Path

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

    # export cleaned HF dataset to .txt files for preprocessing
    txt_dir = CLEAN_DIR / "txt"
    if not txt_dir.exists():
        logger.info("Exporting cleaned dataset to .txt files...")
        dataset = load_from_disk(str(CLEAN_DIR))
        txt_dir.mkdir(parents=True, exist_ok=True)
        for i, example in enumerate(tqdm(dataset, desc="Exporting")):
            text = example.get("text", "")
            if text.strip():
                (txt_dir / f"doc_{i:06d}.txt").write_text(text, encoding="utf-8")
        logger.info(f"Exported {i+1} documents to {txt_dir}")

    from kilonova.preprocessing import preprocess_dataset
    preprocess_dataset(
        input_dir=str(txt_dir),
        output_dir=str(PROCESSED_DIR),
        tokenizer_name="gpt2",
        train_ratio=train_ratio,
    )


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
