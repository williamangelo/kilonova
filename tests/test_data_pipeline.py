"""Tests for data preprocessing and loading pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kilonova.data import TokenDataset, create_dataloaders
from kilonova.preprocessing import preprocess_dataset
import tiktoken


def _create_bin_file(path: Path, tokens: list[int], dtype: str = "uint16"):
    """helper: write tokens to a .bin file with metadata.json."""
    np_dtype = np.uint16 if dtype == "uint16" else np.uint32
    arr = np.array(tokens, dtype=np_dtype)
    bin_path = path / "train.bin"
    arr.tofile(bin_path)

    meta = {
        "train_tokens": len(tokens),
        "val_tokens": 0,
        "total_tokens": len(tokens),
        "vocab_size": 50257,
        "tokenizer": "gpt2",
        "dtype": dtype,
        "train_ratio": 1.0,
        "source_files": 1,
        "source_dir": str(path),
        "created": "2026-01-01T00:00:00+00:00",
    }
    with open(path / "metadata.json", "w") as f:
        json.dump(meta, f)

    return bin_path


class TestTokenDatasetNumSamples:
    """test num_samples calculation for various token/length/stride combos."""

    def test_exact_fit(self, tmp_path):
        """11 tokens, max_length=5, stride=5 → 2 samples (each needs 6 tokens)."""
        bin_path = _create_bin_file(tmp_path, list(range(11)))
        ds = TokenDataset(bin_path, max_length=5)
        assert len(ds) == 2

    def test_tokens_left_over(self, tmp_path):
        """14 tokens, max_length=5, stride=5 → 2 samples (remainder too small)."""
        bin_path = _create_bin_file(tmp_path, list(range(14)))
        ds = TokenDataset(bin_path, max_length=5)
        assert len(ds) == 2

    def test_stride_less_than_max_length(self, tmp_path):
        """11 tokens, max_length=5, stride=2 → 3 samples."""
        bin_path = _create_bin_file(tmp_path, list(range(11)))
        ds = TokenDataset(bin_path, max_length=5, stride=2)
        # starts: 0, 2, 4. start=4 needs tokens[4:10], valid since 10 < 11. start=6 needs tokens[6:12], invalid.
        assert len(ds) == 3

    def test_too_few_tokens(self, tmp_path):
        """5 tokens with max_length=5 → 0 samples (need 6)."""
        bin_path = _create_bin_file(tmp_path, list(range(5)))
        ds = TokenDataset(bin_path, max_length=5)
        assert len(ds) == 0

    def test_exactly_one_sample(self, tmp_path):
        """6 tokens with max_length=5 → exactly 1 sample."""
        bin_path = _create_bin_file(tmp_path, list(range(6)))
        ds = TokenDataset(bin_path, max_length=5)
        assert len(ds) == 1

    def test_max_tokens_reduces_samples(self, tmp_path):
        """100 tokens but max_tokens=11, max_length=5 → 2 samples."""
        bin_path = _create_bin_file(tmp_path, list(range(100)))
        ds = TokenDataset(bin_path, max_length=5, max_tokens=11)
        assert len(ds) == 2


class TestTokenDatasetGetItem:
    """test __getitem__ slicing and shift logic."""

    def test_target_shifted_by_one(self, tmp_path):
        """target should be input shifted right by 1."""
        tokens = list(range(20))
        bin_path = _create_bin_file(tmp_path, tokens)
        ds = TokenDataset(bin_path, max_length=5)

        inp, tgt = ds[0]
        assert inp.tolist() == [0, 1, 2, 3, 4]
        assert tgt.tolist() == [1, 2, 3, 4, 5]

    def test_second_sample_offset_by_stride(self, tmp_path):
        """second sample starts at stride offset."""
        tokens = list(range(20))
        bin_path = _create_bin_file(tmp_path, tokens)
        ds = TokenDataset(bin_path, max_length=5)  # stride defaults to 5

        inp, tgt = ds[1]
        assert inp.tolist() == [5, 6, 7, 8, 9]
        assert tgt.tolist() == [6, 7, 8, 9, 10]

    def test_overlapping_stride(self, tmp_path):
        """with stride=2, consecutive samples overlap by max_length-stride tokens."""
        tokens = list(range(20))
        bin_path = _create_bin_file(tmp_path, tokens)
        ds = TokenDataset(bin_path, max_length=5, stride=2)

        inp0, _ = ds[0]
        inp1, _ = ds[1]
        # sample 0: [0,1,2,3,4], sample 1: [2,3,4,5,6] — overlap is [2,3,4]
        assert inp0.tolist()[2:] == inp1.tolist()[:3]


def _create_train_val_bins(path: Path, train_tokens: list[int], val_tokens: list[int], dtype: str = "uint16"):
    """helper: write train.bin, val.bin, and metadata.json."""
    np_dtype = np.uint16 if dtype == "uint16" else np.uint32

    train_arr = np.array(train_tokens, dtype=np_dtype)
    (path / "train.bin").write_bytes(train_arr.tobytes())

    val_arr = np.array(val_tokens, dtype=np_dtype)
    (path / "val.bin").write_bytes(val_arr.tobytes())

    meta = {
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "total_tokens": len(train_tokens) + len(val_tokens),
        "vocab_size": 50257,
        "tokenizer": "gpt2",
        "dtype": dtype,
        "train_ratio": 0.85,
        "source_files": 1,
        "source_dir": str(path),
        "created": "2026-01-01T00:00:00+00:00",
    }
    with open(path / "metadata.json", "w") as f:
        json.dump(meta, f)


class TestCreateDataloaders:
    """test budget splitting logic in create_dataloaders."""

    def test_data_fraction_applied_independently(self, tmp_path):
        """data_fraction should apply independently to train and val."""
        train_toks = list(range(1000))
        val_toks = list(range(200))
        _create_train_val_bins(tmp_path, train_toks, val_toks)

        train_loader, val_loader = create_dataloaders(
            data_dir=tmp_path,
            batch_size=1,
            max_length=10,
            data_fraction=0.5,
        )

        # data_fraction=0.5 → train uses 500 tokens, val uses 100 tokens
        # train: (500 - 10 - 1) // 10 + 1 = 49
        # val: (100 - 10 - 1) // 10 + 1 = 9
        assert len(train_loader.dataset) == 49
        assert len(val_loader.dataset) == 9

    def test_max_tokens_split_by_ratio(self, tmp_path):
        """max_tokens should split by original train/val ratio."""
        train_toks = list(range(850))
        val_toks = list(range(150))
        _create_train_val_bins(tmp_path, train_toks, val_toks)

        train_loader, val_loader = create_dataloaders(
            data_dir=tmp_path,
            batch_size=1,
            max_length=10,
            max_tokens=100,
        )

        # total=1000, train_ratio=850/1000=0.85, val_ratio=0.15
        # train budget: int(100 * 0.85) = 85 tokens → (85 - 10 - 1) // 10 + 1 = 8
        # val budget: int(100 * 0.15) = 15 tokens → (15 - 10 - 1) // 10 + 1 = 1
        assert len(train_loader.dataset) == 8
        assert len(val_loader.dataset) == 1


class TestPreprocessDataset:
    """test preprocessing pipeline."""

    def _write_txt_files(self, path: Path, contents: list[str]):
        """helper: write numbered .txt files."""
        for i, text in enumerate(contents):
            (path / f"doc_{i:03d}.txt").write_text(text, encoding="utf-8")

    def test_token_counts_match_metadata(self, tmp_path):
        """bin file sizes should match metadata token counts."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        self._write_txt_files(input_dir, [
            "Hello world.",
            "Second document here.",
            "Third one.",
        ])

        meta = preprocess_dataset(
            str(input_dir), str(output_dir), train_ratio=0.7
        )

        train_bin = np.memmap(output_dir / "train.bin", dtype=np.uint16, mode='r')
        val_bin = np.memmap(output_dir / "val.bin", dtype=np.uint16, mode='r')

        assert len(train_bin) == meta["train_tokens"]
        assert len(val_bin) == meta["val_tokens"]
        assert meta["train_tokens"] + meta["val_tokens"] == meta["total_tokens"]

    def test_doc_boundaries_match_actual_positions(self, tmp_path):
        """doc_starts arrays should mark where each document begins in the bin."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        self._write_txt_files(input_dir, [
            "Hello world.",
            "Second document here.",
            "Third one.",
        ])

        meta = preprocess_dataset(
            str(input_dir), str(output_dir), train_ratio=0.7
        )

        assert meta["has_doc_boundaries"] is True

        train_starts = np.load(output_dir / "train_doc_starts.npy")
        val_starts = np.load(output_dir / "val_doc_starts.npy")

        # verify first doc starts at 0 in train
        assert train_starts[0] == 0

        # verify each start aligns with where the previous doc's EOT was
        tokenizer = tiktoken.get_encoding("gpt2")
        train_bin = np.memmap(output_dir / "train.bin", dtype=np.uint16, mode='r')
        for i in range(1, len(train_starts)):
            # token before each doc start should be EOT
            assert train_bin[train_starts[i] - 1] == tokenizer.eot_token

    def test_single_file_input(self, tmp_path):
        """preprocessing a single .txt file should work."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        (input_dir / "solo.txt").write_text("Just one file.", encoding="utf-8")

        meta = preprocess_dataset(str(input_dir), str(output_dir))

        assert meta["source_files"] == 1
        assert meta["total_tokens"] > 0
        assert (output_dir / "train.bin").exists()
        assert (output_dir / "val.bin").exists()
        assert meta["has_doc_boundaries"] is True
        assert (output_dir / "train_doc_starts.npy").exists()
        assert (output_dir / "val_doc_starts.npy").exists()

    def test_split_doc_tracked_in_metadata(self, tmp_path):
        """when a doc straddles the split, split_doc_index should be set."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # create docs with very different sizes so the split lands inside the big one
        self._write_txt_files(input_dir, [
            "Tiny.",
            "A " * 500,  # large doc likely to be split
            "Small end.",
        ])

        meta = preprocess_dataset(
            str(input_dir), str(output_dir), train_ratio=0.5
        )

        # with 50% split on a dataset dominated by doc_001, it should be split
        assert meta["split_doc_index"] is not None

    def test_no_split_yields_null_split_doc_index(self, tmp_path):
        """when all tokens fit in train, split_doc_index is null."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # train_ratio=1.0 means everything goes to train — no split ever occurs
        (input_dir / "doc.txt").write_text("Hello.", encoding="utf-8")

        meta = preprocess_dataset(
            str(input_dir), str(output_dir), train_ratio=1.0
        )

        # no file was ever split, so split_doc_index must be None
        assert meta["split_doc_index"] is None
        # all tokens go to train, val is empty
        assert meta["val_tokens"] == 0
        assert meta["train_tokens"] == meta["total_tokens"]

    def test_empty_file_produces_eot_only(self, tmp_path):
        """empty .txt file should produce a single EOT token."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        (input_dir / "empty.txt").write_text("", encoding="utf-8")
        (input_dir / "nonempty.txt").write_text("Hello world.", encoding="utf-8")

        # train_ratio=1.0 guarantees both docs land in train
        meta = preprocess_dataset(str(input_dir), str(output_dir), train_ratio=1.0)

        train_starts = np.load(output_dir / "train_doc_starts.npy")
        assert len(train_starts) == 2
        # empty file (doc_000) contributes exactly 1 EOT token, so second doc starts at 1
        assert train_starts[0] == 0
        assert train_starts[1] == 1
