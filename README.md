# Kilonova

**Status**: 🚀 Active development

A kilonova is what happens when two neutron stars collide in space and create an explosion of light—this repo is what happens when you train LLMs from scratch without a safety net. Inspired by [Karpathy's nanochat](https://github.com/karpathy/nanochat).

The objective: train language models and experiment with training techniques, architectures, and datasets. Keep it simple. Keep it fast.

## Quick Start

```bash
uv sync
uv run python3 -m kilonova.cli train gpt2-small --data gutenberg --name my-experiment --epochs 5
```

## Installation

```bash
uv sync
uv sync --extra dev  # includes pytest for development
```
