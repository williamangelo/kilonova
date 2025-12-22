# Osmium

A unified CLI for training LLMs from scratch.

## Installation

```bash
uv sync
```

For development (includes pytest):
```bash
uv sync --extra dev
```

## Quick Start

```bash
# download and prepare data
osmium download gutenberg
osmium clean gutenberg
osmium preprocess gutenberg

# train a model
osmium train gpt2-small --data gutenberg --name my-experiment --epochs 5

# generate text
osmium generate my-experiment --prompt "Once upon a time"
```

## CLI Commands

### Core Pipeline

#### `osmium download <dataset>`

Download a dataset from the registry or HuggingFace.

```bash
osmium download gutenberg                    # built-in dataset
osmium download huggingface:username/dataset # custom HuggingFace dataset
osmium download gutenberg --output custom/path
```

#### `osmium clean <dataset>`

Clean a dataset with dataset-specific or generic cleaning.

```bash
osmium clean gutenberg
osmium clean custom --input data/raw/custom --output data/clean/custom
```

#### `osmium preprocess <dataset>`

Tokenize cleaned text into binary format.

```bash
osmium preprocess gutenberg
osmium preprocess gutenberg --train-split 0.9
```

#### `osmium train <architecture>`

Train a GPT model using preprocessed data.

```bash
osmium train gpt2-small --data gutenberg --name quick-test --epochs 5
osmium train gpt2-small --data gutenberg --config configs/baseline.yaml
osmium train gpt2-medium --data gutenberg --name medium-run --epochs 10
```

Options:
- `--data` (required): Dataset name
- `--name`: Experiment name (auto-generated from config filename if using --config)
- `--config`: Load hyperparameters from YAML file
- `--resume`: Resume from latest checkpoint
- `--epochs`: Training epochs (default: 1)
- `--batch-size`: Batch size per step (default: 8)
- `--learning-rate`: Optimizer learning rate (default: 0.0004)
- `--mixed-precision/--no-mixed-precision`: Toggle fp16 mixed precision
- `--compile/--no-compile`: Toggle torch.compile for speedups
- `--device`: Device preference (auto, cuda, cpu)

Available architectures: `gpt2-small`, `gpt2-medium`, `gpt2-large`, `gpt2-xlarge`

#### `osmium evaluate <model>`

Run evaluation on a trained model.

```bash
osmium evaluate my-experiment
osmium evaluate data/runs/my-experiment/checkpoints/epoch-005.pth
```

#### `osmium generate <model>`

Generate text from a trained model.

```bash
osmium generate my-experiment                              # interactive mode
osmium generate my-experiment --prompt "Once upon a time"  # one-off generation
osmium generate my-experiment --temp 0.8 --top-p 0.9
```

Options:
- `--prompt`: Generation prompt (omit for interactive mode)
- `--interactive`: Force interactive mode
- `--temp`: Sampling temperature (default: 1.0)
- `--max-tokens`: Maximum tokens to generate (default: 50)
- `--top-k`: Top-k sampling
- `--top-p`: Nucleus sampling
- `--device`: Device for generation (auto, cuda, cpu)

### Discoverability

#### `osmium list datasets`

Show available datasets and their status.

```bash
osmium list datasets
```

#### `osmium list models`

Show training runs and checkpoints.

```bash
osmium list models
```

#### `osmium info <name>`

Display detailed information about a dataset or model.

```bash
osmium info gutenberg    # dataset info
osmium info my-experiment # model/run info
```

## Directory Structure

```
data/
├── raw/                    # downloaded datasets
│   └── gutenberg/
├── clean/                  # cleaned text files
│   └── gutenberg/
├── processed/              # tokenized binary files
│   └── gutenberg/
│       ├── train.bin
│       ├── val.bin
│       └── metadata.json
└── runs/                   # training experiments
    └── my-experiment/
        ├── config.yaml
        ├── checkpoints/
        │   ├── best.pth
        │   └── epoch-001.pth
        └── logs/
            └── metrics.json
```

## Testing

```bash
uv run pytest tests/ -v
```

## Development

The CLI uses Click for command handling. Commands are organized in `osmium/commands/`.

Adding a new dataset:
1. Add entry to `loaders/cleaning/registry.py` with HuggingFace source
2. Optionally create a custom cleaner in `loaders/cleaning/`

Adding a new architecture:
1. Add config to `models/registry.py`
