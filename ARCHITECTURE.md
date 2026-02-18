# Architecture

## Component Overview

```
loaders/                   # data pipeline
├── downloading/           #   HuggingFace dataset downloads
├── cleaning/              #   dataset-specific text cleaning
│   ├── registry.py        #   maps dataset names → sources + cleaners
│   ├── base.py            #   BaseCleaner interface
│   └── generic.py         #   fallback cleaner for unregistered datasets
├── preprocessing/         #   tokenization to memory-mapped binary
│   └── tokenize.py        #   two-pass streaming: count tokens, then write train/val splits
└── dataloader/            #   training data loading
    ├── streaming.py        #   StreamingTokenDataset (numpy memmap, zero-copy)
    └── factory.py          #   create_dataloaders() with token budgets

models/                    # model system
├── architectures.py       #   @register_architecture decorator + global registry
├── registry.py            #   MODEL_REGISTRY: named configs (gpt2-small, gpt2-medium, ...)
├── loader.py              #   create_model_from_config(): bridges both registries
└── gpt2/                  #   GPT-2 implementation
    └── model.py           #   MultiHeadAttention, FeedForward, TransformerBlock, GPT2

osmium/                    # CLI orchestration
├── cli.py                 #   Click entry point, command registration
├── commands/              #   one module per command
│   ├── download.py
│   ├── clean.py
│   ├── preprocess.py
│   ├── train.py
│   ├── generate.py
│   ├── evaluate.py
│   ├── list_cmd.py
│   └── info.py
├── train/                 #   training system
│   ├── config.py          #   TrainConfig dataclass, platform-aware resolution
│   └── runner.py          #   training loop: mixed precision, grad accumulation, checkpoints
└── utils/
    ├── paths.py           #   PathResolver: convention-based directory layout
    ├── config.py          #   YAML config load/save
    └── platform.py        #   OS detection (macOS, Linux)
```

## Data Flow

```
download → clean → preprocess → train → generate/evaluate

HuggingFace    text files    train.bin     best.pth    generated
dataset     →  (cleaned)  →  val.bin    →  metrics  →  text
                              metadata
```

Each stage reads from the previous stage's output directory under `data/`:

```
data/raw/<dataset>/  →  data/clean/<dataset>/  →  data/processed/<dataset>/  →  data/runs/<name>/
```

Path conventions are enforced by [`osmium/utils/paths.py`](osmium/utils/paths.py) (`PathResolver`).

## Key Design Decisions

### Decorator-based architecture registry

[`models/architectures.py`](models/architectures.py) provides `@register_architecture("name")` which registers a model class in `ARCHITECTURE_REGISTRY` at import time. This enables zero-config extensibility: drop a new file in `models/`, decorate the class, and it's available everywhere.

### Separate model config registry

[`models/registry.py`](models/registry.py) maps variant names (e.g. `gpt2-small`) to hyperparameter dicts that reference an architecture name. This separates "what hyperparameters to use" from "how the model works." [`models/loader.py`](models/loader.py) bridges the two: it fetches the config, looks up the architecture class, and instantiates the model.

### Convention-over-config path resolution

[`osmium/utils/paths.py`](osmium/utils/paths.py) derives all data paths from a single base directory (`data/`) and dataset/run names. Commands don't need explicit path arguments for the common case — they compute paths from names. Override flags exist for non-standard layouts.

### TrainConfig resolution chain

[`osmium/train/config.py`](osmium/train/config.py) resolves training configuration in layers: hardcoded defaults → YAML file → CLI flags. Platform-aware defaults handle mixed precision (auto-enabled on CUDA, disabled on CPU/MPS), `torch.compile` (on by default), and dataloader workers (forced to 0 on macOS due to fork issues).

### Streaming token dataset

[`loaders/dataloader/streaming.py`](loaders/dataloader/streaming.py) uses numpy `memmap` for zero-copy access to tokenized data. Only accessed pages load into memory, so datasets can exceed RAM. The preprocessing step ([`loaders/preprocessing/tokenize.py`](loaders/preprocessing/tokenize.py)) uses a two-pass approach: count tokens first, then write with exact split ratios.

## Extension Points

### Adding a new architecture

1. Create `models/<name>/model.py` with your model class
2. Decorate it with `@register_architecture("<name>")`
3. Add variant configs to [`models/registry.py`](models/registry.py) referencing your architecture name
4. Import the module in [`models/__init__.py`](models/__init__.py)

### Adding a new dataset

1. Add an entry to [`loaders/cleaning/registry.py`](loaders/cleaning/registry.py) mapping the dataset name to its HuggingFace source
2. Optionally create a custom cleaner extending `BaseCleaner` in `loaders/cleaning/`
3. The generic cleaner handles basic text normalization if no custom cleaner is provided

### Adding a new CLI command

1. Create a module in `osmium/commands/`
2. Define a Click command function
3. Register it in [`osmium/cli.py`](osmium/cli.py)
