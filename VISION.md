# Vision

## What Osmium Is

Osmium is a unified CLI for training language models from scratch. It provides the complete pipeline — data acquisition, cleaning, tokenization, training, evaluation, and generation — as a single tool with consistent conventions.

## Who It's For

Practitioners who want to understand and control the full LLM training stack. Osmium prioritizes transparency over convenience: every stage is explicit, every default is documented, and the implementation is readable.

## Principles

- **Unified CLI**: One tool for the entire pipeline. No glue scripts, no notebook-to-script translation.
- **Convention over configuration**: Sensible directory layouts and defaults that work without config files. Override when needed.
- **Educational clarity**: Implementations should be readable and self-contained. Prefer PyTorch primitives over framework abstractions.
- **From-scratch training**: Not fine-tuning, not adapters. Full pretraining from random initialization.

## Non-Goals

<!-- fill in non-goals that matter to you. some candidates: -->
<!-- - production serving / inference optimization -->
<!-- - fine-tuning or adapter methods (LoRA, etc.) -->
<!-- - multi-node distributed training -->
<!-- - support for non-transformer architectures -->

## Direction

<!-- what's next for osmium? some areas the codebase is positioned for: -->
<!-- - new architectures beyond GPT-2 -->
<!-- - larger-scale training (distributed, gradient checkpointing) -->
<!-- - better evaluation (perplexity benchmarks, downstream tasks) -->
<!-- - dataset tooling (mixing, filtering, deduplication) -->
