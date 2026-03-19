"""Evaluate command implementation."""

from __future__ import annotations

from pathlib import Path

import click

from kilonova.commands.generate import (
    load_model,
    generate_text,
    resolve_checkpoint,
)
from kilonova.train.config import resolve_device


# curated prompts for evaluation
EVALUATION_PROMPTS = [
    "Every effort moves you",
    "Once upon a time",
    "The future of artificial intelligence will",
]


def print_section(title: str) -> None:
    """print a section header"""
    click.echo(f"\n{title}")
    click.echo("-" * len(title))


def evaluate_model(model: str, device: str, output: Path | None, prompt: str | None) -> None:
    """run comprehensive evaluation test suite on a trained model

    args:
        model: model name (run name) or path to checkpoint
        device: device for evaluation (auto, cuda, cpu)
        output: optional output file for results
        prompt: optional test prompt for generation
    """
    # resolve checkpoint and device
    checkpoint_path = resolve_checkpoint(model)
    torch_device = resolve_device(device)

    click.echo(f"Loading model: {checkpoint_path}")
    click.echo(f"Device: {torch_device}")

    # load model
    model_instance, config, tokenizer = load_model(checkpoint_path, torch_device)
    click.echo("Model loaded")

    # collect output lines
    output_lines = []

    def log(message: str) -> None:
        """log to console and collect for file output"""
        click.echo(message)
        output_lines.append(message)

    # if custom prompt provided, run single test
    if prompt:
        print_section("Custom Prompt Evaluation")
        log(f"\nPrompt: {prompt}")
        text = generate_text(
            model=model_instance,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            config=config,
            device=torch_device,
        )
        log(text)
    else:
        # run standard test suite
        token_lengths = [15, 50, 100]
        temperatures = [0.0, 0.7, 1.2]

        # test 1: token length variation
        print_section("Test 1: Token Length Variation")
        test_prompt = EVALUATION_PROMPTS[0]

        for length in token_lengths:
            log(f"\n[{length} tokens, temp=0.7] {test_prompt}")
            text = generate_text(
                model=model_instance,
                tokenizer=tokenizer,
                prompt=test_prompt,
                max_tokens=length,
                temperature=0.7,
                top_p=0.9,
                config=config,
                device=torch_device,
            )
            log(text)

        # test 2: temperature comparison
        print_section("Test 2: Temperature Comparison")
        test_prompt = EVALUATION_PROMPTS[1]

        for temp in temperatures:
            log(f"\n[temp={temp}] {test_prompt}")
            text = generate_text(
                model=model_instance,
                tokenizer=tokenizer,
                prompt=test_prompt,
                max_tokens=50,
                temperature=temp,
                top_p=0.9,
                config=config,
                device=torch_device,
            )
            log(text)

        # test 3: sampling strategies
        print_section("Test 3: Sampling Strategies")
        test_prompt = EVALUATION_PROMPTS[2]

        strategies = [
            ("Greedy", {"temperature": 0.0}),
            ("Top-k=40", {"temperature": 0.8, "top_k": 40}),
            ("Top-p=0.9", {"temperature": 0.8, "top_p": 0.9}),
        ]

        for strategy_name, params in strategies:
            log(f"\n[{strategy_name}] {test_prompt}")
            text = generate_text(
                model=model_instance,
                tokenizer=tokenizer,
                prompt=test_prompt,
                max_tokens=50,
                config=config,
                device=torch_device,
                **params,
            )
            log(text)

    # save to file if requested
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write("\n".join(output_lines))
        click.echo(f"\nResults saved to: {output}")

    click.secho("\n✓ Evaluation complete", fg="green")


__all__ = ["evaluate_model"]
