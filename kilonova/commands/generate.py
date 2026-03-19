"""Generate command implementation."""

from __future__ import annotations

from pathlib import Path

import click
import torch
import tiktoken

from osmium.utils import PathResolver
from osmium.train.config import resolve_device
from models.architectures import get_architecture_class


def load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    """load a trained model from checkpoint

    args:
        checkpoint_path: path to model checkpoint file
        device: torch device to load model on

    returns:
        tuple of (model, config, tokenizer)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # strip _orig_mod. prefix if present (from torch.compile)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # extract architecture and instantiate dynamically
    # backward compatibility: default to "gpt2" for old checkpoints
    arch_name = config.pop("architecture", "gpt2")
    ArchClass = get_architecture_class(arch_name)
    model = ArchClass(config)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    # restore architecture field
    config["architecture"] = arch_name

    return model, config, tokenizer


def generate_text(
    model,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    max_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    config: dict | None = None,
    device: torch.device | None = None,
) -> str:
    """generate text from a prompt with various sampling strategies

    args:
        model: model instance
        tokenizer: tiktoken tokenizer
        prompt: starting text prompt
        max_tokens: number of tokens to generate
        temperature: sampling temperature (0.0 = greedy, higher = more random)
        top_k: top-k sampling (None = disabled)
        top_p: top-p (nucleus) sampling (None = disabled)
        config: model configuration dict
        device: torch device

    returns:
        generated text string (prompt + completion)
    """
    if device is None:
        device = next(model.parameters()).device

    if config is None:
        raise ValueError("config must be provided")

    encoded = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = encoded_tensor[:, -config["context_length"]:]
            logits = model(idx_cond)
            logits = logits[:, -1, :]

            # apply top-k filtering
            if top_k is not None and top_k > 0:
                top_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf")).to(logits.device),
                    logits
                )

            # apply top-p (nucleus) sampling
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            # sample next token
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            encoded_tensor = torch.cat((encoded_tensor, idx_next), dim=1)

    # decode full sequence
    full_sequence = encoded_tensor.squeeze(0).tolist()
    generated_text = tokenizer.decode(full_sequence)

    return generated_text


def resolve_checkpoint(model: str) -> Path:
    """resolve model name or path to checkpoint path"""
    resolver = PathResolver()
    model_path = Path(model)

    if model_path.exists():
        return model_path

    # assume it's a run name
    run_dir = resolver.run_dir(model)
    if not run_dir.exists():
        raise click.ClickException(
            f"Run '{model}' not found at {run_dir}\n"
            f"Run 'osmium list models' to see available runs."
        )

    # use best.pth checkpoint if available, otherwise latest
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise click.ClickException(
            f"No checkpoints found for run '{model}' at {checkpoint_dir}"
        )

    best_checkpoint = checkpoint_dir / "best.pth"
    if best_checkpoint.exists():
        return best_checkpoint

    # find latest checkpoint
    checkpoints = sorted(checkpoint_dir.glob("epoch-*.pth"))
    if not checkpoints:
        raise click.ClickException(f"No checkpoints found in {checkpoint_dir}")

    return checkpoints[-1]


def interactive_loop(
    model,
    tokenizer: tiktoken.Encoding,
    config: dict,
    device: torch.device,
    temp: float,
    max_tokens: int,
    top_k: int | None,
    top_p: float | None,
) -> None:
    """run interactive generation loop"""
    click.echo("\nInteractive mode. Type your prompt (Ctrl+C to exit, 'exit' to quit)\n")

    while True:
        try:
            prompt = click.prompt(">>>", prompt_suffix=" ")
        except (EOFError, KeyboardInterrupt):
            click.echo("\nExiting.")
            break

        if prompt.lower() in ("exit", "quit"):
            click.echo("Exiting.")
            break

        if not prompt.strip():
            continue

        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            config=config,
            device=device,
        )
        click.echo(f"\n{text}\n")


def generate_cmd(
    model: str,
    prompt: str | None,
    interactive: bool,
    temp: float,
    max_tokens: int,
    top_k: int | None,
    top_p: float | None,
    device: str,
) -> None:
    """generate text from a trained model

    args:
        model: model name (run name) or path to checkpoint
        prompt: generation prompt (None for interactive mode)
        interactive: force interactive mode
        temp: sampling temperature
        max_tokens: maximum tokens to generate
        top_k: top-k sampling
        top_p: nucleus sampling
        device: device for generation (auto, cuda, cpu)
    """
    # resolve checkpoint path
    checkpoint_path = resolve_checkpoint(model)

    # resolve device
    torch_device = resolve_device(device)

    click.echo(f"Loading model: {checkpoint_path}")
    click.echo(f"Device: {torch_device}")

    model_instance, config, tokenizer = load_model(checkpoint_path, torch_device)
    click.echo("Model loaded\n")

    # determine mode: interactive if --interactive flag or no prompt provided
    if interactive or prompt is None:
        interactive_loop(
            model=model_instance,
            tokenizer=tokenizer,
            config=config,
            device=torch_device,
            temp=temp,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
        )
    else:
        # one-off generation
        text = generate_text(
            model=model_instance,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            config=config,
            device=torch_device,
        )
        click.echo(text)


__all__ = ["generate_cmd"]
