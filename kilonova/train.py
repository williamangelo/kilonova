"""Training runner."""

from __future__ import annotations

import gc
import logging
import math
import time
import uuid
from datetime import datetime
from pathlib import Path

import torch

from kilonova.data import create_dataloaders
from models.architectures import get_model_config, get_architecture_class

logger = logging.getLogger(__name__)


def resolve_device(device_spec: str) -> torch.device:
    """Resolve the requested device string into a torch.device."""
    if device_spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_spec == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but torch.cuda.is_available() is False.")

    return torch.device(device_spec)


def detect_compute_dtype(device: torch.device) -> tuple[torch.dtype, str]:
    """auto-detect optimal compute dtype based on GPU capability.

    bf16 on Ampere+ (SM 80+), fp32 otherwise.
    """
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"CUDA SM {capability[0]}{capability[1]} (bf16)"
        return torch.float32, f"CUDA SM {capability[0]}{capability[1]} (pre-Ampere, fp32)"
    return torch.float32, f"no CUDA ({device.type})"


def train_model(
    *,
    model_name: str,
    data: str,
    device: str,
    num_iterations: int,
    batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
    data_fraction: float | None,
    eval_every: int,
) -> None:
    """Execute the full training pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # validate dataset
    data_dir = Path(data)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_dir}\n"
            f"Run the dataset preparation script first (e.g. uv run scripts/gutenberg.py)"
        )
    if not (data_dir / "train.bin").exists() or not (data_dir / "val.bin").exists():
        raise FileNotFoundError(
            f"Dataset incomplete: missing train.bin or val.bin in {data_dir}\n"
            f"Run the dataset preparation script first (e.g. uv run scripts/gutenberg.py)"
        )

    # create run directory
    date = datetime.now().strftime("%Y%m%d")
    short_id = uuid.uuid4().hex[:7]
    run_id = f"run-{date}-{short_id}"
    run_dir = Path("data/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training run: {run_id}")
    logger.info(f"Run directory: {run_dir}")

    # resolve device
    device = resolve_device(device)

    torch.manual_seed(42)

    if device.type == "cuda":
        torch.cuda.manual_seed(42)
        torch.set_float32_matmul_precision("high")

    compute_dtype, dtype_reason = detect_compute_dtype(device)
    logger.info(f"Device: {device} | dtype: {compute_dtype} ({dtype_reason})")

    # build model
    model_config = get_model_config(model_name)
    arch_name = model_config.pop("architecture")
    model = get_architecture_class(arch_name)(model_config)
    model_config["architecture"] = arch_name

    total_params = sum(p.numel() for p in model.parameters())
    param_str = f"{total_params/1e9:.1f}B" if total_params >= 1e9 else f"{total_params/1e6:.0f}M"
    logger.info(
        f"Model: {model_name} ({param_str} params, "
        f"{model_config['emb_dim']}d, {model_config['n_layers']}L, {model_config['n_heads']}H)"
    )

    model.to(device)
    # torch.compile causes memory bloat on non-cuda backends
    compiled_model = torch.compile(model) if device.type == "cuda" else model

    # data
    train_loader, val_loader = create_dataloaders(
        data_dir=str(data_dir),
        batch_size=batch_size,
        max_length=model_config["context_length"],
        data_fraction=data_fraction,
    )
    train_iter = iter(train_loader)

    # optimizer
    optimizer = torch.optim.AdamW(
        compiled_model.parameters(),
        lr=learning_rate,
        weight_decay=0.1,
        fused=(device.type == "cuda"),
    )

    # lr schedule: linear warmup + cosine decay to 10% of peak
    warmup_steps = max(1, int(0.1 * num_iterations))
    min_lr = learning_rate * 0.1

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return learning_rate * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, num_iterations - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (learning_rate - min_lr) * cosine

    effective_batch = batch_size * grad_accum_steps
    logger.info(
        f"Training: {num_iterations} steps | batch: {batch_size}x{grad_accum_steps}={effective_batch} | "
        f"warmup: {warmup_steps} | min_lr: {min_lr:.2e}"
    )
    if eval_every > 0:
        logger.info(f"Eval every {eval_every} steps")

    use_autocast = compute_dtype != torch.float32
    smooth_loss = 0.0
    best_val_loss = float("inf")

    # freeze long-lived objects out of gc, disable gc during training
    gc.collect()
    gc.freeze()
    gc.disable()

    try:
        for step in range(num_iterations + 1):
            last_step = step == num_iterations

            # eval
            should_eval = eval_every > 0 and (last_step or (step > 0 and step % eval_every == 0))
            if should_eval:
                compiled_model.eval()
                total_loss, count = 0.0, 0
                with torch.no_grad(), torch.amp.autocast(device.type, dtype=compute_dtype, enabled=use_autocast):
                    for i, (inp, tgt) in enumerate(val_loader):
                        if i >= 50:
                            break
                        inp, tgt = inp.to(device), tgt.to(device)
                        logits = compiled_model(inp)
                        total_loss += torch.nn.functional.cross_entropy(logits.flatten(0, 1), tgt.flatten()).item()
                        count += 1
                val_loss = total_loss / count if count > 0 else float("nan")
                compiled_model.train()

                lr = get_lr(min(step, num_iterations - 1))
                logger.info(f"Step {step:05d}/{num_iterations} | val_loss: {val_loss:.4f} | lr: {lr:.2e}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f"New best val loss: {best_val_loss:.4f}")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "config": model_config,
                        "model_name": model_name,
                        "val_loss": best_val_loss,
                        "step": step,
                    }, checkpoint_dir / "best.pth")
                    logger.info(f"Saved checkpoint: {checkpoint_dir / 'best.pth'}")

            if last_step:
                break

            # set lr
            lr = get_lr(step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            t0 = time.time()
            accum_loss = 0.0

            for micro_step in range(grad_accum_steps):
                try:
                    input_batch, target_batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    input_batch, target_batch = next(train_iter)

                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                with torch.amp.autocast(device.type, dtype=compute_dtype, enabled=use_autocast):
                    logits = compiled_model(input_batch)
                    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
                    loss = loss / grad_accum_steps

                loss.backward()
                accum_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(compiled_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            dt = time.time() - t0

            smooth_loss = 0.9 * smooth_loss + 0.1 * accum_loss
            debiased_loss = smooth_loss / (1 - 0.9 ** (step + 1))

            if step % 10 == 0:
                tok_per_sec = (input_batch.numel() * grad_accum_steps) / dt
                logger.info(
                    f"Step {step:05d}/{num_iterations} | loss: {debiased_loss:.4f} | "
                    f"lr: {lr:.2e} | dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,.0f}"
                )

            # periodic gc to avoid unbounded leak
            if step > 0 and step % 5000 == 0:
                gc.collect()
    finally:
        gc.enable()

    logger.info(f"Training complete: {run_id} | best val: {best_val_loss:.4f}")
