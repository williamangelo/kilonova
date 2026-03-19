"""Training runner."""

from __future__ import annotations

import logging
import math
import time

import torch

from kilonova.data import create_dataloaders
from kilonova.train.config import TrainConfig, detect_compute_dtype
from models.architectures import get_model_config, get_architecture_class

logger = logging.getLogger(__name__)


def run_training(config: TrainConfig) -> None:
    """Execute the full training pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    torch.manual_seed(42)
    device = config.device

    if device.type == "cuda":
        torch.cuda.manual_seed(42)
        torch.set_float32_matmul_precision("high")

    compute_dtype, dtype_reason = detect_compute_dtype(device)
    logger.info(f"Device: {device} | dtype: {compute_dtype} ({dtype_reason})")

    # build model
    model_config = get_model_config(config.model)
    arch_name = model_config.pop("architecture")
    model = get_architecture_class(arch_name)(model_config)
    model_config["architecture"] = arch_name

    total_params = sum(p.numel() for p in model.parameters())
    param_str = f"{total_params/1e9:.1f}B" if total_params >= 1e9 else f"{total_params/1e6:.0f}M"
    logger.info(
        f"Model: {config.model} ({param_str} params, "
        f"{model_config['emb_dim']}d, {model_config['n_layers']}L, {model_config['n_heads']}H)"
    )

    model.to(device)
    logger.info("Compiling model with torch.compile")
    compiled_model = torch.compile(model)

    # data
    train_loader, val_loader = create_dataloaders(
        data_dir=str(config.data_dir),
        batch_size=config.batch_size,
        max_length=model_config["context_length"],
        data_fraction=config.data_fraction,
    )
    train_iter = iter(train_loader)

    # optimizer
    optimizer = torch.optim.AdamW(
        compiled_model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.1,
        fused=(device.type == "cuda"),
    )

    # lr schedule: linear warmup + cosine decay to 10% of peak
    num_iterations = config.num_iterations
    warmup_steps = max(1, int(0.1 * num_iterations))
    min_lr = config.learning_rate * 0.1

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return config.learning_rate * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, num_iterations - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (config.learning_rate - min_lr) * cosine

    logger.info(f"Training: {num_iterations} steps | warmup: {warmup_steps} | min_lr: {min_lr:.2e}")
    if config.eval_every > 0:
        logger.info(f"Eval every {config.eval_every} steps")

    # checkpoint dir
    checkpoint_dir = None
    if config.run_dir:
        checkpoint_dir = config.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    use_autocast = compute_dtype != torch.float32
    smooth_loss = 0.0
    best_val_loss = float("inf")

    for step in range(num_iterations + 1):
        last_step = step == num_iterations

        # eval
        should_eval = config.eval_every > 0 and (last_step or (step > 0 and step % config.eval_every == 0))
        if should_eval:
            compiled_model.eval()
            total_loss, count = 0.0, 0
            with torch.no_grad():
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
                if checkpoint_dir:
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "config": model_config,
                        "model_name": config.model,
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

        # get next batch, wrapping around
        try:
            input_batch, target_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_batch, target_batch = next(train_iter)

        input_batch, target_batch = input_batch.to(device), target_batch.to(device)

        t0 = time.time()
        with torch.amp.autocast(device.type, dtype=compute_dtype, enabled=use_autocast):
            logits = compiled_model(input_batch)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(compiled_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_f = loss.item()
        dt = time.time() - t0

        smooth_loss = 0.9 * smooth_loss + 0.1 * loss_f
        debiased_loss = smooth_loss / (1 - 0.9 ** (step + 1))

        if step % 10 == 0:
            tok_per_sec = input_batch.numel() / dt
            logger.info(
                f"Step {step:05d}/{num_iterations} | loss: {debiased_loss:.4f} | "
                f"lr: {lr:.2e} | dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,.0f}"
            )

    logger.info(f"Training complete | best val: {best_val_loss:.4f}")
