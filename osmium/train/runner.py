"""Training runner that drives the GPT optimization loop."""

from __future__ import annotations

import json
import logging
import math
from typing import Sequence

import torch
from tqdm import tqdm

from loaders import create_dataloaders
from models.loader import create_model_from_config
from osmium.train.config import TrainConfig


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that keeps tqdm progress bars intact."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:  # pragma: no cover - defensive
            self.handleError(record)


logger = logging.getLogger(__name__)


def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculate loss for a single batch."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches):
    """Calculate average loss over a dataloader."""
    total_loss = 0.0
    batches_processed = 0

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
        batches_processed += 1

    if batches_processed == 0:
        return float("nan")
    return total_loss / batches_processed


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Evaluate model on train and validation sets."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model_loop(model, train_loader, val_loader, optimizer, device, num_epochs,
                     eval_freq, eval_iter, patience,
                     gradient_accumulation_steps, scaler, scheduler=None, max_grad_norm=1.0,
                     checkpoint_dir=None, model_config=None, model_name=None):
    """execute the training loop with early stopping, gradient accumulation, and mixed precision."""
    train_losses, val_losses, track_tokens_seen, lrs = [], [], [], []
    tokens_seen, global_step = 0, 0
    best_val_loss = float('inf')
    steps_without_improvement = 0
    best_model_state = None

    # initial evaluation before training starts (baseline)
    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    track_tokens_seen.append(0)
    lrs.append(optimizer.param_groups[0]['lr'])
    best_val_loss = val_loss
    best_model_state = {
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'config': model_config,
        'model_name': model_name,
        'val_loss': best_val_loss,
        'epoch': 0,
        'global_step': 0,
    }
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_state, checkpoint_dir / "best.pth")
    logger.info(f"Initial eval | Train: {train_loss:.3f} | Val: {val_loss:.3f}")

    for epoch in range(num_epochs):
        model.train()

        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_idx, (input_batch, target_batch) in enumerate(pbar):
            with torch.amp.autocast(device.type, enabled=(scaler is not None)):
                loss = calc_loss_batch(input_batch, target_batch, model, device)

            loss = loss / gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

                global_step += 1

            tokens_seen += input_batch.numel()

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{(loss.item() * gradient_accumulation_steps):.3f}', 'lr': f'{current_lr:.2e}'})

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                lrs.append(current_lr)

                logger.info(
                    f"Step {global_step:05d} | Train: {train_loss:.3f} | "
                    f"Val: {val_loss:.3f} | LR: {current_lr:.2e}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    steps_without_improvement = 0
                    logger.info(f"New best val loss: {best_val_loss:.3f}")

                    # save best model state (move to cpu to reduce memory pressure)
                    best_model_state = {
                        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                        'config': model_config,
                        'model_name': model_name,
                        'val_loss': best_val_loss,
                        'epoch': epoch,
                        'global_step': global_step,
                    }

                    # save checkpoint immediately if checkpoint_dir is provided
                    if checkpoint_dir:
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        checkpoint_path = checkpoint_dir / "best.pth"
                        torch.save(best_model_state, checkpoint_path)
                        logger.info(f"Saved best checkpoint: {checkpoint_path}")
                else:
                    steps_without_improvement += 1
                    if patience is not None and steps_without_improvement >= patience:
                        logger.info(f"Early stopping after {patience} steps without improvement")
                        return train_losses, val_losses, track_tokens_seen, lrs, best_model_state

    return train_losses, val_losses, track_tokens_seen, lrs, best_model_state


def save_training_metrics(run_dir, train_losses, val_losses, tokens_seen, lrs):
    """save training metrics to logs directory"""
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # convert tensors to lists for json serialization
    def to_list(values):
        if not values:
            return []
        if isinstance(values[0], torch.Tensor):
            return [v.item() for v in values]
        return list(values)

    metrics = {
        "train_losses": to_list(train_losses),
        "val_losses": to_list(val_losses),
        "tokens_seen": to_list(tokens_seen),
        "learning_rates": to_list(lrs),
        "best_val_loss": min(val_losses) if val_losses else None,
    }

    metrics_path = logs_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Training metrics saved: {metrics_path}")


def run_training(config: TrainConfig) -> Sequence[float]:
    """Execute the full training pipeline using an already parsed config."""
    _configure_logging()

    # use TF32 for matmul on ampere+ GPUs (20-30% speedup with minimal precision loss)
    torch.set_float32_matmul_precision('high')

    torch.manual_seed(123)  # fixed seed for reproducibility

    model, model_config = create_model_from_config(config.model)

    train_loader, val_loader = create_dataloaders(
        data_dir=str(config.data_dir),
        batch_size=config.batch_size,
        max_length=model_config["context_length"],
        max_tokens=config.max_tokens,
        data_fraction=config.data_fraction,
        num_workers=config.num_workers,
    )

    total_params = sum(p.numel() for p in model.parameters())

    def format_params(params):
        if params >= 1e9:
            return f"{params/1e9:.1f}B"
        return f"{params/1e6:.0f}M"

    logger.info(
        f"Model: {config.model} ({format_params(total_params)} params, "
        f"{model_config['emb_dim']}d, {model_config['n_layers']}L, {model_config['n_heads']}H)"
    )

    device = config.device
    logger.info(f"Device: {device}")
    model.to(device)

    scaler = None
    if config.mixed_precision:
        scaler = torch.amp.GradScaler('cuda')
        logger.info("Mixed precision (fp16) enabled")

    if config.compile_model:
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)

    # weight_decay=0.1 is standard for transformer training (GPT-2, GPT-3, LLaMA)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.1,
        fused=(device.type == "cuda"),
    )

    if config.gradient_accumulation_steps > 1:
        logger.info(
            "Gradient accumulation: %d steps (effective batch size: %d)",
            config.gradient_accumulation_steps,
            config.batch_size * config.gradient_accumulation_steps,
        )

    total_steps = (len(train_loader) // config.gradient_accumulation_steps) * config.epochs
    # default warmup is 10% of training (common practice from BERT, GPT-2 papers)
    warmup_steps = config.warmup_steps if config.warmup_steps is not None else int(0.1 * total_steps)
    # default min_lr is 10% of peak lr (standard for cosine schedules)
    min_lr = config.min_lr if config.min_lr is not None else config.learning_rate * 0.1

    # linear warmup followed by cosine annealing to min_lr
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(progress * math.pi))
        return (min_lr / config.learning_rate) + (1 - min_lr / config.learning_rate) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info(
        "Learning rate scheduler: warmup=%d steps, total=%d steps, min_lr=%.2e",
        warmup_steps,
        total_steps,
        min_lr,
    )

    logger.info("Starting training")

    # prepare checkpoint directory
    checkpoint_dir = config.run_dir / "checkpoints" if config.run_dir else None

    train_losses, val_losses, tokens_seen, lrs, best_model_state = train_model_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=config.epochs,
        eval_freq=config.eval_freq,
        eval_iter=config.eval_iter,
        patience=config.patience,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        scaler=scaler,
        scheduler=scheduler,
        max_grad_norm=config.max_grad_norm,
        checkpoint_dir=checkpoint_dir,
        model_config=model_config,
        model_name=config.model,
    )

    # save metrics (checkpoint already saved during training when best found)
    if best_model_state:
        if config.run_dir:
            save_training_metrics(config.run_dir, train_losses, val_losses, tokens_seen, lrs)
    else:
        logger.warning("No best model state found, this should not happen")

    logger.info("Training complete")

    return train_losses, val_losses, tokens_seen, lrs


def _configure_logging() -> None:
    """Configure logger to play nicely with tqdm."""
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []

    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
