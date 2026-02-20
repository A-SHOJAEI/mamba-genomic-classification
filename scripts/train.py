#!/usr/bin/env python3
"""Train Mamba, CNN, and Transformer classifiers on genomic tasks."""

import sys
sys.path.insert(0, ".")

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from src.mamba_genomics.data.dataset import load_task
from src.mamba_genomics.models.mamba_classifier import MambaClassifier
from src.mamba_genomics.models.cnn_classifier import CNNClassifier
from src.mamba_genomics.models.transformer_classifier import TransformerClassifier
from src.mamba_genomics.utils.config import load_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_model(model_type, num_classes, config):
    """Build model from config."""
    if model_type == "mamba":
        cfg = config["models"]["mamba"]
        return MambaClassifier(
            vocab_size=config["data"]["vocab_size"],
            num_classes=num_classes,
            d_model=cfg["d_model"],
            n_layer=cfg["n_layer"],
            d_state=cfg["d_state"],
            d_conv=cfg["d_conv"],
            expand=cfg["expand"],
            dropout=cfg["dropout"],
        )
    elif model_type == "cnn":
        cfg = config["models"]["cnn"]
        return CNNClassifier(
            vocab_size=config["data"]["vocab_size"],
            num_classes=num_classes,
            channels=cfg["channels"],
            kernel_sizes=cfg["kernel_sizes"],
            dropout=cfg["dropout"],
        )
    elif model_type == "transformer":
        cfg = config["models"]["transformer"]
        return TransformerClassifier(
            vocab_size=config["data"]["vocab_size"],
            num_classes=num_classes,
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
            max_seq_len=config["data"]["max_seq_len"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with autocast(dtype=torch.bfloat16):
            logits = model(input_ids)

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    mcc = matthews_corrcoef(all_labels, all_preds)

    return {"accuracy": acc, "f1": f1, "mcc": mcc}


def train_one_task(model_type, task_name, config, device):
    """Train and evaluate a model on a single task."""
    train_cfg = config["training"]
    data_cfg = config["data"]

    logger.info(f"  Loading task: {task_name}")
    train_ds, test_ds, num_classes = load_task(
        task_name,
        max_seq_len=data_cfg["max_seq_len"],
        dataset_name=data_cfg["dataset_name"],
    )
    logger.info(f"    Train: {len(train_ds)}, Test: {len(test_ds)}, Classes: {num_classes}")

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg["batch_size"] * 2, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    model = build_model(model_type, num_classes, config).to(device)
    n_params = count_params(model)

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    total_steps = len(train_loader) * train_cfg["epochs"]
    warmup_steps = max(1, int(total_steps * train_cfg["warmup_ratio"]))

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_metrics = None
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate every 2 epochs or at the last epoch
        if (epoch + 1) % 2 == 0 or epoch == train_cfg["epochs"] - 1:
            metrics = evaluate(model, test_loader, device)
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_metrics = metrics.copy()

            logger.info(
                f"    Epoch {epoch+1:3d}/{train_cfg['epochs']} | "
                f"Loss: {avg_loss:.4f} | "
                f"Acc: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"MCC: {metrics['mcc']:.4f}"
            )

    elapsed = time.time() - start_time
    best_metrics["train_time"] = elapsed
    best_metrics["n_params"] = n_params
    best_metrics["task"] = task_name

    return best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=["mamba", "cnn", "transformer", "all"], default="all")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Specific tasks to run (default: all from config)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tasks = args.tasks or config["data"]["tasks"]
    models_to_run = ["mamba", "cnn", "transformer"] if args.model == "all" else [args.model]

    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["logs_dir"]).mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_type in models_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_type.upper()}")
        logger.info(f"{'='*60}")

        model_results = {}
        for task_name in tasks:
            try:
                metrics = train_one_task(model_type, task_name, config, device)
                model_results[task_name] = metrics
                logger.info(
                    f"  >> {task_name}: Acc={metrics['accuracy']:.4f}, "
                    f"F1={metrics['f1']:.4f}, MCC={metrics['mcc']:.4f}, "
                    f"Time={metrics['train_time']:.0f}s"
                )
            except Exception as e:
                logger.error(f"  FAILED on {task_name}: {e}")
                model_results[task_name] = {"error": str(e)}

            torch.cuda.empty_cache()

        all_results[model_type] = model_results

        # Save per-model results
        with open(results_dir / f"{model_type}_results.json", "w") as f:
            json.dump(model_results, f, indent=2)

    # Save combined results
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Task':<25} {'Metric':<8} " + " ".join(f"{m:>12}" for m in models_to_run))
    logger.info("-" * (35 + 13 * len(models_to_run)))

    for task_name in tasks:
        accs = []
        for model_type in models_to_run:
            r = all_results.get(model_type, {}).get(task_name, {})
            accs.append(r.get("accuracy", 0))
        acc_strs = " ".join(f"{a:>12.4f}" for a in accs)
        logger.info(f"{task_name:<25} {'Acc':<8} {acc_strs}")

    # Average across tasks
    avg_strs = []
    for model_type in models_to_run:
        results = all_results.get(model_type, {})
        valid = [r["accuracy"] for r in results.values() if "accuracy" in r]
        avg = np.mean(valid) if valid else 0
        avg_strs.append(f"{avg:>12.4f}")
    logger.info("-" * (35 + 13 * len(models_to_run)))
    logger.info(f"{'AVERAGE':<25} {'Acc':<8} {' '.join(avg_strs)}")


if __name__ == "__main__":
    main()
