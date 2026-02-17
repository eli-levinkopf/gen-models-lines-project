"""
train.py — Training Loop for the Lines Model (Student Distillation)

Implements the "Lines" / Rectified Flow training:
    1. Load pre-generated (N, C) pairs from data/teacher_pairs.pt
    2. For each batch:
       - Sample random t ~ Uniform(0, 1)
       - Compute the mixed state:  X_t = t * C + (1-t) * N
       - Student predicts clean image:  Pred = Student(X_t, t)
       - Loss = MSE(Pred, C)
    3. The student learns to map ANY point on the straight line N→C
       directly to C, enabling 1-step generation at inference.

Usage:
    conda activate gen_models
    python train.py [--epochs 50] [--batch_size 256] [--lr 1e-4]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import StudentUNet


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_teacher_pairs(data_path: Path, num_pairs: int | None = None) -> TensorDataset:
    """Load the cached (Noise, Clean) pairs into a TensorDataset."""
    print(f"Loading teacher pairs from {data_path} ...")
    data = torch.load(data_path, weights_only=True)
    noise = data["noise"]   # (N, 3, 32, 32)
    clean = data["clean"]   # (N, 3, 32, 32)
    if num_pairs is not None and num_pairs < noise.shape[0]:
        noise = noise[:num_pairs]
        clean = clean[:num_pairs]
        print(f"  Using {num_pairs} of {data['noise'].shape[0]} pairs (subset).")
    print(f"  Loaded {noise.shape[0]} pairs.  "
          f"noise {noise.shape}, clean {clean.shape}\n")
    return TensorDataset(noise, clean)


# ──────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────

def train(
    data_path: str | Path = "data/teacher_pairs.pt",
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-4,
    save_dir: str | Path = "checkpoints",
    save_every: int = 10,
    log_every: int = 50,
    baseline: bool = False,
    num_pairs: int | None = None,
) -> None:
    data_path = Path(data_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    mode_name = "Baseline (Naive Distillation)" if baseline else "Lines Model"
    device = get_device()
    print(f"Using device: {device}")
    print(f"Training mode: {mode_name}\n")

    # ── Data ──
    dataset = load_teacher_pairs(data_path, num_pairs=num_pairs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )
    num_batches = len(loader)

    # ── Model ──
    model = StudentUNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Student U-Net: {total_params:,} params ({total_params/1e6:.1f}M)\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * num_batches
    )
    criterion = nn.MSELoss()

    # ── Training log ──
    history: list[dict] = []

    # ── Loop ──
    print("Starting training ...\n")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t_start = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for step, (noise, clean) in enumerate(pbar, 1):
            noise = noise.to(device)  # N
            clean = clean.to(device)  # C

            # ── Lines Model core logic ──
            if baseline:
                # Baseline: always t=0, student sees pure noise, predicts C
                t = torch.zeros(noise.shape[0], device=device)   # (B,)
                x_t = noise  # pure noise, no interpolation
            else:
                # Lines Model: sample mixing factor t ~ Uniform(0, 1)
                t = torch.rand(noise.shape[0], device=device)    # (B,)

                # Mixed state:  X_t = t * C + (1-t) * N
                #   At t=0 → pure noise,  at t=1 → clean image
                t_view = t[:, None, None, None]                  # (B,1,1,1)
                x_t = t_view * clean + (1.0 - t_view) * noise   # (B,3,32,32)

            # ── Forward pass ──
            pred = model(x_t, t) # Student predicts C from (X_t, t)

            # ── Loss: how well does the student recover C? ──
            loss = criterion(pred, clean)

            # ── Backward ──
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            if step % log_every == 0 or step == num_batches:
                pbar.set_postfix(loss=f"{loss.item():.5f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / num_batches
        elapsed = time.time() - t_start
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{epochs}  |  "
              f"loss = {avg_loss:.5f}  |  "
              f"lr = {current_lr:.2e}  |  "
              f"time = {elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "lr": current_lr,
            "time_s": round(elapsed, 1),
        })

        # ── Checkpoint ──
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = save_dir / f"student_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "avg_loss": avg_loss,
            }, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path}")

    # ── Save final model (weights only) ──
    final_path = save_dir / "student_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete.  Final weights → {final_path}")

    # ── Save training log ──
    log_path = save_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training log → {log_path}")


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Student U-Net via Lines Model distillation."
    )
    parser.add_argument("--data_path", type=str, default="data/teacher_pairs.pt",
                        help="Path to teacher_pairs.pt from data_gen.py.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50).")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size (default: 256).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4).")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory for checkpoints (default: checkpoints/).")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs (default: 10).")
    parser.add_argument("--baseline", action="store_true",
                        help="Train naive baseline (always t=0, no interpolation). "
                             "Saves to checkpoints_baseline/ by default.")
    parser.add_argument("--num_pairs", type=int, default=None,
                        help="Use only the first N pairs (for data-efficiency experiments).")

    args = parser.parse_args()

    # Default save_dir changes for baseline mode
    save_dir = args.save_dir
    if args.baseline and save_dir == "checkpoints":
        save_dir = "checkpoints_baseline"

    train(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=save_dir,
        save_every=args.save_every,
        baseline=args.baseline,
        num_pairs=args.num_pairs,
    )
