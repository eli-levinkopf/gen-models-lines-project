"""
evaluate.py — Evaluation & Visualization for the Lines Model

Produces all the evidence needed for the project report:
    1. Speedup comparison:   Teacher (100 DDIM steps) vs Student (1 step)
    2. Visual quality:       Side-by-side grid of Teacher vs Student outputs
    3. Trajectory validation: Student predictions at t=0.0, 0.2, 0.5, 0.8, 1.0
    4. MSE metric:           Quantitative error on held-out pairs

All outputs are saved to a `results/` directory as PNG images and a JSON
summary.

Usage:
    conda activate gen_models
    python evaluate.py [--checkpoint checkpoints/student_final.pt]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
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


def to_image_range(x: torch.Tensor) -> torch.Tensor:
    """Clamp from [-1, 1] to [0, 1] for saving as image."""
    return (x.clamp(-1, 1) + 1) / 2


def _save_labeled_grid(
    rows: list[torch.Tensor],
    row_labels: list[str],
    col_labels: list[str] | None,
    output_path: Path,
    title: str = "",
) -> None:
    """Save a labeled image grid using matplotlib.

    Args:
        rows: List of tensors, each (N, 3, H, W) — one tensor per row.
        row_labels: Label for each row (left side).
        col_labels: Optional label for each column (top).
        output_path: Where to save.
        title: Figure suptitle.
    """
    nrows = len(rows)
    ncols = rows[0].shape[0]

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 1.3, nrows * 1.5 + (0.6 if title else 0)),
        squeeze=False,
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)

    for r, (row_imgs, label) in enumerate(zip(rows, row_labels)):
        for c in range(ncols):
            ax = axes[r, c]
            img = row_imgs[c].permute(1, 2, 0).numpy()  # (H, W, 3)
            ax.imshow(img)
            ax.axis("off")
            # Column labels on top of first row
            if r == 0 and col_labels and c < len(col_labels):
                ax.set_title(col_labels[c], fontsize=9)

    # Row labels — placed via fig.text so they aren't hidden by axis("off")
    plt.tight_layout()
    for r, label in enumerate(row_labels):
        # Get the vertical center of the row from the first cell's position
        bbox = axes[r, 0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(0.01, y_center, label, fontsize=10, fontweight="bold",
                 va="center", ha="left")

    # Add left margin for row labels
    fig.subplots_adjust(left=0.18)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def teacher_generate_ddim(
    unet: torch.nn.Module,
    scheduler,
    noise: torch.Tensor,
    device: torch.device,
    num_steps: int = 100,
) -> torch.Tensor:
    """Run DDIM reverse process (same as data_gen.py)."""
    from diffusers import DDIMScheduler

    scheduler.set_timesteps(num_steps)
    image = noise.clone().to(device)
    timesteps_device = scheduler.timesteps.to(device)
    timesteps_int = scheduler.timesteps.tolist()

    for t_dev, t_int in zip(timesteps_device, timesteps_int):
        t_input = t_dev.expand(image.shape[0])
        model_output = unet(image, t_input).sample
        image = scheduler.step(model_output, t_int, image).prev_sample
        del model_output

    result = image.cpu()
    del image
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


# ──────────────────────────────────────────────────────────────
#  1. Speedup Benchmark
# ──────────────────────────────────────────────────────────────

def benchmark_speedup(
    student: torch.nn.Module,
    unet: torch.nn.Module,
    scheduler,
    device: torch.device,
    num_images: int = 16,
    num_steps: int = 100,
) -> dict:
    """Compare wall-clock time: Teacher (DDIM) vs Student (1 step)."""
    print("\n" + "=" * 60)
    print("  BENCHMARK: Teacher vs Student Speed")
    print("=" * 60)

    noise = torch.randn(num_images, 3, 32, 32)

    # ── Teacher timing ──
    # Warm-up run
    _ = teacher_generate_ddim(unet, scheduler, noise[:1], device, num_steps)

    t0 = time.time()
    teacher_out = teacher_generate_ddim(unet, scheduler, noise, device, num_steps)
    teacher_time = time.time() - t0

    # ── Student timing ──
    student.eval()
    t_zero = torch.zeros(num_images, device=device)  # t=0 → pure noise input
    noise_dev = noise.to(device)

    # Warm-up
    _ = student(noise_dev[:1], t_zero[:1])
    if device.type == "mps":
        torch.mps.synchronize()

    t0 = time.time()
    student_out = student(noise_dev, t_zero)
    if device.type == "mps":
        torch.mps.synchronize()
    student_time = time.time() - t0

    speedup = teacher_time / student_time

    print(f"\n  Teacher ({num_steps} DDIM steps, {num_images} images): {teacher_time:.3f}s")
    print(f"  Student (1 step, {num_images} images):                  {student_time:.3f}s")
    print(f"  Speedup: {speedup:.1f}×\n")

    return {
        "teacher_time_s": round(teacher_time, 3),
        "student_time_s": round(student_time, 3),
        "speedup": round(speedup, 1),
        "num_images": num_images,
        "teacher_steps": num_steps,
    }


# ──────────────────────────────────────────────────────────────
#  2. Visual Quality (Side-by-Side)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_comparison_grid(
    student: torch.nn.Module,
    unet: torch.nn.Module,
    scheduler,
    device: torch.device,
    num_images: int = 8,
    num_steps: int = 100,
    output_path: Path = Path("results/comparison_grid.png"),
) -> None:
    """Generate labeled grid: row 1 = Teacher, row 2 = Student."""
    print("Generating comparison grid ...")

    noise = torch.randn(num_images, 3, 32, 32)

    # Teacher output
    teacher_out = teacher_generate_ddim(unet, scheduler, noise, device, num_steps)

    # Student output (1 step, t=0 → noise input)
    student.eval()
    t_zero = torch.zeros(num_images, device=device)
    student_out = student(noise.to(device), t_zero).cpu()

    teacher_imgs = to_image_range(teacher_out)
    student_imgs = to_image_range(student_out)

    row_labels = [f"Teacher\n({num_steps} DDIM steps)", "Student (1 step)"]
    rows = [teacher_imgs, student_imgs]
    _save_labeled_grid(rows, row_labels, None, output_path,
                       title="Teacher vs Student Comparison")
    print(f"  → Saved: {output_path}\n")


# ──────────────────────────────────────────────────────────────
#  3. Trajectory Validation
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def visualize_trajectory(
    student: torch.nn.Module,
    data_path: Path,
    device: torch.device,
    num_samples: int = 4,
    t_values: list[float] | None = None,
    output_path: Path = Path("results/trajectory.png"),
) -> None:
    """
    For a few (N, C) pairs, show two rows per sample:
        Row 1 (Input):  X_t = t*C + (1-t)*N  at each t   → proves the formula
        Row 2 (Output): Student(X_t, t)       at each t   → should all ≈ C
    Last column is Ground-Truth C for reference.
    """
    if t_values is None:
        t_values = [0.0, 0.2, 0.5, 0.8, 1.0]

    print("Generating trajectory visualization ...")

    data = torch.load(data_path, weights_only=True)
    noise = data["noise"][:num_samples]   # (K, 3, 32, 32)
    clean = data["clean"][:num_samples]

    student.eval()
    all_rows = []
    row_labels = []

    for i in range(num_samples):
        input_row = []   # X_t inputs at each t
        pred_row = []    # Student predictions at each t
        n_i = noise[i:i+1]   # (1, 3, 32, 32)
        c_i = clean[i:i+1]

        for t_val in t_values:
            t_tensor = torch.tensor([t_val], device=device)
            t_view = t_tensor[:, None, None, None]
            x_t = (t_view * c_i.to(device) + (1.0 - t_view) * n_i.to(device))

            input_row.append(to_image_range(x_t.cpu()))

            # Student prediction from this X_t
            pred = student(x_t, t_tensor).cpu()
            pred_row.append(to_image_range(pred))

        # Ground-truth clean as last column (same for both rows)
        input_row.append(to_image_range(c_i))
        pred_row.append(to_image_range(c_i))

        all_rows.append(torch.cat(input_row, dim=0))  # (ncols, 3, 32, 32)
        all_rows.append(torch.cat(pred_row, dim=0))

        row_labels.append(f"#{i+1} Input Xt")
        row_labels.append(f"#{i+1} Prediction")

    col_labels = [f"t={t}" for t in t_values] + ["Ground Truth"]
    _save_labeled_grid(all_rows, row_labels, col_labels, output_path,
                       title="Trajectory: Input X_t vs Student Prediction at each t")
    print(f"  → Saved: {output_path}")
    print(f"    For each sample: top row = input X_t, bottom row = student prediction")
    print(f"    The student should recover the same image regardless of t.\n")


# ──────────────────────────────────────────────────────────────
#  4. MSE Metric (Quantitative)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_mse_metric(
    student: torch.nn.Module,
    data_path: Path,
    device: torch.device,
    num_eval: int = 5000,
    batch_size: int = 256,
) -> dict:
    """
    Compute MSE between Student(N, t=0) and Teacher's C on held-out pairs.
    Uses the last `num_eval` pairs from the dataset as a hold-out set.
    """
    print("Computing MSE metric on held-out set ...")

    data = torch.load(data_path, weights_only=True)
    noise = data["noise"][-num_eval:]   # last N pairs as hold-out
    clean = data["clean"][-num_eval:]

    student.eval()
    total_mse = 0.0
    count = 0

    for i in tqdm(range(0, num_eval, batch_size), desc="  MSE eval", leave=False):
        n_batch = noise[i:i+batch_size].to(device)
        c_batch = clean[i:i+batch_size].to(device)
        t_zero = torch.zeros(n_batch.shape[0], device=device)

        pred = student(n_batch, t_zero)
        mse = F.mse_loss(pred, c_batch, reduction="sum").item()
        total_mse += mse
        count += n_batch.shape[0]

    avg_mse = total_mse / count
    # Per-pixel MSE (divide by C×H×W = 3×32×32)
    per_pixel_mse = avg_mse / (3 * 32 * 32)

    print(f"  Hold-out MSE (per sample):  {avg_mse:.5f}")
    print(f"  Hold-out MSE (per pixel):   {per_pixel_mse:.7f}")
    print(f"  RMSE (per pixel, [-1,1]):    {per_pixel_mse**0.5:.5f}\n")

    return {
        "num_eval": num_eval,
        "mse_per_sample": round(avg_mse, 5),
        "mse_per_pixel": round(per_pixel_mse, 7),
        "rmse_per_pixel": round(per_pixel_mse**0.5, 5),
    }


# ──────────────────────────────────────────────────────────────
#  5. Random Student Samples (uncurated)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_random_samples(
    student: torch.nn.Module,
    device: torch.device,
    num_images: int = 64,
    output_path: Path = Path("results/student_samples.png"),
    label: str = "Student",
) -> None:
    """Generate a grid of random images from the student (1-step)."""
    print(f"Generating random {label.lower()} samples ...")
    student.eval()

    noise = torch.randn(num_images, 3, 32, 32, device=device)
    t_zero = torch.zeros(num_images, device=device)
    samples = student(noise, t_zero)

    imgs = to_image_range(samples.cpu())  # (N, 3, 32, 32)
    nrow = 8
    nrows_grid = (num_images + nrow - 1) // nrow

    fig, axes = plt.subplots(nrows_grid, nrow, figsize=(nrow * 1.2, nrows_grid * 1.2))
    fig.suptitle(f"{label} — {num_images} Random 1-Step Generations", fontsize=14, fontweight="bold")
    for idx in range(nrows_grid * nrow):
        ax = axes[idx // nrow, idx % nrow] if nrows_grid > 1 else axes[idx % nrow]
        if idx < num_images:
            ax.imshow(imgs[idx].permute(1, 2, 0).numpy())
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {output_path}\n")


# ──────────────────────────────────────────────────────────────
#  6. Three-Way Comparison Grid (Lines vs Baseline vs Teacher)
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_three_way_grid(
    student: torch.nn.Module,
    baseline: torch.nn.Module,
    unet: torch.nn.Module,
    scheduler,
    device: torch.device,
    num_images: int = 8,
    num_steps: int = 100,
    output_path: Path = Path("results/lines_vs_baseline.png"),
) -> None:
    """Generate 3-row grid: Teacher (top) | Lines Student (mid) | Baseline (bot)."""
    print("Generating Lines vs Baseline vs Teacher grid ...")

    noise = torch.randn(num_images, 3, 32, 32)

    # Teacher output
    teacher_out = teacher_generate_ddim(unet, scheduler, noise, device, num_steps)

    # Lines student output
    student.eval()
    t_zero = torch.zeros(num_images, device=device)
    student_out = student(noise.to(device), t_zero).cpu()

    # Baseline output
    baseline.eval()
    baseline_out = baseline(noise.to(device), t_zero).cpu()

    teacher_imgs = to_image_range(teacher_out)
    student_imgs = to_image_range(student_out)
    baseline_imgs = to_image_range(baseline_out)

    row_labels = [f"Teacher ({num_steps} DDIM steps)", "Lines Student (1 step)", "Baseline (1 step)"]
    rows = [teacher_imgs, student_imgs, baseline_imgs]
    _save_labeled_grid(rows, row_labels, None, output_path,
                       title="Teacher vs Lines Model vs Naive Baseline")
    print(f"  → Saved: {output_path}\n")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main(
    checkpoint: str = "checkpoints/student_final.pt",
    baseline_checkpoint: str | None = None,
    data_path: str = "data/teacher_pairs.pt",
    output_dir: str = "outputs",
    teacher_model_id: str = "google/ddpm-cifar10-32",
    num_steps: int = 100,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}\n")

    # ── Load Student (Lines Model) ──
    print("Loading student model (Lines) ...")
    student = StudentUNet().to(device)
    student.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    student.eval()
    total_params = sum(p.numel() for p in student.parameters())
    print(f"  Student: {total_params:,} params loaded from {checkpoint}\n")

    # ── Load Baseline (if provided) ──
    baseline = None
    if baseline_checkpoint and Path(baseline_checkpoint).exists():
        print("Loading baseline model (Naive Distillation) ...")
        baseline = StudentUNet().to(device)
        baseline.load_state_dict(torch.load(baseline_checkpoint, weights_only=True, map_location=device))
        baseline.eval()
        print(f"  Baseline loaded from {baseline_checkpoint}\n")

    # ── Load Teacher (for comparison) ──
    print("Loading teacher model for comparison ...")
    from diffusers import DDIMScheduler, DDPMPipeline
    pipe = DDPMPipeline.from_pretrained(teacher_model_id)
    unet = pipe.unet.to(device).eval()
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    del pipe
    print(f"  Teacher: {sum(p.numel() for p in unet.parameters()):,} params\n")

    results = {}

    # ── 1. Speedup Benchmark ──
    results["speedup"] = benchmark_speedup(
        student, unet, scheduler, device, num_images=16, num_steps=num_steps
    )

    # ── 2. Visual Comparison Grid ──
    generate_comparison_grid(
        student, unet, scheduler, device,
        num_images=8, num_steps=num_steps,
        output_path=output_dir / "comparison_grid.png",
    )

    # ── 3. Trajectory Validation ──
    visualize_trajectory(
        student,
        data_path=Path(data_path),
        device=device,
        num_samples=4,
        output_path=output_dir / "trajectory.png",
    )

    # ── 4. MSE Metric ──
    results["mse"] = compute_mse_metric(
        student,
        data_path=Path(data_path),
        device=device,
        num_eval=5000,
    )

    # ── 5. Random Student Samples ──
    generate_random_samples(
        student, device,
        num_images=64,
        output_path=output_dir / "student_samples.png",
        label="Lines Student",
    )

    # ── 6. Baseline Comparison (if baseline model provided) ──
    if baseline is not None:
        print("\n" + "=" * 60)
        print("  BASELINE vs LINES MODEL COMPARISON")
        print("=" * 60)

        # Baseline MSE
        results["baseline_mse"] = compute_mse_metric(
            baseline,
            data_path=Path(data_path),
            device=device,
            num_eval=5000,
        )

        # Baseline random samples
        generate_random_samples(
            baseline, device,
            num_images=64,
            output_path=output_dir / "baseline_samples.png",
            label="Baseline (Naive)",
        )

        # Side-by-side: Lines vs Baseline vs Teacher
        generate_three_way_grid(
            student, baseline, unet, scheduler, device,
            num_images=8, num_steps=num_steps,
            output_path=output_dir / "lines_vs_baseline.png",
        )

        # Summary comparison
        lines_mse = results["mse"]["mse_per_sample"]
        base_mse = results["baseline_mse"]["mse_per_sample"]
        improvement = ((base_mse - lines_mse) / base_mse) * 100
        results["comparison"] = {
            "lines_mse": lines_mse,
            "baseline_mse": base_mse,
            "improvement_pct": round(improvement, 2),
            "lines_wins": lines_mse < base_mse,
        }
        print(f"\n  Lines MSE:    {lines_mse:.5f}")
        print(f"  Baseline MSE: {base_mse:.5f}")
        print(f"  Improvement:  {improvement:.2f}%")
        print(f"  Lines Model {'WINS' if lines_mse < base_mse else 'LOSES'} ✓\n")

    # ── Save summary ──
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation summary → {summary_path}")
    print("\nDone. Check the results/ folder for all visualizations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained Lines Model student.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/student_final.pt",
                        help="Path to student weights.")
    parser.add_argument("--baseline_checkpoint", type=str, default=None,
                        help="Path to baseline model weights (optional, for comparison).")
    parser.add_argument("--data_path", type=str, default="data/teacher_pairs.pt",
                        help="Path to teacher pairs for trajectory / MSE eval.")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory for output images and metrics.")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="DDIM steps for teacher comparison (default: 100).")

    args = parser.parse_args()
    main(
        checkpoint=args.checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
    )
