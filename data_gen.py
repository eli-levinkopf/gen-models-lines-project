"""
data_gen.py — Teacher Data Generation for the Lines Model

Generates (Noise, Clean) pairs by running a pre-trained DDPM teacher
(google/ddpm-cifar10-32) to create deterministic couplings:
    N ~ N(0, I)  →  C = Teacher(N)

Pairs are saved incrementally in chunks to disk so that partial progress
is preserved if the process is interrupted.  A final consolidated file
`data/teacher_pairs.pt` is written at the end.

Usage:
    conda activate gen_models
    python data_gen.py [--num_samples 50000] [--batch_size 128] [--num_steps 100] [--data_dir data]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, DDPMPipeline
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device (MPS for Apple Silicon)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def teacher_generate(
    unet: torch.nn.Module,
    scheduler,
    noise: torch.Tensor,
    device: torch.device,
    batch_idx: int = 0,
    num_batches: int = 1,
) -> torch.Tensor:
    """
    Run the full DDPM reverse process starting from *noise*.

    Args:
        unet:       The pre-trained UNet model.
        scheduler:  DDPMScheduler with timesteps already set.
        noise:      Starting noise tensor (B, 3, 32, 32).
        device:     Torch device.
        batch_idx:  Current batch number (for progress bar label).
        num_batches: Total number of batches (for progress bar label).

    Returns:
        clean:      Generated images tensor (B, 3, 32, 32), values in [-1, 1].
    """
    B = noise.shape[0]

    # Move everything to device ONCE (avoid 1000× CPU→GPU transfers)
    image = noise.clone().to(device)
    timesteps_device = scheduler.timesteps.to(device)

    # Pre-compute integer timesteps list for scheduler.step()
    timesteps_int = scheduler.timesteps.tolist()

    num_steps = len(timesteps_device)
    pbar = tqdm(
        zip(timesteps_device, timesteps_int),
        total=num_steps,
        desc=f"  Batch {batch_idx+1}/{num_batches} ({num_steps} steps)",
        leave=False,
        unit="step",
        bar_format="{l_bar}{bar:30}{r_bar}",
    )
    for t_dev, t_int in pbar:
        t_input = t_dev.expand(B)                 
        model_output = unet(image, t_input).sample
        image = scheduler.step(model_output, t_int, image).prev_sample
        del model_output                            # free intermediate tensors

    result = image.cpu()
    del image
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


# ──────────────────────────────────────────────────────────────
#  Main generation loop
# ──────────────────────────────────────────────────────────────

def generate_pairs(
    num_samples: int = 50_000,
    batch_size: int = 128,
    num_steps: int = 100,
    data_dir: str | Path = "data",
    model_id: str = "google/ddpm-cifar10-32",
) -> None:
    data_dir = Path(data_dir)
    chunks_dir = data_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    # ── Load Teacher ──
    print(f"Loading teacher model: {model_id} ...")
    pipe = DDPMPipeline.from_pretrained(model_id)
    unet = pipe.unet.to(device).eval()

    # Use DDIM scheduler for faster generation (fewer steps)
    # DDIM can skip timesteps while maintaining quality
    if num_steps < 1000:
        scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        sampler_name = "DDIM"
    else:
        scheduler = pipe.scheduler
        sampler_name = "DDPM"
    del pipe                                 # free pipeline to reclaim memory

    scheduler.set_timesteps(num_steps)
    num_timesteps = len(scheduler.timesteps)
    print(f"Teacher loaded — {sampler_name} sampler, {num_timesteps} timesteps.\n")

    # ── Check for existing chunks (resume support) ──
    existing_chunks = sorted(chunks_dir.glob("chunk_*.pt"))
    generated_so_far = 0
    if existing_chunks:
        # each chunk stores {"noise": ..., "clean": ...} with shape (chunk_bs, ...)
        for ch in existing_chunks:
            generated_so_far += torch.load(ch, weights_only=True)["noise"].shape[0]
        print(f"Found {len(existing_chunks)} existing chunks "
              f"({generated_so_far} samples). Resuming...\n")

    remaining = num_samples - generated_so_far
    if remaining <= 0:
        print("All samples already generated. Skipping to consolidation.")
    else:
        num_batches = (remaining + batch_size - 1) // batch_size
        chunk_idx = len(existing_chunks)

        pbar = tqdm(total=remaining, desc="Generating pairs", unit="img")
        start = time.time()

        for b in range(num_batches):
            current_bs = min(batch_size, remaining - b * batch_size)

            # ── Sample noise N ~ N(0, I) ──
            noise = torch.randn(current_bs, 3, 32, 32)

            # ── Run Teacher: N → C ──
            clean = teacher_generate(
                unet, scheduler, noise, device,
                batch_idx=b, num_batches=num_batches,
            )

            # ── Save chunk immediately ──
            chunk_path = chunks_dir / f"chunk_{chunk_idx:05d}.pt"
            torch.save({"noise": noise, "clean": clean}, chunk_path)
            chunk_idx += 1

            pbar.update(current_bs)

        pbar.close()
        elapsed = time.time() - start
        print(f"\nGeneration complete — {remaining} samples in {elapsed:.1f}s "
              f"({remaining / elapsed:.1f} img/s).\n")

    # ── Consolidate chunks into one file ──
    print("Consolidating chunks into a single file ...")
    all_chunks = sorted(chunks_dir.glob("chunk_*.pt"))
    all_noise, all_clean = [], []
    for ch in tqdm(all_chunks, desc="Loading chunks"):
        data = torch.load(ch, weights_only=True)
        all_noise.append(data["noise"])
        all_clean.append(data["clean"])

    all_noise = torch.cat(all_noise, dim=0)[:num_samples]
    all_clean = torch.cat(all_clean, dim=0)[:num_samples]

    consolidated_path = data_dir / "teacher_pairs.pt"
    torch.save({"noise": all_noise, "clean": all_clean}, consolidated_path)

    size_mb = consolidated_path.stat().st_size / (1024 ** 2)
    print(f"\nSaved {all_noise.shape[0]} pairs → {consolidated_path}  ({size_mb:.1f} MB)")
    print(f"  noise shape: {all_noise.shape}")
    print(f"  clean shape: {all_clean.shape}")


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate (Noise, Clean) teacher pairs for the Lines Model."
    )
    parser.add_argument("--num_samples", type=int, default=50_000,
                        help="Total number of pairs to generate (default: 50000).")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for teacher generation (default: 128).")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Number of diffusion steps (default: 100). "
                             "Uses DDIM when < 1000, DDPM at 1000.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root directory to save data (default: data/).")
    parser.add_argument("--model_id", type=str, default="google/ddpm-cifar10-32",
                        help="HuggingFace model ID for the teacher DDPM.")

    args = parser.parse_args()
    generate_pairs(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        data_dir=args.data_dir,
        model_id=args.model_id,
    )
