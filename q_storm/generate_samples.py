#!/usr/bin/env python3
"""
Standalone sample generation script for qStorm training.

Pre-generates samples to disk cache for instant loading during training.
This eliminates sampling overhead and GPU contention during training.

Usage:
    # Generate 1 million samples with default batch size
    python -m q_storm.generate_samples --n-samples 1000000
    
    # Generate 1 million samples with larger batches (more GPU strain, faster)
    python -m q_storm.generate_samples --n-samples 1000000 --batch-size 20000
    
    # Generate using CPU only (slower but no GPU required)
    python -m q_storm.generate_samples --n-samples 1000000 --cpu
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

from q_storm.StormSampler import BufferedStormSampler, DEFAULT_CACHE_DIR
from varpi.tain_varpi import VarPi

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_samples(
    n_samples: int,
    batch_size: int,
    device: torch.device,
    taus: list,
    cache_dir: str = DEFAULT_CACHE_DIR,
    use_gpu: bool = True
) -> int:
    """
    Generate and cache samples for training with incremental chunked saving.

    Samples are saved in chunks to prevent memory explosion. Each chunk
    contains up to SAMPLES_PER_CACHE_CHUNK samples (1M by default).

    Args:
        n_samples: Total number of samples to generate
        batch_size: Samples per batch (larger = more GPU strain, faster)
        device: Device to use for generation
        taus: Quantile levels
        cache_dir: Cache directory path
        use_gpu: Whether to use GPU for generation

    Returns:
        Total number of samples generated
    """
    from q_storm.StormSampler import SAMPLES_PER_CACHE_CHUNK

    # Initialize sampler
    gen_device = device if use_gpu else torch.device('cpu')

    tqdm.write(f"[Generate] Initializing sampler on {gen_device}...")
    sampler = BufferedStormSampler(
        device=gen_device,
        taus=taus,
        cache_dir=cache_dir,
        use_cache=True
    )

    # Check existing cache (discover chunks without loading)
    cache_info = sampler._load_from_cache(batch_size)
    existing_samples = 0
    if cache_info is not None:
        chunk_paths, existing_samples = cache_info
        tqdm.write(
            f"[Generate] Found existing cache: {len(chunk_paths)} chunk(s) ({existing_samples:,} samples)"
        )

    # Calculate how many more samples we need
    samples_to_generate = max(0, n_samples - existing_samples)

    if samples_to_generate == 0:
        tqdm.write(
            f"[Generate] Cache already has {existing_samples:,} samples (requested {n_samples:,}). Nothing to do."
        )
        return 0

    tqdm.write(
        f"[Generate] Generating {samples_to_generate:,} additional samples "
        f"(batch size: {batch_size:,})..."
    )
    tqdm.write(f"[Generate] Samples will be saved in chunks of {SAMPLES_PER_CACHE_CHUNK:,} samples")

    # Generate and save samples incrementally in chunks
    total_generated = 0
    current_chunk = []
    current_chunk_samples = 0
    n_batches_needed = (samples_to_generate + batch_size - 1) // batch_size  # Ceiling division

    start_time = time.time()
    with tqdm(total=samples_to_generate, desc="Generating samples", unit="samples") as pbar:
        for _ in range(n_batches_needed):
            batch = sampler._sample_batch(batch_size)
            n_batch_samples = len(batch["S_prime"])

            if n_batch_samples > 0:
                # Move to CPU for storage
                cpu_batch = {k: v.cpu() for k, v in batch.items()}
                current_chunk.append(cpu_batch)
                current_chunk_samples += n_batch_samples
                total_generated += n_batch_samples

                pbar.update(n_batch_samples)
                pbar.set_postfix(
                    {"chunk_samples": current_chunk_samples,
                     "rate": f"{total_generated/(time.time()-start_time):.0f}/s"})

                # Save chunk when it reaches the size limit
                if current_chunk_samples >= SAMPLES_PER_CACHE_CHUNK:
                    tqdm.write(f"[Generate] Saving chunk ({current_chunk_samples:,} samples)...")
                    sampler._save_to_cache(current_chunk, batch_size)
                    current_chunk = []
                    current_chunk_samples = 0

                # Stop if we've generated enough
                if total_generated >= samples_to_generate:
                    break
            else:
                tqdm.write("[Generate] Warning: Empty batch generated, retrying...")

    elapsed = time.time() - start_time

    # Save final chunk if not empty
    if current_chunk:
        tqdm.write(f"[Generate] Saving final chunk ({current_chunk_samples:,} samples)...")
        sampler._save_to_cache(current_chunk, batch_size)

    # Summary
    total_samples = existing_samples + total_generated
    tqdm.write("")
    tqdm.write("=" * 60)
    tqdm.write("SAMPLE GENERATION COMPLETE")
    tqdm.write("=" * 60)
    tqdm.write(f"  Samples generated:     {total_generated:,}")
    tqdm.write(f"  Total samples cached:  {total_samples:,}")
    tqdm.write(f"  Batch size used:       {batch_size:,}")
    tqdm.write(f"  Generation time:       {elapsed:.1f} seconds")
    tqdm.write(f"  Rate:                  {total_generated/elapsed:.1f} samples/sec")
    tqdm.write(f"  Cache location:        {cache_dir}")
    tqdm.write("=" * 60)

    return total_generated


def main():
    parser = argparse.ArgumentParser(
        description='Pre-generate samples for qStorm training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1 million samples with default batch size (10000)
  python -m q_storm.generate_samples --n-samples 1000000
  
  # Generate 1 million samples with larger batches (more GPU strain, faster)
  python -m q_storm.generate_samples --n-samples 1000000 --batch-size 20000
  
  # Generate using CPU only (slower but no GPU required)
  python -m q_storm.generate_samples --n-samples 1000000 --cpu
"""
    )

    # Core parameters
    parser.add_argument(
        '--n-samples', type=int, required=True,
        help='Total number of samples to generate'
    )
    parser.add_argument(
        '--batch-size', type=int, default=10000,
        help='Samples per batch (larger = more GPU strain, faster generation, default: 10000)'
    )

    # Device options
    parser.add_argument(
        '--cpu', action='store_true',
        help='Use CPU for generation (slower but no GPU required)'
    )

    # Cache options
    parser.add_argument(
        '--cache-dir', type=str, default=DEFAULT_CACHE_DIR,
        help=f'Cache directory (default: {DEFAULT_CACHE_DIR})'
    )
    parser.add_argument(
        '--clear-cache', action='store_true',
        help='Clear existing cache before generating'
    )

    # Config
    parser.add_argument(
        '--config', type=str, default='config.json',
        help='Path to config file (default: config.json)'
    )

    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        taus = config["general"]["quantiles"]
    except FileNotFoundError:
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)
    except KeyError:
        logger.error("Config file missing 'general.quantiles' key")
        sys.exit(1)

    # Setup device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tqdm.write(f"[Generate] Device: {device}")
    tqdm.write(f"[Generate] Total samples: {args.n_samples:,}")
    tqdm.write(f"[Generate] Batch size: {args.batch_size:,}")

    # Clear cache if requested
    if args.clear_cache:
        cache_dir = Path(args.cache_dir)
        if cache_dir.exists():
            for f in cache_dir.glob("buffer_*.pt"):
                f.unlink()
                tqdm.write(f"[Generate] Deleted: {f}")

    # Generate samples
    generate_samples(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        device=device,
        taus=taus,
        cache_dir=args.cache_dir,
        use_gpu=not args.cpu
    )


if __name__ == "__main__":
    main()
