import os
import pickle
from typing import Dict, Any, Tuple, List
import torch
import numpy as np
from dataclasses import dataclass
from torch import Tensor
from utils.dist_utils import generate_smooth_pdf
from varpi.tain_varpi import VarPi
from varpi.wasserstein_min import get_best_lstm_pdf_params


class StormSampler:
    """
    Optimized single-class sampler that combines PDF processing and sampling
    for maximum performance.
    """

    def __init__(
        self,
        device: torch.device,
        taus: np.ndarray,
        min_time: int = 15,
        max_time: int = 31,
        rf_min: float = 0.0,
        rf_max: float = 0.05348,
        s_max: float = 5.0,
        noise_std: float = 0.001
    ):
        self.device = device
        self.pdf_params = get_best_lstm_pdf_params(None)
        self.taus = np.array(taus)
        self.varPi = torch.load('models/varpi.pth')

        # Cache config params as tensors where possible for speed
        self.min_time = torch.tensor(min_time, device=device)
        self.max_time = torch.tensor(max_time, device=device)
        self.rf_range = torch.tensor(rf_max - rf_min, device=device)
        self.rf_min = torch.tensor(rf_min, device=device)
        self.s_max = torch.tensor(s_max, device=device)
        self.t_max = max_time/365
        self.noise_std = noise_std

        # Load and cache quantiles
        self._cache_quantiles()

    def _cache_quantiles(self) -> None:
        """Load or generate and cache varphi quantiles"""
        if os.path.exists("varphi_quantiles.pkl"):
            with open("varphi_quantiles.pkl", "rb") as f:
                self.quantiles = pickle.load(f)
        else:
            with open("walk_dataset.pkl", "rb") as f:
                walk_dataset = pickle.load(f)
            quantiles = walk_dataset.X
            quantiles = list({tuple(arr) for arr in quantiles})
            self.quantiles = torch.stack([torch.tensor(arr) for arr in quantiles])

    @torch.no_grad()  # Optimization: disable gradients for sampling
    def _fast_sample(self, n_points: int) -> Dict[str, Tensor]:
        """Optimized single-pass sampling without class switching overhead"""
        # Initial sampling with noise
        sampled_idx = torch.randint(0, len(self.quantiles), (n_points,), device=self.device)
        sampled_quantiles = self.quantiles.to(self.device)[sampled_idx] + \
            torch.randn(n_points, self.quantiles.size(1), device=self.device) * self.noise_std

        # Batch process varphi statistics
        stats, valid_indices = [], []
        for i, quants in enumerate(sampled_quantiles):
            try:
                grid, pdf, _ = generate_smooth_pdf(
                    quants.cpu().numpy(),
                    self.taus,
                    **self.pdf_params
                )
                if not (np.isnan(pdf).any() or np.isnan(grid).any()):
                    mean = np.trapezoid(grid * pdf, grid)
                    std = np.sqrt(np.trapezoid((grid - mean)**2 * pdf, grid))
                    stats.append([std])
                    valid_indices.append(i)
            except Exception:
                continue

        # Take only valid samples
        valid_indices = valid_indices[:n_points]
        sampled_quantiles = sampled_quantiles[valid_indices]

        # Generate time samples in one go
        n_valid = len(valid_indices)
        T = torch.randint(self.min_time, self.max_time, (n_valid, 1),
                          dtype=torch.float32, device=self.device)
        varphi_t = torch.rand(n_valid, 1, device=self.device)
        varphi_T = T/30

        # Compute VarPi outputs
        varpi_quantiles = self.varPi(sampled_quantiles, varphi_t, varphi_T)

        # Process VarPi PDFs
        grids, pdfs = [], []
        valid_mask = torch.ones(n_valid, dtype=torch.bool)

        for i, quants in enumerate(varpi_quantiles):
            try:
                grid, pdf, _ = generate_smooth_pdf(
                    quants.cpu().numpy(),
                    self.taus,
                    **self.pdf_params
                )
                if np.isnan(pdf).any() or np.isnan(grid).any():
                    valid_mask[i] = False
                    continue
                grids.append(grid)
                pdfs.append(pdf)
            except Exception:
                valid_mask[i] = False

        # Convert to tensors
        varpi_grids = torch.tensor(np.array(grids), device=self.device)
        varpi_pdfs = torch.tensor(np.array(pdfs), device=self.device)
        varpi_quantiles = varpi_quantiles[valid_mask]

        # Generate rate samples efficiently
        n_final = len(varpi_quantiles)
        rf = torch.rand(n_final, 1, device=self.device) * self.rf_range + self.rf_min
        S_prime = (torch.rand(n_final, 1, device=self.device) * self.s_max)
        t_prime = (torch.rand(n_final, 1, device=self.device) * self.t_max)

        return {
            "S_prime": S_prime,
            "t_prime": t_prime,
            "rf": rf,
            "varpi_q": varpi_quantiles,
            "varpi_pdfs": varpi_pdfs,
            "varpi_grids": varpi_grids,
        }

    def sample(self, n_points: int) -> Dict[str, Tensor]:
        """
        Get exactly n_points samples with proper gradient tracking.
        """
        samples = self._fast_sample(n_points)

        # Get more samples if needed
        while len(samples["S_prime"]) < n_points:
            diff = n_points - len(samples["S_prime"])
            additional = self._fast_sample(diff * 2)
            samples = {
                key: torch.cat([samples[key], additional[key]], dim=0)[:n_points]
                for key in samples
            }

        # Enable gradients for final output
        return {
            key: value.float().requires_grad_(True)
            for key, value in samples.items()
        }


if __name__ == "__main__":
    import json
    import time

    # Test the optimized sampler
    n_samples = 1000
    with open("config.json", "r") as f:
        CONFIG = json.load(f)

    device = torch.device("cuda")
    taus = CONFIG["general"]["quantiles"]
    # Initialize sampler
    sampler = StormSampler(device, taus)

    # Benchmark sampling
    start_time = time.time()
    samples = sampler.sample(n_samples)
    end_time = time.time()

    # Print stats
    total_time = end_time - start_time
    print(f"Total samples: {n_samples}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per sample: {(total_time / n_samples):.4f} seconds")
