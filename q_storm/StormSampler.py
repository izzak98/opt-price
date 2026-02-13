import os
import pickle
import queue
import threading
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
from joblib import Parallel, delayed
from utils.dist_utils import generate_smooth_pdf
from varpi.wasserstein_min import get_best_lstm_pdf_params


# Module-level constants
DAYS_PER_YEAR = 365
DAYS_PER_MONTH = 30.0
OVERSAMPLE_RATIO = 1.5
DEFAULT_N_JOBS = -1  # Use all available cores
BUFFER_MULTIPLIER = 15  # Buffer size multiplier (15x batch size)
REFILL_THRESHOLD = 5  # Refill when buffer drops below this multiplier
REFILL_CHECK_INTERVAL = 0.1  # Seconds between buffer checks
DEFAULT_CACHE_DIR = "samples_cache"  # Default directory for sample caching
SAMPLES_PER_CACHE_CHUNK = 1_000_000  # Maximum samples per cache chunk file


def _process_single_pdf(
    quantiles: np.ndarray,
    taus: np.ndarray,
    pdf_params: dict
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Process a single quantile array to generate PDF.
    Returns (grid, pdf) tuple or None if invalid.
    """
    try:
        grid, pdf, _ = generate_smooth_pdf(quantiles, taus, **pdf_params)
        if np.isnan(pdf).any() or np.isnan(grid).any():
            return None
        return (grid, pdf)
    except Exception:
        return None


class StormSampler:
    """
    Optimized sampler using parallel PDF processing for qStorm training.

    Uses joblib for parallel CPU-bound PDF generation to significantly
    improve sampling throughput.
    """

    # Class-level constants
    LOGNORMAL_SIGMA = 1.0

    def __init__(
        self,
        device: torch.device,
        taus: np.ndarray,
        min_time: int = 15,
        max_time: int = 31,
        rf_min: float = 0.0,
        rf_max: float = 0.05348,
        s_max: float = 5.0,
        noise_std: float = 0.001,
        n_jobs: int = DEFAULT_N_JOBS
    ):
        """
        Initialize the StormSampler.

        Args:
            device: Target device for tensors (cuda/cpu)
            taus: Quantile levels array
            min_time: Minimum time horizon in days
            max_time: Maximum time horizon in days
            rf_min: Minimum risk-free rate
            rf_max: Maximum risk-free rate
            s_max: Maximum normalized stock price
            noise_std: Standard deviation for quantile noise
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.device = device
        self.pdf_params = get_best_lstm_pdf_params(None)
        self.taus = np.array(taus)
        self.n_jobs = n_jobs

        # Load VarPi model
        self.varPi = torch.load('models/varpi.pth', weights_only=False)
        self.varPi = self.varPi.to(device)
        self.varPi.eval()

        # Store configuration
        self.min_time = min_time
        self.max_time = max_time
        self.rf_min = rf_min
        self.rf_range = rf_max - rf_min
        self.s_max_val = s_max
        self.noise_std = noise_std

        # Pre-compute log-normal center
        self._lognormal_mu = np.log(s_max / 2.0)

        # Load and cache quantiles
        self._load_quantiles()

    def _load_quantiles(self) -> None:
        """Load cached quantiles from file."""
        cache_path = "varphi_quantiles.pkl"

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.quantiles = pickle.load(f)
                if isinstance(self.quantiles, torch.Tensor):
                    self.quantiles = self.quantiles.to(self.device)
            return

        # Fall back to walk_dataset.pkl
        from varpi.gen_walks import WalkDataSet

        class _WalkDataSetUnpickler(pickle.Unpickler):
            """Handle legacy pickle format where WalkDataSet was in __main__."""

            def find_class(self, module, name):
                if module == "__main__" and name == "WalkDataSet":
                    return WalkDataSet
                return super().find_class(module, name)

        with open("walk_dataset.pkl", "rb") as f:
            walk_dataset = _WalkDataSetUnpickler(f).load()

        # Deduplicate and convert to tensor
        unique_quantiles = list({tuple(arr) for arr in walk_dataset.X})
        self.quantiles = torch.stack(
            [torch.tensor(arr, dtype=torch.float32) for arr in unique_quantiles]
        ).to(self.device)

    def _parallel_pdf_processing(
        self,
        quantiles_np: np.ndarray
    ) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
        """
        Process quantiles to PDFs in parallel using joblib.

        Args:
            quantiles_np: Array of shape (n_samples, n_quantiles)

        Returns:
            Tuple of (valid_indices, grids, pdfs)
        """
        # Run parallel PDF generation
        results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_process_single_pdf)(q, self.taus, self.pdf_params)
            for q in quantiles_np
        )

        # Filter valid results
        valid_indices = []
        grids = []
        pdfs = []

        for i, result in enumerate(results):
            if result is not None:
                valid_indices.append(i)
                grids.append(result[0])
                pdfs.append(result[1])

        return valid_indices, grids, pdfs

    @torch.no_grad()
    def _sample_batch(self, n_points: int, add_noise: bool = True, use_device: Optional[torch.device] = None) -> Dict[str, Tensor]:
        """
        Generate a batch of samples using parallel PDF processing.

        This method oversamples by OVERSAMPLE_RATIO to account for
        invalid samples that get filtered out.

        Args:
            n_points: Number of samples to generate
            use_device: Device to use for sampling (None = self.device)
        """
        # Determine device to use
        device = use_device if use_device is not None else self.device

        # Ensure quantiles are on correct device
        if self.quantiles.device != device:
            self.quantiles = self.quantiles.to(device)

        # Oversample to account for filtering
        n_oversample = int(n_points * OVERSAMPLE_RATIO)

        # Sample quantiles with noise
        sampled_idx = torch.randint(
            0, len(self.quantiles), (n_oversample,), device=device
        )
        if add_noise:
            sampled_quantiles = self.quantiles[sampled_idx] + (
                torch.randn(n_oversample, self.quantiles.size(1), device=device)
                * self.noise_std
            )
        else:
            sampled_quantiles = self.quantiles[sampled_idx]

        # Move to CPU for parallel PDF processing
        quantiles_np = sampled_quantiles.cpu().numpy()

        # Parallel PDF validation (first pass - just to filter valid quantiles)
        valid_indices, _, _ = self._parallel_pdf_processing(quantiles_np)

        if not valid_indices:
            # Edge case: no valid samples, return empty
            return self._empty_sample_dict(device)

        # Keep only valid quantiles
        valid_indices_tensor = torch.tensor(valid_indices, device=device)
        sampled_quantiles = sampled_quantiles[valid_indices_tensor]
        n_valid = len(valid_indices)

        # Generate time samples
        T = torch.randint(
            self.min_time, self.max_time, (n_valid, 1),
            dtype=torch.float32, device=device
        )
        varphi_t = torch.rand(n_valid, 1, device=device)
        varphi_T = T / DAYS_PER_MONTH

        # Ensure all inputs are on same device as model
        sampled_quantiles = sampled_quantiles.to(device)
        varphi_t = varphi_t.to(device)
        varphi_T = varphi_T.to(device)

        # Compute VarPi outputs in one batch
        # Use appropriate model based on device
        varpi_quantiles = self.varPi(sampled_quantiles, varphi_t, varphi_T)

        # Parallel PDF generation for VarPi outputs
        varpi_np = varpi_quantiles.cpu().numpy()
        varpi_valid_idx, grids, pdfs = self._parallel_pdf_processing(varpi_np)

        if not varpi_valid_idx:
            return self._empty_sample_dict(device)

        # Filter to final valid samples
        varpi_valid_tensor = torch.tensor(varpi_valid_idx, device=device)
        varpi_quantiles = varpi_quantiles[varpi_valid_tensor]
        sampled_quantiles = sampled_quantiles[varpi_valid_tensor]
        T = T[varpi_valid_tensor]
        varphi_t = varphi_t[varpi_valid_tensor]

        # Convert PDFs to tensors
        varpi_grids = torch.tensor(
            np.array(grids), dtype=torch.float32, device=device
        )
        varpi_pdfs = torch.tensor(
            np.array(pdfs), dtype=torch.float32, device=device
        )

        # Generate remaining samples
        n_final = len(varpi_quantiles)

        # Risk-free rate sampling
        rf = (
            torch.rand(n_final, 1, device=device) * self.rf_range + self.rf_min
        ) / DAYS_PER_YEAR

        # Log-normal stock price sampling
        S_prime = torch.exp(
            self._lognormal_mu +
            self.LOGNORMAL_SIGMA * torch.randn(n_final, 1, device=device)
        )
        S_prime = torch.clamp(S_prime, min=0.0, max=self.s_max_val)

        # Time to expiration
        t_prime = varphi_t * T

        return {
            "S_prime": S_prime,
            "t_prime": t_prime,
            "rf": rf,
            "varpi_q": varpi_quantiles,
            "varpi_pdfs": varpi_pdfs,
            "varpi_grids": varpi_grids,
            "varphi_q": sampled_quantiles
        }

    def _empty_sample_dict(self, device: Optional[torch.device] = None) -> Dict[str, Tensor]:
        """Return an empty sample dictionary for edge cases."""
        device = device if device is not None else self.device
        n_quantiles = len(self.taus)
        return {
            "S_prime": torch.empty(0, 1, device=device),
            "t_prime": torch.empty(0, 1, device=device),
            "rf": torch.empty(0, 1, device=device),
            "varpi_q": torch.empty(0, n_quantiles, device=device),
            "varpi_pdfs": torch.empty(0, 1000, device=device),
            "varpi_grids": torch.empty(0, 1000, device=device),
            "varphi_q": torch.empty(0, n_quantiles, device=device)
        }

    def sample(self, n_points: int, add_noise: bool = True) -> Dict[str, Tensor]:
        """
        Get n_points samples with proper gradient tracking.

        Uses oversampling internally to minimize the need for
        additional sampling rounds.

        Args:
            n_points: Number of samples to generate

        Returns:
            Dictionary of sample tensors with gradients enabled
        """
        samples = self._sample_batch(n_points, add_noise)
        current_count = len(samples["S_prime"])

        # If oversampling wasn't enough, get more samples
        # This should be rare with OVERSAMPLE_RATIO = 1.5
        while current_count < n_points:
            needed = n_points - current_count
            # Request extra to minimize iterations
            additional = self._sample_batch(int(needed * OVERSAMPLE_RATIO) + 1, add_noise)

            if len(additional["S_prime"]) == 0:
                # Avoid infinite loop if sampling keeps failing
                break

            samples = {
                key: torch.cat([samples[key], additional[key]], dim=0)
                for key in samples
            }
            current_count = len(samples["S_prime"])

        # Trim to exact count and enable gradients
        return {
            key: value[:n_points].float().requires_grad_(True)
            for key, value in samples.items()
        }


class BufferedStormSampler(StormSampler):
    """
    Asynchronous buffered sampler that pre-samples data in background.

    Maintains a buffer of pre-sampled data (15x batch size) and continuously
    refills it in a background thread, ensuring training never blocks on sampling.

    Features:
    - Disk caching: Samples are saved to disk for instant loading on re-runs
    - Progress bars: tqdm progress tracking for pre-filling operations
    - Background refilling: Continuous async sampling to keep buffer full
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
        noise_std: float = 0.001,
        n_jobs: int = DEFAULT_N_JOBS,
        buffer_multiplier: int = BUFFER_MULTIPLIER,
        refill_threshold: int = REFILL_THRESHOLD,
        cache_dir: str = DEFAULT_CACHE_DIR,
        use_cache: bool = True
    ):
        """
        Initialize the BufferedStormSampler.

        Args:
            device: Target device for tensors (cuda/cpu)
            taus: Quantile levels array
            min_time: Minimum time horizon in days
            max_time: Maximum time horizon in days
            rf_min: Minimum risk-free rate
            rf_max: Maximum risk-free rate
            s_max: Maximum normalized stock price
            noise_std: Standard deviation for quantile noise
            n_jobs: Number of parallel jobs (-1 for all cores)
            buffer_multiplier: Buffer size multiplier (default 15x batch size)
            refill_threshold: Refill when buffer drops below this multiplier
            cache_dir: Directory for caching samples to disk
            use_cache: Whether to use disk caching (default True)
        """
        super().__init__(
            device=device,
            taus=taus,
            min_time=min_time,
            max_time=max_time,
            rf_min=rf_min,
            rf_max=rf_max,
            s_max=s_max,
            noise_std=noise_std,
            n_jobs=n_jobs
        )

        self.buffer_multiplier = buffer_multiplier
        self.refill_threshold = refill_threshold
        self.batch_size = None
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache

        # CPU device for background sampling (avoids GPU contention)
        self._cpu_device = torch.device('cpu')

        # CPU copy of VarPi model for background worker (lazy initialization)
        self.varPi_cpu: Optional[torch.nn.Module] = None

        # Thread-safe queue for buffered samples
        self._buffer: queue.Queue = queue.Queue()

        # Thread control
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._buffer_lock = threading.Lock()

        # Lazy loading state for chunked cache
        self._available_chunks: List[Path] = []  # List of chunk file paths to load
        self._chunk_index: int = 0  # Current chunk index being loaded
        self._chunks_loaded: int = 0  # Counter for tracking loaded chunks
        self._total_chunks: int = 0  # Total chunks available

        # Statistics
        self._total_sampled = 0
        self._total_consumed = 0
        self._cache_hits = 0
        self._cache_misses = 0

        # Logger
        self.logger = logging.getLogger(__name__)

        # Create cache directory if it doesn't exist
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cpu_varpi(self) -> torch.nn.Module:
        """
        Get or create CPU copy of VarPi model for background sampling.
        Lazy initialization to avoid unnecessary CPU memory if not needed.
        """
        if self.varPi_cpu is None:
            self.logger.debug("Creating CPU copy of VarPi model for background sampling")
            self.varPi_cpu = self.varPi.cpu()
            self.varPi_cpu.eval()
        return self.varPi_cpu

    @torch.no_grad()
    def _sample_batch(self, n_points: int, use_device: Optional[torch.device] = None) -> Dict[str, Tensor]:
        """
        Override to support CPU device for background sampling.

        When use_device is CPU, uses CPU VarPi model to avoid GPU contention.
        """
        device = use_device if use_device is not None else self.device
        use_cpu = device == self._cpu_device

        # Oversample to account for filtering
        n_oversample = int(n_points * OVERSAMPLE_RATIO)

        # For CPU sampling, quantiles need to be on CPU
        if use_cpu:
            quantiles_cpu = self.quantiles.cpu()
            sampled_idx = torch.randint(
                0, len(quantiles_cpu), (n_oversample,), device=device
            )
            sampled_quantiles = quantiles_cpu[sampled_idx] + (
                torch.randn(n_oversample, quantiles_cpu.size(1), device=device)
                * self.noise_std
            )
        else:
            sampled_idx = torch.randint(
                0, len(self.quantiles), (n_oversample,), device=device
            )
            sampled_quantiles = self.quantiles[sampled_idx] + (
                torch.randn(n_oversample, self.quantiles.size(1), device=device)
                * self.noise_std
            )

        # Move to CPU for parallel PDF processing
        quantiles_np = sampled_quantiles.cpu().numpy()

        # Parallel PDF validation (first pass - just to filter valid quantiles)
        valid_indices, _, _ = self._parallel_pdf_processing(quantiles_np)

        if not valid_indices:
            return self._empty_sample_dict(device)

        # Keep only valid quantiles
        valid_indices_tensor = torch.tensor(valid_indices, device=device)
        sampled_quantiles = sampled_quantiles[valid_indices_tensor]
        n_valid = len(valid_indices)

        # Generate time samples
        T = torch.randint(
            self.min_time, self.max_time, (n_valid, 1),
            dtype=torch.float32, device=device
        )
        varphi_t = torch.rand(n_valid, 1, device=device)
        varphi_T = T / DAYS_PER_MONTH

        # Compute VarPi outputs - use CPU model if sampling on CPU
        if use_cpu:
            varpi_model = self._get_cpu_varpi()
            varpi_quantiles = varpi_model(sampled_quantiles, varphi_t, varphi_T)
        else:
            varpi_quantiles = self.varPi(sampled_quantiles, varphi_t, varphi_T)

        # Parallel PDF generation for VarPi outputs
        varpi_np = varpi_quantiles.cpu().numpy()
        varpi_valid_idx, grids, pdfs = self._parallel_pdf_processing(varpi_np)

        if not varpi_valid_idx:
            return self._empty_sample_dict(device)

        # Filter to final valid samples
        varpi_valid_tensor = torch.tensor(varpi_valid_idx, device=device)
        varpi_quantiles = varpi_quantiles[varpi_valid_tensor]
        sampled_quantiles = sampled_quantiles[varpi_valid_tensor]
        T = T[varpi_valid_tensor]
        varphi_t = varphi_t[varpi_valid_tensor]

        # Convert PDFs to tensors
        varpi_grids = torch.tensor(
            np.array(grids), dtype=torch.float32, device=device
        )
        varpi_pdfs = torch.tensor(
            np.array(pdfs), dtype=torch.float32, device=device
        )

        # Generate remaining samples
        n_final = len(varpi_quantiles)

        # Risk-free rate sampling
        rf = (
            torch.rand(n_final, 1, device=device) * self.rf_range + self.rf_min
        ) / DAYS_PER_YEAR

        # Log-normal stock price sampling
        S_prime = torch.exp(
            self._lognormal_mu +
            self.LOGNORMAL_SIGMA * torch.randn(n_final, 1, device=device)
        )
        S_prime = torch.clamp(S_prime, min=0.0, max=self.s_max_val)

        # Time to expiration
        t_prime = varphi_t * T

        return {
            "S_prime": S_prime,
            "t_prime": t_prime,
            "rf": rf,
            "varpi_q": varpi_quantiles,
            "varpi_pdfs": varpi_pdfs,
            "varpi_grids": varpi_grids,
            "varphi_q": sampled_quantiles
        }

    def _get_cache_key(self, batch_size: int) -> str:
        """
        Generate a unique cache key based on sampler configuration.

        The key is a SHA256 hash of parameters that affect sampling output.
        """
        key_data = {
            "batch_size": batch_size,
            "buffer_multiplier": self.buffer_multiplier,
            "taus": tuple(self.taus.tolist()),
            "noise_std": self.noise_std,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "rf_min": self.rf_min,
            "rf_range": self.rf_range,
            "s_max": self.s_max_val,
        }
        key_str = str(sorted(key_data.items()))
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _get_cache_path(self, batch_size: int, chunk_idx: Optional[int] = None) -> Path:
        """
        Get the cache file path for given batch size and optional chunk index.

        Args:
            batch_size: Batch size for cache key generation
            chunk_idx: Optional chunk index (for chunked caches)

        Returns:
            Path to cache file (chunked or single-file format)
        """
        cache_key = self._get_cache_key(batch_size)
        if chunk_idx is not None:
            return self.cache_dir / f"buffer_{cache_key}_chunk_{chunk_idx}.pt"
        else:
            return self.cache_dir / f"buffer_{cache_key}.pt"

    def _save_to_cache(self, samples_list: List[Dict[str, Tensor]], batch_size: int) -> bool:
        """
        Save sampled batches to disk cache, splitting into chunks of SAMPLES_PER_CACHE_CHUNK.

        Args:
            samples_list: List of sample batch dictionaries
            batch_size: Batch size used for cache key

        Returns:
            True if save succeeded, False otherwise
        """
        if not self.use_cache:
            return False

        try:
            # Move tensors to CPU for storage
            cpu_samples = []
            for batch in samples_list:
                cpu_batch = {k: v.cpu() for k, v in batch.items()}
                cpu_samples.append(cpu_batch)

            # Split into chunks of SAMPLES_PER_CACHE_CHUNK samples
            chunk_idx = 0
            current_chunk = []
            current_chunk_samples = 0
            chunks_saved = 0

            for batch in cpu_samples:
                batch_size_actual = len(batch["S_prime"])

                # Check if adding this batch would exceed chunk size
                if current_chunk_samples + batch_size_actual > SAMPLES_PER_CACHE_CHUNK and current_chunk:
                    # Save current chunk
                    chunk_path = self._get_cache_path(batch_size, chunk_idx)
                    torch.save({
                        "samples": current_chunk,
                        "batch_size": batch_size,
                        "buffer_multiplier": self.buffer_multiplier,
                        "taus": self.taus.tolist(),
                        "chunk_idx": chunk_idx,
                        "chunk_samples": current_chunk_samples,
                    }, chunk_path)

                    self.logger.debug("Saved chunk %d: %d batches (%d samples) to %s",
                                      chunk_idx, len(current_chunk), current_chunk_samples, chunk_path.name)
                    chunks_saved += 1

                    # Start new chunk
                    chunk_idx += 1
                    current_chunk = []
                    current_chunk_samples = 0

                # Add batch to current chunk
                current_chunk.append(batch)
                current_chunk_samples += batch_size_actual

            # Save final chunk if not empty
            if current_chunk:
                chunk_path = self._get_cache_path(batch_size, chunk_idx)
                torch.save({
                    "samples": current_chunk,
                    "batch_size": batch_size,
                    "buffer_multiplier": self.buffer_multiplier,
                    "taus": self.taus.tolist(),
                    "chunk_idx": chunk_idx,
                    "chunk_samples": current_chunk_samples,
                }, chunk_path)

                self.logger.debug("Saved chunk %d: %d batches (%d samples) to %s",
                                  chunk_idx, len(current_chunk), current_chunk_samples, chunk_path.name)
                chunks_saved += 1

            total_samples = sum(len(b["S_prime"]) for b in cpu_samples)
            self.logger.info("Saved %d batches (%d samples) across %d chunk file(s)",
                             len(cpu_samples), total_samples, chunks_saved)
            return True

        except Exception as e:
            self.logger.warning("Failed to save cache: %s", e)
            return False

    def _load_from_cache(
        self,
        batch_size: int,
        max_samples: Optional[int] = None
    ) -> Optional[Tuple[List[Path], int]]:
        """
        Discover available cache chunks without loading sample data.

        This method scans for both chunked cache files and old single-file caches
        (for backward compatibility), returning metadata about available chunks.

        Args:
            batch_size: Batch size for cache key
            max_samples: Maximum number of samples to consider (None = all)

        Returns:
            Tuple of (chunk_paths, total_samples) if cache found, None otherwise
            - chunk_paths: List of Path objects to chunk files (sorted by chunk index)
            - total_samples: Total samples available across all chunks
        """
        if not self.use_cache or not self.cache_dir.exists():
            return None

        cache_key = self._get_cache_key(batch_size)

        # Check for chunked cache files (new format)
        chunked_files = list(self.cache_dir.glob(f"buffer_{cache_key}_chunk_*.pt"))

        # Check for old single-file cache (backward compatibility)
        old_cache_path = self._get_cache_path(batch_size, chunk_idx=None)

        chunk_paths = []
        total_samples = 0

        if chunked_files:
            # New chunked format detected
            # Sort by chunk index
            def extract_chunk_idx(path):
                name = path.stem  # e.g., "buffer_abc123_chunk_0"
                parts = name.split("_chunk_")
                if len(parts) == 2:
                    try:
                        return int(parts[1])
                    except ValueError:
                        return -1
                return -1

            chunked_files_sorted = sorted(chunked_files, key=extract_chunk_idx)

            # Scan chunks to get total samples
            for chunk_path in chunked_files_sorted:
                try:
                    # Load only metadata, not samples
                    data = torch.load(chunk_path, weights_only=False, map_location='cpu')

                    # Validate cache matches current sampler config
                    cached_taus = tuple(data.get("taus", []))
                    current_taus = tuple(self.taus.tolist())

                    if cached_taus and cached_taus != current_taus:
                        self.logger.debug("Skipping cache %s: quantiles mismatch", chunk_path.name)
                        continue

                    # Count samples in this chunk
                    chunk_samples = data.get("chunk_samples", 0)
                    if chunk_samples == 0:
                        # Fallback: count manually if metadata missing
                        chunk_samples = sum(len(b.get("S_prime", []))
                                            for b in data.get("samples", []))

                    chunk_paths.append(chunk_path)
                    total_samples += chunk_samples

                    # Stop if we've discovered enough samples
                    if max_samples is not None and total_samples >= max_samples:
                        break

                except Exception as e:
                    self.logger.warning("Failed to read chunk metadata %s: %s", chunk_path.name, e)
                    continue

            if chunk_paths:
                self.logger.info("Discovered %d chunk(s) with %d total samples",
                                 len(chunk_paths), total_samples)
                self._cache_hits += 1
                return (chunk_paths, total_samples)

        elif old_cache_path.exists():
            # Old single-file format detected (backward compatibility)
            try:
                data = torch.load(old_cache_path, weights_only=False, map_location='cpu')

                # Validate cache matches current sampler config
                cached_taus = tuple(data.get("taus", []))
                current_taus = tuple(self.taus.tolist())

                if cached_taus and cached_taus != current_taus:
                    self.logger.debug("Skipping old cache: quantiles mismatch")
                    return None

                # Count samples
                total_samples = sum(len(b.get("S_prime", [])) for b in data.get("samples", []))

                self.logger.info("Discovered old single-file cache with %d samples (will be treated as one chunk)",
                                 total_samples)
                self._cache_hits += 1
                return ([old_cache_path], total_samples)

            except Exception as e:
                self.logger.warning("Failed to read old cache metadata: %s", e)
                return None

        # No cache found
        self._cache_misses += 1
        return None

    def _load_next_chunk(self) -> int:
        """
        Load the next chunk from _available_chunks into the buffer.

        This method loads samples from disk and adds them to the buffer queue.
        Samples are kept on CPU in the buffer and moved to GPU when consumed.

        Returns:
            Number of batches loaded from the chunk (0 if no more chunks)
        """
        if self._chunk_index >= len(self._available_chunks):
            self.logger.debug("No more chunks to load (index=%d, total=%d)",
                              self._chunk_index, len(self._available_chunks))
            return 0

        chunk_path = self._available_chunks[self._chunk_index]

        try:
            # Load chunk data (samples stay on CPU)
            data = torch.load(chunk_path, weights_only=False, map_location='cpu')
            samples = data.get("samples", [])

            if not samples:
                self.logger.warning("Chunk %s is empty", chunk_path.name)
                self._chunk_index += 1
                return 0

            # Add batches to buffer (samples remain on CPU)
            batches_added = 0
            samples_added = 0
            for batch in samples:
                self._buffer.put(batch)
                batches_added += 1
                samples_added += len(batch.get("S_prime", []))

            self._chunks_loaded += 1
            self._chunk_index += 1

            self.logger.info("Loaded chunk %d/%d: %d batches (%d samples) from %s",
                             self._chunks_loaded, self._total_chunks,
                             batches_added, samples_added, chunk_path.name)

            return batches_added

        except Exception as e:
            self.logger.error("Failed to load chunk %s: %s", chunk_path.name, e)
            self._chunk_index += 1
            return 0

    def clear_cache(self) -> int:
        """
        Clear all cached samples from disk.

        Returns:
            Number of cache files deleted
        """
        if not self.cache_dir.exists():
            return 0

        deleted = 0
        for cache_file in self.cache_dir.glob("buffer_*.pt"):
            try:
                cache_file.unlink()
                deleted += 1
            except Exception as e:
                self.logger.warning("Failed to delete %s: %s", cache_file, e)

        self.logger.info("Cleared %d cache files", deleted)
        return deleted

    def _calculate_samples_needed(
        self,
        batch_size: int,
        n_epochs: Optional[int] = None,
        resample_frequency: Optional[int] = None,
        min_resample_frequency: int = 5
    ) -> int:
        """
        Calculate total samples needed for a training run.

        Accounts for adaptive resampling:
        - First 20 epochs: resample every min_resample_frequency epochs
        - After 20 epochs: resample every resample_frequency epochs

        Args:
            batch_size: Number of samples per resampling (n_points)
            n_epochs: Total number of training epochs
            resample_frequency: How often to resample after epoch 20 (every N epochs)
            min_resample_frequency: How often to resample in first 20 epochs (every N epochs)

        Returns:
            Total number of samples needed
        """
        if n_epochs is not None and resample_frequency is not None:
            # Calculate based on actual training needs with adaptive resampling:
            # - 1 initial batch
            # - n_resamples batches during training (accounting for adaptive frequency)
            # - buffer_multiplier batches for safety margin

            # Training loop logic:
            # for e in range(n_epochs):
            #   if e > 0:
            #     if e < 20:
            #       resample_freq = min_resample_frequency
            #     else:
            #       resample_freq = resample_frequency
            #     if e % resample_freq == 0:
            #       resample()

            # Simulate the training loop to count resamples exactly
            n_resamples = 0
            resample_epochs = []
            for e in range(1, n_epochs):  # Start from epoch 1 (epoch 0 is initial sample)
                if e < 20:
                    resample_freq = min_resample_frequency
                else:
                    resample_freq = resample_frequency

                if e % resample_freq == 0:
                    n_resamples += 1
                    resample_epochs.append(e)

            total_batches = 1 + n_resamples + self.buffer_multiplier
            total_samples = batch_size * total_batches
            return total_samples
        else:
            # Conservative estimate: just buffer multiplier
            return batch_size * self.buffer_multiplier

    def start_buffer(
        self,
        batch_size: int,
        n_epochs: Optional[int] = None,
        resample_frequency: Optional[int] = None,
        min_resample_frequency: int = 5
    ) -> None:
        """
        Initialize and start the background buffer worker (conditionally).

        Uses lazy loading for chunked caches: loads only the first chunk eagerly,
        remaining chunks are loaded on-demand by the background worker.

        Args:
            batch_size: Expected batch size for sampling (n_points)
            n_epochs: Total number of training epochs (for calculating needs)
            resample_frequency: Resampling frequency after epoch 20 (for calculating needs)
            min_resample_frequency: Resampling frequency in first 20 epochs (for calculating needs)
        """
        # If worker is still alive, stop it first and wait
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self.logger.warning("Buffer worker still running, stopping first...")
            self.stop()  # This will wait and clean up

        # Reset worker thread reference and state
        self._worker_thread = None
        self.batch_size = batch_size
        self._stop_event.clear()

        # Reset lazy loading state
        self._available_chunks = []
        self._chunk_index = 0
        self._chunks_loaded = 0
        self._total_chunks = 0

        # Clear existing buffer
        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                break

        # Calculate number of batches needed (not samples, since get_batch consumes entire batches)
        # 1 initial + n_resamples + buffer_multiplier
        n_resamples = 0
        for e in range(1, n_epochs if n_epochs else 1):
            if e < 20:
                resample_freq = min_resample_frequency
            else:
                resample_freq = resample_frequency
            if e % resample_freq == 0:
                n_resamples += 1

        batches_needed = 1 + n_resamples + self.buffer_multiplier

        # Discover available cache chunks (without loading data)
        cache_info = self._load_from_cache(batch_size, max_samples=None)

        if cache_info is not None:
            chunk_paths, total_cached_samples = cache_info
            self._available_chunks = chunk_paths
            self._total_chunks = len(chunk_paths)

            tqdm.write(
                f"[Cache] Discovered {self._total_chunks} chunk(s) with {total_cached_samples:,} total samples"
            )

            # Load ONLY the first chunk eagerly
            if self._available_chunks:
                tqdm.write(f"[Cache] Loading first chunk into buffer...")
                self._load_next_chunk()
                tqdm.write("[Cache] First chunk loaded (samples on CPU, moved to GPU when consumed)")

            # Check if all chunks provide enough samples for training
            # Estimate: assume each chunk is roughly the same size
            # We'll need batches_needed batches, and we have total_cached_samples samples
            # Conservative check: if we have enough samples for the training, cache is sufficient
            samples_needed = batches_needed * batch_size
            cache_sufficient = total_cached_samples >= samples_needed

            if cache_sufficient and self._chunk_index >= len(self._available_chunks):
                # All chunks loaded and cache is sufficient
                tqdm.write(
                    f"[Sampler] Cache sufficient ({total_cached_samples:,} samples >= {samples_needed:,} needed). "
                    f"No background worker needed."
                )
                return
        else:
            self._cache_misses += 1
            tqdm.write("[Cache] No cache found")

        # Cache insufficient or more chunks to load - need background worker
        target_buffer_batches = self.buffer_multiplier
        current_buffer_batches = self._buffer.qsize()

        if current_buffer_batches < target_buffer_batches:
            # Pre-fill buffer to target size (in batches)
            # First, try loading more chunks if available
            needed_batches = target_buffer_batches - current_buffer_batches
            loaded_from_chunks = 0

            # Load more chunks until buffer is full or no more chunks
            while (current_buffer_batches < target_buffer_batches and
                   self._chunk_index < len(self._available_chunks)):
                batches_loaded = self._load_next_chunk()
                if batches_loaded > 0:
                    loaded_from_chunks += batches_loaded
                    current_buffer_batches = self._buffer.qsize()
                else:
                    break

            # If still need more batches, generate them
            if current_buffer_batches < target_buffer_batches:
                needed_batches = target_buffer_batches - current_buffer_batches
                tqdm.write(
                    f"[Sampler] Pre-filling buffer with {needed_batches} generated batches..."
                )

                samples_for_cache = []
                chunk_size = batch_size

                with tqdm(total=needed_batches, desc="Pre-filling buffer", unit="batch",
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                    prefill_batches = 0
                    while prefill_batches < needed_batches:
                        batch = self._sample_batch(chunk_size)
                        n_samples = len(batch["S_prime"])

                        if n_samples > 0:
                            self._buffer.put(batch)
                            samples_for_cache.append(batch)
                            prefill_batches += 1
                            pbar.update(1)
                        else:
                            # If sampling fails, try smaller chunk
                            chunk_size = max(1, chunk_size // 2)

                tqdm.write(f"[Sampler] Pre-filled {prefill_batches} additional batches")

                # Save newly generated samples to cache
                if self.use_cache and samples_for_cache:
                    self._save_to_cache(samples_for_cache, batch_size)

        # Start background worker thread if:
        # 1. More chunks remain to be loaded, OR
        # 2. We might need to generate more samples during training
        need_worker = (self._chunk_index < len(self._available_chunks) or
                       current_buffer_batches < batches_needed)

        if need_worker:
            chunks_remaining = len(self._available_chunks) - self._chunk_index
            tqdm.write(
                f"[Sampler] Starting background worker ({chunks_remaining} chunks remaining)"
            )
            self._worker_thread = threading.Thread(
                target=self._refill_worker,
                daemon=True,
                name="BufferRefillWorker"
            )
            self._worker_thread.start()
            tqdm.write("[Sampler] Background buffer worker started")
        else:
            tqdm.write("[Sampler] All chunks loaded, no background worker needed")

    def _refill_worker(self) -> None:
        """
        Background worker that continuously monitors and refills the buffer.

        Lazy loading strategy:
        1. When buffer is low, first try to load next chunk from cache
        2. Only generate new samples if all chunks are exhausted

        This ensures memory-efficient operation with chunked caches.
        """
        if self.batch_size is None:
            self.logger.error("Buffer not initialized. Call start_buffer() first.")
            return

        target_size = self.batch_size * self.buffer_multiplier
        refill_size = self.batch_size * self.refill_threshold

        while not self._stop_event.is_set():
            try:
                # Check current buffer size
                current_size = self._buffer.qsize()

                if current_size < refill_size:
                    # Buffer is low, refill it
                    self.logger.debug(
                        "Buffer low (%d < %d), refilling...",
                        current_size, refill_size
                    )

                    # Strategy 1: Try loading next chunk if available
                    if self._chunk_index < len(self._available_chunks):
                        batches_loaded = self._load_next_chunk()
                        if batches_loaded > 0:
                            self.logger.debug(
                                "Refilled from chunk. Buffer size: %d",
                                self._buffer.qsize()
                            )
                            # Continue to next iteration to check buffer again
                            continue

                    # Strategy 2: Generate new samples if no chunks available
                    needed = target_size - current_size
                    self.logger.debug(
                        "No chunks available, generating %d samples...", needed
                    )

                    chunk_size = self.batch_size
                    refilled = 0

                    while refilled < needed and not self._stop_event.is_set():
                        # Use CPU device for background sampling to avoid GPU contention
                        batch = self._sample_batch(chunk_size, use_device=self._cpu_device)
                        n_samples = len(batch["S_prime"])

                        if n_samples > 0:
                            self._buffer.put(batch)
                            refilled += n_samples
                            self._total_sampled += n_samples
                        else:
                            # Sampling failed, try smaller chunk
                            chunk_size = max(1, chunk_size // 2)

                    if refilled > 0:
                        self.logger.debug(
                            "Generated %d samples. Buffer size: %d",
                            refilled, self._buffer.qsize()
                        )

                # Sleep briefly before next check
                self._stop_event.wait(REFILL_CHECK_INTERVAL)

            except Exception as e:
                self.logger.error("Error in buffer refill worker: %s", e, exc_info=True)
                # Continue running despite errors
                self._stop_event.wait(REFILL_CHECK_INTERVAL)

    def get_batch(self, n_points: int) -> Dict[str, Tensor]:
        """
        Get n_points samples from the buffer (non-blocking).

        Works even without background worker if samples are already in buffer
        (e.g., loaded from cache). Falls back to direct sampling only if buffer
        is empty and no worker is running.

        Args:
            n_points: Number of samples to retrieve

        Returns:
            Dictionary of sample tensors with gradients enabled
        """
        # Try to get samples from buffer first (even if no worker thread)
        collected_samples = []
        collected_count = 0

        # Collect enough samples from buffer
        while collected_count < n_points:
            try:
                # Non-blocking get
                batch = self._buffer.get_nowait()
                collected_samples.append(batch)
                collected_count += len(batch["S_prime"])
                self._total_consumed += len(batch["S_prime"])
            except queue.Empty:
                # Buffer is empty
                if self._worker_thread is None or not self._worker_thread.is_alive():
                    # No worker thread and buffer empty - fall back to direct sampling
                    if collected_count == 0:
                        # No samples at all, use direct sampling
                        self.logger.warning(
                            "Buffer empty and no worker active. Falling back to direct sampling."
                        )
                        return self.sample(n_points)
                    else:
                        # Got some samples but need more
                        self.logger.warning(
                            "Buffer empty (needed %d, got %d). Falling back to direct sampling for remainder.",
                            n_points, collected_count
                        )
                        remaining = n_points - collected_count
                        direct_samples = self.sample(remaining)
                        # Ensure samples are on correct device
                        for key in direct_samples:
                            if direct_samples[key].device != self.device:
                                direct_samples[key] = direct_samples[key].to(self.device)
                        if len(direct_samples["S_prime"]) > 0:
                            collected_samples.append(direct_samples)
                        break
                else:
                    # Worker is running but buffer temporarily empty - wait a bit or fall back
                    self.logger.debug(
                        "Buffer temporarily empty (needed %d, got %d). Worker active, falling back to direct sampling.",
                        n_points, collected_count
                    )
                    remaining = n_points - collected_count
                    direct_samples = self.sample(remaining)
                    # Ensure samples are on correct device
                    for key in direct_samples:
                        if direct_samples[key].device != self.device:
                            direct_samples[key] = direct_samples[key].to(self.device)
                    if len(direct_samples["S_prime"]) > 0:
                        collected_samples.append(direct_samples)
                    break

        # Combine all collected samples
        if not collected_samples:
            # Complete fallback to direct sampling
            samples = self.sample(n_points)
            # Ensure all samples are on correct device
            result = {}
            for key, value in samples.items():
                if value.device != self.device:
                    value = value.to(self.device)
                result[key] = value
            return result

        # Concatenate all batches
        combined = {}
        for key in collected_samples[0].keys():
            tensors = [batch[key] for batch in collected_samples]
            combined[key] = torch.cat(tensors, dim=0)

        # Move to target device (GPU) if samples came from CPU background worker
        # This ensures training always gets GPU tensors
        result = {}
        for key, value in combined.items():
            if value.device != self.device:
                value = value.to(self.device)
            result[key] = value[:n_points].float().requires_grad_(True)

        return result

    def stop(self) -> None:
        """
        Gracefully stop the background buffer worker and clear buffer.

        Signals the worker to stop and waits for it to finish
        the current sampling batch. Also clears buffer to free memory.
        """
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self.logger.info("Stopping buffer worker...")
            self._stop_event.set()

            # Wait longer for worker to finish (increased timeout)
            self._worker_thread.join(timeout=10.0)
            if self._worker_thread.is_alive():
                self.logger.warning("Buffer worker did not stop within timeout")
            else:
                self.logger.info(
                    "Buffer worker stopped. Total sampled: %d, Total consumed: %d",
                    self._total_sampled, self._total_consumed
                )

        # Clear buffer to free memory
        self.clear_buffer()

        # Delete CPU VarPi model to free memory
        if hasattr(self, 'varPi_cpu') and self.varPi_cpu is not None:
            del self.varPi_cpu
            self.varPi_cpu = None

    def clear_buffer(self) -> int:
        """
        Clear all samples from the in-memory buffer.

        Returns:
            Number of batches cleared
        """
        cleared = 0
        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
                cleared += 1
            except queue.Empty:
                break

        if cleared > 0:
            self.logger.debug("Cleared %d batches from buffer", cleared)

        # Check worker status (moved from unreachable code)
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self.logger.warning("Buffer worker still alive after clear")

        return cleared

    def get_buffer_stats(self) -> Dict[str, any]:
        """
        Get current buffer statistics including chunk loading info.

        Returns:
            Dictionary with buffer size, statistics, cache info, and chunk status
        """
        return {
            "buffer_size": self._buffer.qsize(),
            "target_size": self.batch_size * self.buffer_multiplier if self.batch_size else 0,
            "refill_threshold": self.batch_size * self.refill_threshold if self.batch_size else 0,
            "total_sampled": self._total_sampled,
            "total_consumed": self._total_consumed,
            "worker_alive": self._worker_thread.is_alive() if self._worker_thread else False,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_enabled": self.use_cache,
            "chunks_available": self._total_chunks,
            "chunks_loaded": self._chunks_loaded,
            "current_chunk": self._chunk_index,
        }

    def sample(self, n_points: int) -> Dict[str, Tensor]:
        """
        Direct sampling (fallback method) with device consistency.

        This method is inherited from StormSampler and provides
        backward compatibility. For buffered sampling, use get_batch().
        """
        # Ensure we use GPU device for direct sampling
        samples = super().sample(n_points)
        # Double-check all tensors are on correct device
        result = {}
        for key, value in samples.items():
            if value.device != self.device:
                value = value.to(self.device)
            result[key] = value
        return result


if __name__ == "__main__":
    import json
    import time

    # Test the optimized sampler
    n_samples = 1000
    with open("config.json", "r", encoding="utf-8") as f:
        CONFIG = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    taus = CONFIG["general"]["quantiles"]

    # Initialize sampler
    print(f"Initializing sampler on {device}...")
    sampler = StormSampler(device, taus)

    # Benchmark sampling
    print(f"Sampling {n_samples} points...")
    start_time = time.time()
    samples = sampler.sample(n_samples)
    end_time = time.time()

    # Print stats
    total_time = end_time - start_time
    print(f"Total samples: {len(samples['S_prime'])}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Samples per second: {n_samples / total_time:.1f}")
    print(f"Average time per sample: {(total_time / n_samples) * 1000:.2f} ms")
