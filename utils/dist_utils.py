import warnings
import numpy as np
from typing import List, Optional, Tuple
from scipy.interpolate import PchipInterpolator
from scipy.stats import wasserstein_distance
from joblib import Parallel, delayed


# Pre-compute common convolution kernels for reuse
_KERNEL_CACHE = {}


def _get_smoothing_kernel(window: int) -> np.ndarray:
    """Get or create a cached smoothing kernel."""
    if window not in _KERNEL_CACHE:
        _KERNEL_CACHE[window] = np.ones(window) / window
    return _KERNEL_CACHE[window]


def generate_smooth_pdf(
    quantiles: np.ndarray,
    taus: np.ndarray,
    min_density: float = 1e-3,
    eps: float = 1e-6,
    window: int = 61,
    grid_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a smoothed PDF from quantiles with controls to prevent spikes
    and ensure the CDF is between [0, 1].
    
    Args:
        quantiles: Array of quantile values
        taus: Array of quantile levels (probabilities)
        min_density: Minimum density value to prevent zeros
        eps: Epsilon for detecting duplicate quantiles
        window: Smoothing window size for convolution
        grid_points: Number of points in output grid
        
    Returns:
        Tuple of (grid_x, smoothed_density, cdf)
    """
    # Filter duplicate quantiles
    unique_mask = np.concatenate(([True], np.diff(quantiles) > eps))
    quantiles_filtered = quantiles[unique_mask]
    taus_filtered = taus[unique_mask]

    # Create dense grid
    grid_x = np.linspace(quantiles_filtered[0], quantiles_filtered[-1], grid_points)

    # Monotonic spline interpolation for CDF
    try:
        cdf_interpolator = PchipInterpolator(quantiles_filtered, taus_filtered, extrapolate=False)
        cdf = cdf_interpolator(grid_x)
    except Exception:
        # Fallback to linear interpolation
        cdf = np.interp(grid_x, quantiles_filtered, taus_filtered)

    # Ensure valid CDF: monotonic and in [0, 1]
    cdf = np.clip(cdf, 0, 1)
    cdf = np.maximum.accumulate(cdf)
    
    # Normalize to exact [0, 1] range
    cdf -= cdf[0]
    if cdf[-1] > 0:
        cdf /= cdf[-1]

    # Compute PDF from CDF gradient (suppress divide-by-zero warnings for edge cases)
    # These warnings occur when grid spacing is very small or identical values exist
    with warnings.catch_warnings():
        # Suppress numpy RuntimeWarnings about invalid values/divide operations
        # This catches warnings like "invalid value encountered in divide"
        warnings.filterwarnings('ignore', message='.*invalid value.*', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='.*divide.*', category=RuntimeWarning)
        density = np.gradient(cdf, grid_x)

    # Smooth with cached kernel
    kernel = _get_smoothing_kernel(window)
    smoothed_density = np.convolve(density, kernel, mode='same')

    # Ensure non-negative density
    smoothed_density = np.maximum(smoothed_density, min_density)

    # Normalize to integrate to 1
    area = np.trapezoid(smoothed_density, grid_x)
    smoothed_density = smoothed_density / area

    # Regenerate consistent CDF
    dx = grid_x[1] - grid_x[0]
    cdf = np.cumsum(smoothed_density) * dx

    return grid_x, smoothed_density, cdf


def generate_smooth_pdf_safe(
    quantiles: np.ndarray,
    taus: np.ndarray,
    **kwargs
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Safe wrapper for generate_smooth_pdf that returns None on failure.
    
    Useful for batch processing where some inputs may be invalid.
    
    Args:
        quantiles: Array of quantile values
        taus: Array of quantile levels
        **kwargs: Additional arguments passed to generate_smooth_pdf
        
    Returns:
        Tuple of (grid_x, pdf, cdf) or None if generation fails
    """
    try:
        grid, pdf, cdf = generate_smooth_pdf(quantiles, taus, **kwargs)
        if np.isnan(pdf).any() or np.isnan(grid).any():
            return None
        return (grid, pdf, cdf)
    except Exception:
        return None


def batch_generate_pdfs(
    quantiles_batch: np.ndarray,
    taus: np.ndarray,
    n_jobs: int = -1,
    **kwargs
) -> Tuple[List[int], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Generate PDFs for a batch of quantile arrays in parallel.
    
    Uses joblib for parallel processing across CPU cores.
    
    Args:
        quantiles_batch: Array of shape (n_samples, n_quantiles)
        taus: Array of quantile levels
        n_jobs: Number of parallel jobs (-1 for all cores)
        **kwargs: Additional arguments passed to generate_smooth_pdf
        
    Returns:
        Tuple of (valid_indices, grids, pdfs, cdfs) where:
        - valid_indices: List of indices that produced valid PDFs
        - grids: List of grid arrays for valid samples
        - pdfs: List of PDF arrays for valid samples
        - cdfs: List of CDF arrays for valid samples
    """
    # Process in parallel
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(generate_smooth_pdf_safe)(q, taus, **kwargs)
        for q in quantiles_batch
    )
    
    # Collect valid results
    valid_indices = []
    grids = []
    pdfs = []
    cdfs = []
    
    for i, result in enumerate(results):
        if result is not None:
            valid_indices.append(i)
            grids.append(result[0])
            pdfs.append(result[1])
            cdfs.append(result[2])
    
    return valid_indices, grids, pdfs, cdfs


def calculate_wasserstein(
    cdf: np.ndarray,
    predicted_grid: np.ndarray,
    realized_returns: np.ndarray
) -> float:
    """
    Calculate Wasserstein distance between predicted PDF and realized returns.

    Args:
        cdf: Array of CDF values
        predicted_grid: Array of x values where CDF is evaluated
        realized_returns: Array of actual realized returns

    Returns:
        Wasserstein distance (float)
    """
    # Sample from predicted distribution using inverse CDF
    n_samples = len(realized_returns)
    uniform_samples = np.random.uniform(0, 1, n_samples)
    predicted_samples = np.interp(uniform_samples, cdf, predicted_grid)

    return wasserstein_distance(predicted_samples, realized_returns)
