import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import wasserstein_distance


def generate_smooth_pdf(quantiles, taus, min_density=1e-3, eps=1e-6, window=61, grid_points=1000):
    """
    Generate a smoothed PDF from quantiles with additional controls to prevent spikes
    and ensure the CDF is between [0, 1].
    """
    # Constants
    og_quants = quantiles.copy()
    og_taus = taus.copy()

    unique_mask = np.concatenate(([True], np.diff(quantiles) > eps))
    quantiles = quantiles[unique_mask]
    taus = taus[unique_mask]

    # Create denser grid
    grid_x = np.linspace(quantiles[0], quantiles[-1], grid_points)

    # Monotonic spline for the CDF
    try:
        cdf_monotonic = PchipInterpolator(quantiles, taus, extrapolate=False)
        cdf = cdf_monotonic(grid_x)
    except Exception as e:
        # print(f"Quantiles: {og_quants}")
        # print(f"Taus: {og_taus}")
        # print("Falling back to linear interpolation:", e)
        cdf = np.interp(grid_x, quantiles, taus)

    # Clamp CDF to [0,1], then ensure it's monotonically non-decreasing
    cdf = np.clip(cdf, 0, 1)
    cdf = np.maximum.accumulate(cdf)
    # Rescale so that it starts exactly at 0 and ends exactly at 1
    cdf -= cdf[0]
    if cdf[-1] > 0:
        cdf /= cdf[-1]

    # Approximate PDF from finite differences (or use derivative if PCHIP)
    density = np.gradient(cdf, grid_x)

    smoothed_density = np.convolve(density, np.ones(window)/window, mode='same')

    # Ensure non-negative and non-zero density
    smoothed_density = np.maximum(smoothed_density, min_density)

    # Normalize PDF to integrate to 1
    area = np.trapz(smoothed_density, grid_x)
    smoothed_density = smoothed_density / area

    # regenerate CDF
    cdf = np.cumsum(smoothed_density) * (grid_x[1] - grid_x[0])

    return grid_x, smoothed_density, cdf


def calculate_wasserstein(cdf, predicted_grid, realized_returns):
    """
    Calculate Wasserstein distance between predicted PDF and realized returns.

    Args:
    pdf: array of predicted probability density values
    predicted_grid: array of x values where density is evaluated
    realized_returns: array of actual realized returns

    Returns:
    float: Wasserstein distance
    """

    # Sample predicted returns from the PDF
    random_uniform_samples = np.random.uniform(0, 1, len(realized_returns))
    predicted_samples = np.interp(random_uniform_samples, cdf, predicted_grid)

    # Calculate Wasserstein distance
    return wasserstein_distance(predicted_samples, realized_returns)
