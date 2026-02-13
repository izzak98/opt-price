import torch
import numpy as np
from accelerate import Accelerator

accelerator = Accelerator()


def fast_normal_cdf(x, mu=0, sigma=1):
    """
    Approximate standard normal CDF Φ(x) using a polynomial/erf-based approximation.

    Args:
        x: Tensor of shape (batch_size, 1) or compatible
        mu: Mean (default 0)
        sigma: Std dev (default 1)

    Returns:
        Tensor of same shape as x with values ≈ Φ((x - mu) / sigma).
    """
    z = (x - mu) / (sigma * (2**0.5))
    t = 1 / (1 + 0.3275911 * torch.abs(z))
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    erf = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * torch.exp(-z * z)
    return 0.5 * (1 + torch.sign(z) * erf)


def gaussian_pdf(x, mu=0, sigma=1):
    """
    Standard Gaussian PDF φ(x) = (1 / (σ sqrt(2π))) * exp(-0.5 * ((x - μ)/σ)^2).

    Args:
        x: Tensor of any shape
        mu: Mean
        sigma: Std dev

    Returns:
        Tensor of same shape as x with values φ((x - μ)/σ).
    """
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * (2 * np.pi) ** 0.5)


def torch_interp(x, xp, fp):
    """
    Batched linear interpolation, analogous to numpy.interp, but in PyTorch.

    Shapes:
        x:  (batch_size, 1)             – query points
        xp: (len_values,) or (batch_size, len_values) – x-coordinates of data points
        fp: (batch_size, len_values)    – y-coordinates of data points

    For each batch b, this computes a linear interpolation of fp[b] at x[b]
    using xp[b] as the x-grid.

    Returns:
        (batch_size, 1) tensor of interpolated values.
    """
    batch_size = x.shape[0]

    # Handle case where xp is shared across batch
    if len(xp.shape) == 1:
        xp = xp.unsqueeze(0).expand(batch_size, -1)

    # Infer device from input tensors (prefer x, then xp, then fp)
    if isinstance(x, torch.Tensor):
        device = x.device
    elif isinstance(xp, torch.Tensor):
        device = xp.device
    elif isinstance(fp, torch.Tensor):
        device = fp.device
    else:
        device = accelerator.device

    # Convert any remaining non-tensor inputs to the inferred device
    x = torch.as_tensor(x).to(device)
    xp = torch.as_tensor(xp).to(device)
    fp = torch.as_tensor(fp).to(device)

    # Ensure xp is contiguous to avoid performance warning
    if not xp.is_contiguous():
        xp = xp.contiguous()

    # Find indices of values in original array for each batch
    idxs = torch.searchsorted(xp, x)  # (batch_size, 1)

    # Clip indices to valid range
    idxs = torch.clamp(idxs, 1, xp.shape[1] - 1)

    # Calculate interpolation weights per batch
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)  # (batch_size, 1)
    x0 = xp[batch_indices, idxs - 1]
    x1 = xp[batch_indices, idxs]
    y0 = fp[batch_indices, idxs - 1]
    y1 = fp[batch_indices, idxs]

    # Compute weights with numerical stability check to avoid divide-by-zero warnings
    denominator = x1 - x0
    # Avoid division by zero (when x0 == x1, use linear interpolation at x0)
    weights = torch.where(
        torch.abs(denominator) > 1e-10,
        (x - x0) / denominator,
        torch.zeros_like(x)
    )

    # Interpolate: y = y0 + w * (y1 - y0)
    return y0 + weights * (y1 - y0)


def calc_varrho(taus, varpi_quants, varpi_grid, varpi_pdf, t, eps=1e-4):
    """
    Compute the stochastic coefficient ρ used in the SPDE.

    Conceptually, ρ rescales the Gaussian density to match the learned return
    distribution defined by (varpi_quants, varpi_pdf).

    Inputs:
        taus:          (num_quantiles,) tensor of quantile levels in [0, 1]
        varpi_quants:  (batch_size, num_quantiles) learned quantiles Q(u)
        varpi_grid:    (batch_size, grid_size) support grid for learned PDF
        varpi_pdf:     (batch_size, grid_size) learned PDF values f(x) on grid
        t:             (batch_size, 1) time variable (same units as in trainer)
        eps:           small constant for numerical stability

    Mathematically:
        1. z      = w_t / sqrt(t)
        2. ρ_G    = φ(z) = standard normal PDF at z
        3. u      = Φ(z) ∈ (0, 1)       (standard normal CDF)
        4. q(u)   = Q(u)                (inverse CDF / quantile function of varpi)
        5. f(q)   = density of varpi at q
        6. ρ      = ρ_G / (sqrt(t) * f(q))

       where steps (4) and (5) are implemented via interpolation over
       varpi_quants/taus and varpi_grid/varpi_pdf respectively.

    Returns:
        varrho: (batch_size, 1) tensor, non-negative.
    """
    # Ensure t is positive to avoid division by zero
    t_safe = torch.clamp(t, min=eps)

    # z = N(0,1) - ensure on same device as input
    device = t.device
    z = torch.randn(t.size(0), 1, device=device)

    # ρ_G = φ(z): standard normal PDF at z (stable computation)
    rho_gauss = torch.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi))

    # u = Φ(z): standard normal CDF at z ∈ (0, 1)
    u = 0.5 * (1 + torch.erf(z / np.sqrt(2)))

    # Clamp u to avoid explosions at lower and upper tails
    u_safe = torch.clamp(u, min=eps, max=1 - eps)

    # Step 4: q(u) = Q(u) via interpolation over learned quantiles
    # taus:       (num_quantiles,)
    # varpi_quants: (batch_size, num_quantiles)
    # Ensure taus tensor is on the same device
    if isinstance(taus, torch.Tensor):
        taus_tensor = taus.to(device)
    else:
        taus_tensor = torch.tensor(taus, device=device)
    rho_quant = torch_interp(u_safe, taus_tensor, varpi_quants)

    # Step 5: f(q) via interpolation over learned PDF grid
    # varpi_grid: (batch_size, grid_size)
    # varpi_pdf:  (batch_size, grid_size)
    rho_varpi = torch_interp(rho_quant, varpi_grid, varpi_pdf)

    # Regularize PDF to avoid division by zero
    rho_varpi_safe = torch.clamp(rho_varpi, min=eps)

    # Step 6: ρ = ρ_G / (sqrt(t) * f(q)), ensure non-negative
    varrho = rho_gauss / (torch.sqrt(t_safe) * rho_varpi_safe)
    return varrho
