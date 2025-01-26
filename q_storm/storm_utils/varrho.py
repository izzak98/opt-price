import torch
import numpy as np
from accelerate import Accelerator

accelerator = Accelerator()


def fast_normal_cdf(x, mu=0, sigma=1):
    z = (x - mu) / (sigma * (2**0.5))
    t = 1 / (1 + 0.3275911 * torch.abs(z))
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    erf = 1 - ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t * torch.exp(-z*z)
    return 0.5 * (1 + torch.sign(z) * erf)


def gaussian_pdf(x, mu=0, sigma=1):
    return torch.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * (2 * np.pi)**0.5)


def torch_interp(x, xp, fp):
    """
    PyTorch equivalent of np.interp with batched inputs

    Args:
        x: Points to evaluate at (batch_size, 1)
        xp: Known x values (len_values) or (batch_size, len_values)
        fp: Known y values (batch_size, len_values)

    Returns:
        Interpolated values of shape (batch_size, 1)
    """
    batch_size = x.shape[0]

    # Handle case where xp is shared across batch
    if len(xp.shape) == 1:
        xp = xp.unsqueeze(0).expand(batch_size, -1)

    # Convert any remaining non-tensor inputs
    x = torch.as_tensor(x).to(accelerator.device)
    xp = torch.as_tensor(xp).to(accelerator.device)
    fp = torch.as_tensor(fp).to(accelerator.device)

    # Find indices of values in original array for each batch
    idxs = torch.searchsorted(xp, x)  # (batch_size, 1)

    # Clip indices to valid range
    idxs = torch.clamp(idxs, 1, xp.shape[1]-1)

    # Calculate weights
    batch_indices = torch.arange(batch_size).unsqueeze(1)  # (batch_size, 1)
    weights = (x - xp[batch_indices, idxs-1]) / \
        (xp[batch_indices, idxs] - xp[batch_indices, idxs-1])

    # Interpolate
    return fp[batch_indices, idxs-1] + weights * (fp[batch_indices, idxs] - fp[batch_indices, idxs-1])


def calc_varrho(taus, varpi_quants, varpi_grid, varpi_pdf, w_t, t, eps=1e-4):
    # Ensure t is positive to avoid division by zero
    t_safe = torch.clamp(t, min=eps)

    # Compute Gaussian CDF/PDF
    z = w_t / torch.sqrt(t_safe)
    rho_gauss = torch.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi))  # Stable Gaussian PDF
    u = 0.5 * (1 + torch.erf(z / np.sqrt(2)))  # Stable Gaussian CDF

    # Clamp u to avoid extrapolation errors (ensure within [eps, 1-eps])
    u_safe = torch.clamp(u, min=eps, max=1 - eps)

    # Interpolate quantile and PDF (extrapolate with nearest values if needed)
    rho_quant = torch_interp(u_safe, torch.tensor(taus), varpi_quants)
    rho_varpi = torch_interp(rho_quant, varpi_grid, varpi_pdf)

    # Regularize varpi PDF to avoid division by zero
    rho_varpi_safe = torch.clamp(rho_varpi, min=eps)

    # Compute varrho (ensure non-negative)
    varrho = rho_gauss / (torch.sqrt(t_safe) * rho_varpi_safe)
    return varrho
