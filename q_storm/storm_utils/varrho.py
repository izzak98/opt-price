import torch
import numpy as np


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
    PyTorch equivalent of np.interp
    x: points to evaluate at
    xp: known x values
    fp: known y values
    """
    if isinstance(x, float):
        x = torch.tensor([x])
    if isinstance(xp, (list, np.ndarray)):
        xp = torch.tensor(xp)
    if isinstance(fp, (list, np.ndarray)):
        fp = torch.tensor(fp)

    # Find indices of values in original array
    idxs = torch.searchsorted(xp, x)

    # Clip indices to valid range
    idxs = torch.clamp(idxs, 1, len(xp)-1)

    # Calculate weights
    weights = (x - xp[idxs-1]) / (xp[idxs] - xp[idxs-1])

    # Interpolate
    return fp[idxs-1] + weights * (fp[idxs] - fp[idxs-1])


def calc_varrho(taus, varpi_quants, varpi_grid, varpi_pdf, w_t, t):
    u = fast_normal_cdf(w_t/(t**0.5))
    rho_quant = torch_interp(u, torch.tensor(taus), varpi_quants)
    rho_varpi = torch_interp(rho_quant, varpi_grid, varpi_pdf)**(-1)
    rho_gauss = gaussian_pdf(w_t/(t**0.5))
    rho_term = 1/(t**0.5)
    varrho = rho_term * rho_varpi * rho_gauss
    return varrho
