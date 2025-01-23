import pickle
import json
import torch
import numpy as np
from accelerate import Accelerator

from utils.dist_utils import generate_smooth_pdf
from varpi.wasserstein_min import get_best_lstm_pdf_params
from varpi.tain_varpi import VarPi
from varpi.gen_walks import WalkDataSet
from q_storm.storm_utils.varrho import calc_varrho

accelerator = Accelerator()

with open("config.json", "r") as f:
    CONFIG = json.load(f)


class StormTrainer():
    def __init__(self,
                 lambda_p=1,
                 lambda_u=1,
                 lambda_o=1,
                 lambda_i=1,):
        self.lambda_p = lambda_p
        self.lambda_u = lambda_u
        self.lambda_o = lambda_o
        self.lambda_i = lambda_i
        assert all(l >= 1 for l in [lambda_p, lambda_u, lambda_o, lambda_i])

    def residual_loss(
        self,
        model,
        S_prime,
        K_prime,
        t_prime,
        rf,
        varpi_q,
        S_mu,
        S_std,
        T,
        taus,
        varpi_grid,
        varpi_pdf,
        w_t
    ):
        V, S, _, t, _ = model(S_prime, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T)
        varrho = calc_varrho(taus, varpi_q, varpi_grid, varpi_pdf, w_t, t)

        dv_ds = torch.autograd.grad(V, S, torch.ones_like(V), create_graph=True)[0]
        dv2_ds2 = torch.autograd.grad(dv_ds, S, torch.ones_like(V), create_graph=True)[0]
        dv_dt = torch.autograd.grad(V, t, torch.ones_like(V), create_graph=True)[0]

        term1 = -0.5 * dv_ds * S * (varrho)**2
        term2 = -dv_dt
        term3 = -0.5 * dv2_ds2 * S**2 * (varrho)**2
        term4 = dv_ds * rf * S
        term5 = - dv_ds * rf * V
        resdiual = term1 + term2 + term3 + term4 + term5
        return torch.mean(resdiual**2)

    def payoff_loss(self, model, S_prime, K_prime, t_max, rf, varpi_q, S_mu, S_std, T):
        raise NotImplementedError("payoff_loss method must be implemented.")

    def under_loss(self, model, S_prime, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T):
        raise NotImplementedError("under_loss method must be implemented.")

    def over_loss(self, model, S_prime, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T):
        raise NotImplementedError("over_loss method must be implemented.")

    def inequality_loss(self, model, S_prime, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T):
        raise NotImplementedError("inequality_loss method must be implemented.")

    def forward(self, model, sampled_data, taus, mc_samples):
        total_residual_loss = 0
        S_prime = sampled_data["S_prime"]
        K_prime = sampled_data["K_prime"]
        t_prime = sampled_data["t_prime"]
        rf = sampled_data["rf"]
        varpi_q = sampled_data["varpi_q"]
        S_mu = sampled_data["S_mu"]
        S_std = sampled_data["S_std"]
        T = sampled_data["T"]
        varpi_grid = sampled_data["varpi_grids"]
        varpi_pdf = sampled_data["varpi_pdfs"]

        W_t = torch.randn(mc_samples, S_prime.size(0), 1)
        for w_t in W_t:
            residual_loss = self.residual_loss(model,
                                               S_prime,
                                               K_prime,
                                               t_prime,
                                               rf,
                                               varpi_q,
                                               S_mu,
                                               S_std,
                                               T,
                                               taus,
                                               varpi_grid,
                                               varpi_pdf,
                                               w_t
                                               )
            total_residual_loss += residual_loss
        total_residual_loss /= mc_samples

        payoff_loss = self.payoff_loss(model,
                                       S_prime,
                                       K_prime,
                                       t_prime,
                                       rf,
                                       varpi_q,
                                       S_mu,
                                       S_std,
                                       T)

        under_loss = self.under_loss(model,
                                     S_prime,
                                     K_prime,
                                     t_prime,
                                     rf,
                                     varpi_q,
                                     S_mu,
                                     S_std,
                                     T)

        over_loss = self.over_loss(model,
                                   S_prime,
                                   K_prime,
                                   t_prime,
                                   rf,
                                   varpi_q,
                                   S_mu,
                                   S_std,
                                   T)

        inequality_loss = self.inequality_loss(model,
                                               S_prime,
                                               K_prime,
                                               t_prime,
                                               rf,
                                               varpi_q,
                                               S_mu,
                                               S_std,
                                               T)

        total_loss = self.lambda_p * payoff_loss + \
            self.lambda_u * under_loss + \
            self.lambda_o * over_loss + \
            self.lambda_i * inequality_loss + \
            total_residual_loss
        return total_loss


class CallStormTrainer(StormTrainer):
    def __init__(self, lambda_p=1, lambda_u=1, lambda_o=1, lambda_i=1):
        super().__init__(lambda_p, lambda_u, lambda_o, lambda_i)

    def payoff_loss(self, model, S_prime, K_prime, t_max, rf, varpi_q, S_mu, S_std, T):
        assert torch.all(t_max == 1)
        V, S, K, _, _ = model(S_prime, K_prime, t_max, rf, varpi_q, S_mu, S_std, T)
        payoff = V - torch.max(S - K, torch.zeros_like(S))
        return torch.mean(payoff**2)

    def under_loss(self, model, S_prime, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T):
        S_min = -S_mu/(S_std*S_mu)
        V, _, _, _, _ = model(S_min, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T)
        return torch.mean(V**2)

    def over_loss(self, model, S_prime, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T):
        S_max = 10 * (S_std * S_mu) + S_mu
        V, S, K, t = model(S_max, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T)
        max_value = V - (S - K * torch.exp(-rf * (T-t)))
        return torch.mean(max_value**2)

    def inequality_loss(self, model, S_prime, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T):
        V, S, K, _ = model(S_prime, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T)
        inequality = torch.max(V - (S - K), torch.zeros_like(S))
        return torch.mean(inequality**2)


class PutStormTrainer(StormTrainer):
    def __init__(self, lambda_p=1, lambda_u=1, lambda_o=1, lambda_i=1):
        super().__init__(lambda_p, lambda_u, lambda_o, lambda_i)

    def payoff_loss(self, model, S_prime, K_prime, t_max, rf, varpi_q, S_mu, S_std, T):
        assert torch.all(t_max == 1)
        V, S, K, _, _ = model(S_prime, K_prime, t_max, rf, varpi_q, S_mu, S_std, T)
        payoff = V - torch.max(K - S, torch.zeros_like(S))
        return torch.mean(payoff**2)

    def under_loss(self, model, S_prime, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T):
        S_min = -S_mu/(S_std*S_mu)
        V, _, K, t = model(S_min, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T)
        max_value = V - K * torch.exp(-rf * (T-t))
        return torch.mean(max_value**2)

    def over_loss(self, model, S_prime, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T):
        S_max = 10 * (S_std * S_mu) + S_mu
        V, _, _, _ = model(S_max, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T)
        return torch.mean(V**2)

    def inequality_loss(self, model, S_prime, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T):
        V, S, K, _ = model(S_prime, K_prime, t_prime, rf, varpi_q, S_mu, S_std, T)
        inequality = torch.max(V - (K - S), torch.zeros_like(S))
        return torch.mean(inequality**2)


def sample_optimized(n_points, quantiles, pdf_params, taus, varPi: VarPi):
    """
    Optimized version of the sampling function with NaN filtering
    """
    device = accelerator.device
    taus_tensor = torch.tensor(taus, device=device)

    # Oversample initially to account for NaN filtering
    oversample_factor = 1.2  # Sample 20% more than needed
    n_oversample = int(n_points * oversample_factor)

    # Batch sample indices and quantiles
    sampled_idx = torch.randint(0, len(quantiles), (n_oversample,))
    sampled_quantiles = quantiles[sampled_idx] + \
        torch.randn(n_oversample, quantiles.size(1)) * 0.001

    # Price range sampling
    min_price, max_price = 1.07, 432.07
    S_mu = torch.rand(n_oversample, 1, device=device) * (max_price - min_price) + min_price

    # Batch process varphi statistics with NaN checking
    def batch_pdf_stats(quantiles_batch):
        stats = []
        valid_indices = []
        for i, quants in enumerate(quantiles_batch):
            try:
                grid, pdf, _ = generate_smooth_pdf(
                    quants.cpu().numpy(), np.array(taus), **pdf_params)
                if not (np.isnan(pdf).any() or np.isnan(grid).any()):
                    mean = np.trapezoid(grid * pdf, grid)
                    std = np.sqrt(np.trapezoid((grid - mean)**2 * pdf, grid))
                    stats.append([std])
                    valid_indices.append(i)
            except Exception:
                continue
        return torch.tensor(stats, device=device), valid_indices

    # Get valid indices from initial PDF computation
    varphi_sigmas, valid_indices = batch_pdf_stats(sampled_quantiles)

    if len(valid_indices) < n_points:
        raise ValueError(
            f"Not enough valid samples after NaN filtering. Got {len(valid_indices)}, needed {n_points}")

    # Take only the first n_points valid samples
    valid_indices = valid_indices[:n_points]
    varphi_sigmas = varphi_sigmas[:n_points]

    # Filter all tensors to keep only valid indices
    sampled_quantiles = sampled_quantiles[valid_indices]
    S_mu = S_mu[valid_indices]

    S_std_scaling = torch.randn(n_points, 1, device=device) * (varphi_sigmas/10)
    S_std = varphi_sigmas + S_std_scaling

    # Time sampling
    T = torch.randint(15, 31, (n_points, 1), dtype=torch.float, device=device)
    T_prime = T/30
    t_prime = torch.rand(n_points, 1, device=device)
    t = t_prime * T

    # VarPi computation
    sampled_quantiles = sampled_quantiles.to(accelerator.device)
    t_prime = t_prime.to(accelerator.device)
    T_prime = T_prime.to(accelerator.device)
    varpi_quantiles = varPi(sampled_quantiles, t_prime, T_prime)

    # Batch process varpi PDFs
    def batch_compute_pdfs(quantiles_batch):
        grids, pdfs = [], []
        valid_mask = torch.ones(len(quantiles_batch), dtype=torch.bool)
        for i, quants in enumerate(quantiles_batch):
            try:
                grid, pdf, _ = generate_smooth_pdf(
                    quants.cpu().numpy(), np.array(taus), **pdf_params)
                if np.isnan(pdf).any() or np.isnan(grid).any():
                    valid_mask[i] = False
                    continue
                grids.append(grid)
                pdfs.append(pdf)
            except Exception:
                valid_mask[i] = False
        return (torch.tensor(grids, device=device),
                torch.tensor(pdfs, device=device),
                valid_mask)

    # Compute varpi PDFs and get final valid mask
    varpi_grids, varpi_pdfs, final_valid_mask = batch_compute_pdfs(varpi_quantiles)

    # Apply final filtering if needed
    if not torch.all(final_valid_mask):
        n_final = torch.sum(final_valid_mask).item()
        if n_final < n_points:
            raise ValueError(
                f"Too many samples filtered in final PDF generation. Got {n_final}, needed {n_points}")

        # Filter all tensors
        sampled_quantiles = sampled_quantiles[final_valid_mask]
        S_mu = S_mu[final_valid_mask]
        S_std = S_std[final_valid_mask]
        T = T[final_valid_mask]
        T_prime = T_prime[final_valid_mask]
        t_prime = t_prime[final_valid_mask]
        varpi_quantiles = varpi_quantiles[final_valid_mask]
        varpi_grids = varpi_grids[final_valid_mask]
        varpi_pdfs = varpi_pdfs[final_valid_mask]

    # Random sampling
    S_prime = torch.randn(n_points, 1, device=device)
    K_prime = torch.randn(n_points, 1, device=device)

    # Risk-free rate sampling
    rf_min, rf_max = 0, 0.05348000
    rf = torch.rand(n_points, 1, device=device) * (rf_max - rf_min) + rf_min

    return {
        "S_prime": S_prime,
        "K_prime": K_prime,
        "t_prime": t_prime,
        "rf": rf,
        "varpi_q": varpi_quantiles,
        "S_mu": S_mu,
        "S_std": S_std,
        "T": T,
        "varpi_pdfs": varpi_pdfs,
        "varpi_grids": varpi_grids
    }


def main():
    varPi = torch.load('models/varpi.pth')
    varPi.eval()
    varPi.requires_grad_(False)
    # load walk data
    with open("walk_dataset.pkl", "rb") as f:
        walk_dataset = pickle.load(f)
    quantiles = walk_dataset.X
    quantiles = list({tuple(arr) for arr in quantiles})
    quantiles = [torch.tensor(arr) for arr in quantiles]
    varphi_quantiles = torch.stack(quantiles)
    del walk_dataset, quantiles
    pdf_parms = get_best_lstm_pdf_params(None)
    taus = CONFIG["general"]["quantiles"]
    n_points = 100
    varPi, varphi_quantiles = accelerator.prepare(varPi, varphi_quantiles)
    sample_data = sample_optimized(n_points, varphi_quantiles, pdf_parms, taus, varPi)
    print(sample_data)


if __name__ == "__main__":
    main()
