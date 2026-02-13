import torch
from torch.utils.data import DataLoader
from torch import Tensor
import numpy as np
from tqdm import tqdm
from utils.dist_utils import generate_smooth_pdf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quantile_loss(y_true, y_pred, tau) -> Tensor:
    """
    Calculate the quantile loss for a single quantile.

    Args:
    y_true (torch.Tensor): True values (batch_size, seq_len, 1)
    y_pred (torch.Tensor): Predicted values (batch_size, 1)
    tau (float): Quantile value

    Returns:
    torch.Tensor: Quantile loss
    """
    diff = y_true - y_pred.view(-1, 1).unsqueeze(1)  # Expand y_pred to (batch_size, seq_len, 1)
    return torch.mean(torch.max(tau * diff, (tau - 1) * diff))


def aggregate_quantile_loss(y_true, y_pred, taus) -> Tensor:
    """
    Calculate the aggregated quantile loss for multiple quantiles.

    Args:
    y_true (torch.Tensor): True values (batch_size, seq_len, 1)
    y_pred (torch.Tensor): Predicted values (batch_size, num_quantiles)
    taus (list or torch.Tensor): List of quantile values

    Returns:
    torch.Tensor: Aggregated quantile loss
    """
    losses = [quantile_loss(y_true, y_pred[:, i], tau) for i, tau in enumerate(taus)]
    return torch.mean(torch.stack(losses))


class TwoStageQuantileLoss(torch.nn.Module):
    """Quantile loss for two-stage training."""

    def __init__(self, taus) -> None:
        super().__init__()
        self.taus = taus

    def forward(self, y_pred_raw, y_true_raw, y_pred_std, y_true_std) -> Tensor:
        """
        Calculate the two-stage quantile loss.

        Args:
        y_pred_raw (torch.Tensor): Predicted raw returns (batch_size, num_quantiles)
        y_true_raw (torch.Tensor): True raw returns (batch_size, seq_len, 1)
        y_pred_std (torch.Tensor): Predicted standardized returns (batch_size, num_quantiles)
        y_true_std (torch.Tensor): True standardized returns (batch_size, seq_len, 1)

        Returns:
        torch.Tensor: Two-stage quantile loss
        """
        raw_loss = aggregate_quantile_loss(y_true_raw, y_pred_raw, self.taus)
        std_loss = aggregate_quantile_loss(y_true_std, y_pred_std, self.taus)

        return (raw_loss + std_loss)


def get_all_quantiles(data_set, model):
    all_estimated_quantiles = {}
    group_names = [
        "cryptocurrencies",
        "currency pairs",
        "commodities",
        "euro stoxx 50",
        "s&p 500",
        "nikkei 225"
    ]
    for group_name in tqdm(group_names, desc='Inferencing models'):
        group_assets = [a["asset"] for a in data_set.datas[group_name]]
        sub_values = {
            "observed_returns": [],
            "future_returns": [],
            "all_pred_quantiles": [],
        }
        for asset in group_assets:
            data_set.set_main_asset(asset)
            data_loader = DataLoader(data_set, batch_size=1024, shuffle=False)
            model.eval()
            for x, s, z, y, obs in data_loader:
                x, s, z, y = x.to(DEVICE), s.to(DEVICE), z.to(DEVICE), y.to(DEVICE)
                s = s.mean(dim=1)

                with torch.no_grad():
                    _, estimated_quantiles = model(x, s, z)
                sub_values["all_pred_quantiles"].append(estimated_quantiles.detach().cpu().numpy())
                # Convert to numpy and ensure consistent 1D shape for concatenation
                # obs and y have shape (batch_size, 1) from the dataset
                # We need to ensure all arrays are 1D before concatenation
                obs_np = obs.cpu().numpy()
                y_np = y.cpu().numpy()
                # Reshape to ensure 1D: (batch_size, 1) -> (batch_size,)
                # Handle edge cases where batch_size might be 1
                if obs_np.ndim > 1:
                    obs_np = obs_np.reshape(-1)
                elif obs_np.ndim == 0:
                    obs_np = obs_np.reshape(1)
                if y_np.ndim > 1:
                    y_np = y_np.reshape(-1)
                elif y_np.ndim == 0:
                    y_np = y_np.reshape(1)
                sub_values["observed_returns"].append(obs_np)
                sub_values["future_returns"].append(y_np)
        if len(sub_values["observed_returns"]) == 0:
            continue
        sub_values["observed_returns"] = np.concatenate(sub_values["observed_returns"], axis=0)
        sub_values["future_returns"] = np.concatenate(sub_values["future_returns"], axis=0)/100
        sub_values["all_pred_quantiles"] = np.concatenate(
            sub_values["all_pred_quantiles"], axis=0)/100

        all_estimated_quantiles[group_name] = sub_values
    return all_estimated_quantiles
