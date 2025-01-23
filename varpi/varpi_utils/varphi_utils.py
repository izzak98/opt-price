import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from utils.dist_utils import generate_smooth_pdf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
                sub_values["observed_returns"].append(obs.cpu().numpy().squeeze())
                sub_values["future_returns"].append(y.cpu().numpy().squeeze())
        if len(sub_values["observed_returns"]) == 0:
            continue
        sub_values["observed_returns"] = np.concatenate(sub_values["observed_returns"], axis=0)
        sub_values["future_returns"] = np.concatenate(sub_values["future_returns"], axis=0)/100
        sub_values["all_pred_quantiles"] = np.concatenate(
            sub_values["all_pred_quantiles"], axis=0)/100

        all_estimated_quantiles[group_name] = sub_values
    return all_estimated_quantiles
