import numpy as np
import json
import os
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from utils.dist_utils import generate_smooth_pdf
from varpi.wasserstein_min import get_best_lstm_pdf_params

with open('config.json') as f:
    CONFIG = json.load(f)


def generate_random_walk(grid, cdf, T):
    random_uniform_samples = np.random.uniform(0, 1, T)
    predicted_samples = np.interp(random_uniform_samples, cdf, grid)
    predicted_samples = np.concatenate([[0], predicted_samples])
    discrete_random_walk = np.cumsum(predicted_samples)
    return discrete_random_walk


def gen_quantiles(grid, cdf, num_samples, T, taus):
    walks = []
    for _ in range(num_samples):
        walks.append(generate_random_walk(grid, cdf, T))
    walks = np.array(walks)
    quantiles = [np.array([0]*len(taus))]
    for i in range(1, T):
        sub_quantiles = np.quantile(walks[:, i], taus)
        quantiles.append(sub_quantiles)
    quantiles = np.array(quantiles)
    return quantiles


def load_quants(t):
    folder_name = "stored_quants"
    file_name = f"quantiles_{t}.pkl"
    file_path = os.path.join(folder_name, file_name)
    with open(file_path, "rb") as f:
        quants = pickle.load(f)
    return quants


class WalkDataSet(Dataset):
    def __init__(self):
        self.X = []
        self.t = []
        self.T = []
        self.Y = []
        self.max_T = 0

    def ingest(self, varphi_quants, varpi_quants, T):
        # varphi shape (num_quantiles)
        # varpi shape (T, num_quantiles)
        # T is the number of time steps (max 30)
        if self.max_T < T:
            self.max_T = T
        for t in range(T):
            self.X.append(varphi_quants)
            self.t.append(t)
            self.T.append(T)
            self.Y.append(varpi_quants[t])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        X = torch.tensor(X, dtype=torch.float32)

        T = self.T[idx]

        t = self.t[idx]
        t = torch.tensor([t/T], dtype=torch.float32)

        T = torch.tensor([T/30], dtype=torch.float32)

        Y = self.Y[idx]

        return X, t, T, Y


def get_walk_dataset():
    already_exists = os.path.exists("walk_dataset.pkl")
    if already_exists:
        with open("walk_dataset.pkl", "rb") as f:
            dataset = pickle.load(f)
        if dataset.max_T == 30:
            return dataset
        else:
            starting_t = dataset.max_T + 1
    else:
        starting_t = 15
    best_pdf_params = get_best_lstm_pdf_params(None)
    taus = CONFIG["general"]["quantiles"]
    taus = np.array(taus)
    dataset = WalkDataSet()
    for t in tqdm(range(starting_t, 31), desc="Generating Walks"):
        quants = load_quants(t)
        for asset_class in tqdm(quants.keys(), desc="Asset Classes", leave=False):
            varphi_quants = quants[asset_class]["all_pred_quantiles"]
            for varphi_quant in tqdm(varphi_quants, desc="Quantiles", leave=False):
                grid, _, cdf = generate_smooth_pdf(varphi_quant, taus, **best_pdf_params)
                varpi_quants = gen_quantiles(grid, cdf, 10000, t, taus)
                dataset.ingest(varphi_quant, varpi_quants, t)
        with open("walk_dataset.pkl", "wb") as f:
            pickle.dump(dataset, f)
    return dataset


if __name__ == "__main__":
    dataset = get_walk_dataset()
    print(len(dataset))
