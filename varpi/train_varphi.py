import json
import optuna
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from varpi.QLSTM import LSTM_Model
from varpi.varpi_utils.varphi_utils import TwoStageQuantileLoss
from utils.data_utils import collate_fn, DynamicBatchSampler, get_dataset

accelerator = Accelerator()

with open("config.json", "r") as file:
    CONFIG = json.load(file)


def create_model():
    study = optuna.load_study(
        study_name="LSTM",
        storage="sqlite:///optuna_dbs/optuna.db"
    )
    best_params = study.best_params

    model = LSTM_Model(
        lstm_layers=best_params['raw_lstm_layers'],
        lstm_h=best_params['raw_lstm_h'],
        hidden_layers=[best_params[f'raw_hidden_layer_{i}']
                       for i in range(best_params['raw_hidden_layers'])],
        hidden_activation=best_params['hidden_activation'],
        market_lstm_layers=best_params['market_lstm_layers'],
        market_lstm_h=best_params['market_lstm_h'],
        market_hidden_layers=[best_params[f'market_hidden_layer_{i}'] for i in range(
            best_params['market_hidden_layers'])],
        market_hidden_activation=best_params['market_activation'],
        dropout=best_params['dropout'],
        layer_norm=best_params['use_layer_norm']
    )
    return model, best_params


def validade_model(model, criterion, val_loader):
    """Validate the model on the validation set."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        running_len = 0
        for x, s, z, y, sy in val_loader:
            normalized_output, raw_output = model(x, s, z)
            loss = criterion(raw_output, y, normalized_output, sy)
            total_loss += loss.item()
            running_len += 1
    return total_loss / running_len


def train_model(model, l1_reg, optimizer, criterion, train_loader, val_loader, verbose=True):
    patience = 10
    n_epochs = 100
    best_loss = float("inf")
    best_weights = {}
    for e in range(n_epochs):
        model.train()
        total_loss = 0
        count = 0
        if verbose:
            p_bar = tqdm(train_loader, desc="Training", leave=False)
        else:
            p_bar = train_loader
        for X, s, Z, y, sy in p_bar:
            optimizer.zero_grad()
            normalized_output, raw_output = model(X, s, Z)
            loss = criterion(raw_output, y, normalized_output, sy)
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.sum(torch.abs(param))
            loss += l1_reg * l1_loss

            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
            count += 1
            if verbose and isinstance(p_bar, tqdm):
                p_bar.set_postfix({'loss': total_loss / count})
        val_loss = validade_model(model, criterion, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict()
            patience = 10
        else:
            patience -= 1
            if patience == 0:
                break
        if verbose:
            out = (
                f"Epoch {e+1}/{n_epochs} - "
                f"Train Loss: {total_loss / count:.4f} - "
                f"Val Loss: {val_loss:.4f}"
            )
            tqdm.write(out)
    model.load_state_dict(best_weights)
    return best_loss, model


def train(model, best_params):
    batch_size = best_params["batch_size"]
    normalization_window = best_params["normalazation_window"]
    taus = CONFIG["general"]["quantiles"]

    validation_start_date = CONFIG["general"]["dates"]["validation_period"]["start_date"]
    validation_end_date = CONFIG["general"]["dates"]["validation_period"]["end_date"]
    train_dataset = get_dataset(
        normalization_window, "1998-01-01", validation_start_date)
    val_dataset = get_dataset(normalization_window, validation_start_date, validation_end_date)
    train_batch_sampler = DynamicBatchSampler(
        train_dataset, batch_size=batch_size)
    val_batch_sampler = DynamicBatchSampler(val_dataset, batch_size=batch_size)

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_batch_sampler, collate_fn=collate_fn)

    l1_reg = best_params["l1_reg"]
    l2_reg = best_params["l2_reg"]
    lr = best_params["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    criterion = TwoStageQuantileLoss(taus)

    prep_out = (
        f"Model has been prepared for training with the following parameters:\n"
        f"Batch Size: {batch_size}\n"
        f"Normalization Window: {normalization_window}\n"
        f"Validation Start Date: {validation_start_date}\n"
        f"Validation End Date: {validation_end_date}\n"
        f"L1 Regularization: {l1_reg:.6f}\n"
        f"L2 Regularization: {l2_reg:.6f}\n"
        f"Optimizer: Adam\n"
        f"Learning Rate: {lr:.6f}\n"
        f"Train Dataset Size: {len(train_dataset)}\n"
        f"Validation Dataset Size: {len(val_dataset)}\n"
        f"Model has {sum(p.numel() for p in model.parameters())} parameters"
    )
    print(prep_out)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader)

    model.compile()

    best_loss, best_model = train_model(
        model, l1_reg, optimizer, criterion, train_loader, val_loader)

    print(f"Best Validation Loss: {best_loss}")
    torch.save(best_model, "models/varphi.pth")


if __name__ == "__main__":
    model, best_params = create_model()
    train(model, best_params)
