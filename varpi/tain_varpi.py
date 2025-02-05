import os
import torch
from torch import nn
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import random_split, DataLoader

accelerator = Accelerator()


class VarPi(nn.Module):
    def __init__(self, hidden_layers: list[int]):
        super().__init__()
        self.input_layer = nn.Linear(37+1+1, hidden_layers[0])
        self.layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.ReLU())
        self.output_layer = nn.Linear(hidden_layers[-1], 37)

    def forward(self, q, t, T):
        X = torch.cat([q, t, T], dim=1)
        f = self.input_layer(X)
        for layer in self.layers:
            f = layer(f)
        u = self.output_layer(f)
        u, _ = torch.sort(u)
        return u


def mae_loss(q_pred, q_true):
    return torch.mean(torch.abs(q_pred - q_true))


def train_varpi(model: VarPi,
                optimzer,
                scheduler,
                criterion,
                train_data_loader,
                val_data_loader,
                epochs,
                patience,
                verbose=True):
    model.train()
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = model.state_dict()
    for e in range(epochs):
        total_loss = 0
        counter = 0
        if verbose:
            p_bar = tqdm(train_data_loader, desc=f"Epoch {e+1}/{epochs}", leave=False)
        else:
            p_bar = train_data_loader
        for q, t, T, tq in p_bar:
            optimzer.zero_grad()
            u = model(q, t, T)
            loss = criterion(u, tq)

            accelerator.backward(loss)
            optimzer.step()
            total_loss += loss.item()

            if verbose:
                p_bar.set_postfix({"loss": total_loss / (counter + 1)})
            counter += 1
        if scheduler is not None:
            scheduler.step()
        val_loss = validate_varpi(model, criterion, val_data_loader, verbose=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        out = (
            f"Epoch {e+1}/{epochs} "
            f"Train Loss: {total_loss / len(train_data_loader):.4f} "
            f"Val Loss: {val_loss:.4f}"
        )
        tqdm.write(out)

    model.load_state_dict(best_weights)
    return best_val_loss, model


def validate_varpi(model, criterion, val_data_loader, verbose):
    total_loss = 0
    model.eval()
    if verbose:
        p_bar = tqdm(val_data_loader, desc="Validating Model", leave=False)
    else:
        p_bar = val_data_loader
    with torch.no_grad():
        for q, t, T, tq in p_bar:
            u = model(q, t, T)
            loss = criterion(u, tq)
            total_loss += loss.item()
    return total_loss / len(val_data_loader)


def main(varpi_dataset, batch_size=1024):
    hidden_dims = [4096, 4096, 4096, 2048, 2048, 1024, 256, 256, 64, 32]
    model = VarPi(hidden_dims)
    if os.path.exists("models/varpi.pth"):
        model = torch.load("models/varpi.pth")
        return model
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = mae_loss
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, 0.5)

    dataset_size = len(varpi_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, _ = random_split(
        varpi_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model, optim, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optim, train_loader, val_loader, scheduler)
    model.compile()

    best_loss, model = train_varpi(
        model=model,
        optimzer=optim,
        scheduler=scheduler,
        criterion=criterion,
        train_data_loader=train_loader,
        val_data_loader=val_loader,
        epochs=100,
        patience=10,
        verbose=True
    )
    torch.save(model, "models/varpi.pth")
    return model


if __name__ == "__main__":
    from varpi.gen_walks import get_walk_dataset, WalkDataSet
    varpi_dataset = get_walk_dataset()
    model = main(varpi_dataset)
    print(model)
    q = varpi_dataset[-1][0].view(1, -1).to("cuda")
    t = varpi_dataset[-1][1].view(1, -1).to("cuda")
    T = varpi_dataset[-1][2].view(1, -1).to("cuda")
    print(model(q, t, T))
