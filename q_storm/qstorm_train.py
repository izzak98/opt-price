import os
import pickle
import json
from tqdm import tqdm
import torch
import numpy as np
from accelerate import Accelerator
from q_storm.qStorm import QStorm
from q_storm.StormSampler import StormSampler
from q_storm.StormTrainer import CallStormTrainer
from varpi.tain_varpi import VarPi

accelerator = Accelerator()

with open("config.json", "r") as f:
    CONFIG = json.load(f)

torch.autograd.set_detect_anomaly(True)


def train_storm(model,
                optimizer,
                trainer,
                scheduler,
                n_epochs,
                batch_size,
                n_points,
                taus,
                sampler,
                verbose=True):
    model.train()
    if verbose:
        pbar = tqdm(range(n_epochs), desc="Training")
    else:
        pbar = range(n_epochs)
    sampled_data = sampler.sample(n_points)
    for e in pbar:
        if e % 10 == 0 and e > 0:
            sampled_data = sampler.sample(n_points)
        n_samples = len(sampled_data["S_prime"])
        n_batches = n_samples // batch_size
        n_batches = max(n_batches, 1)
        total_loss = 0

        # L-BFGS works with full batches instead of mini-batches
        def closure():
            optimizer.zero_grad()
            losses = trainer.forward(model, sampled_data, taus, 5)
            loss = losses["total_loss"]
            accelerator.backward(loss)
            return loss
        try:
            optimizer.step(closure)
        except Exception as e:
            print(e)
            continue

        losses = trainer.forward(model, sampled_data, taus, 1)
        total_loss = losses["total_loss"]

        if verbose and isinstance(pbar, tqdm):
            losses = {k: v.item() for k, v in losses.items()}
            pbar.set_postfix(losses)

        if scheduler is not None:
            scheduler.step(total_loss)
        # if total_loss < best_loss:
        #     best_loss = total_loss
        #     best_model_weights = model.state_dict()
    last_loss = total_loss
    return last_loss, model


def main():
    # load walk data
    taus = CONFIG["general"]["quantiles"]
    n_points = 10000
    q_storm = QStorm(40, [1054, 1054, 256, 64, 32], 32, 4, 'relu')
    sampler = StormSampler(accelerator.device, taus)

    # Use L-BFGS as the optimizer
    optimizer = torch.optim.LBFGS(q_storm.parameters(), lr=0.01, max_iter=20, tolerance_grad=1e-7)

    scheduler = None  # Optionally include a scheduler if desired
    trainer = CallStormTrainer()

    q_storm, optimizer, scheduler = accelerator.prepare(q_storm, optimizer, scheduler)

    best_loss, q_storm = train_storm(
        model=q_storm,
        optimizer=optimizer,
        trainer=trainer,
        scheduler=scheduler,
        n_epochs=600,
        batch_size=1024,  # Batch size is irrelevant for L-BFGS
        n_points=n_points,
        taus=taus,
        sampler=sampler,)
    print(best_loss)
    save_path = "models/q_storm.pth"
    torch.save(q_storm, save_path)


if __name__ == "__main__":
    main()
