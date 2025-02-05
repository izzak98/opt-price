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
from torch.utils.tensorboard import SummaryWriter

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
                verbose=True,
                use_lfbgs=True):

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='runs/storm_training')

    best_loss = float("inf")
    best_model_weights = model.state_dict()
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

        if use_lfbgs:
            def closure():
                optimizer.zero_grad()
                losses = trainer.forward(model, sampled_data, taus, 5)
                loss = losses["total_loss"]
                accelerator.backward(loss)
                # Log L-BFGS iteration losses
                writer.add_scalar('Loss/train_step', loss.item(), e * n_batches)
                return loss

        try:
            if use_lfbgs:
                optimizer.step(closure)
            else:
                batch_losses = []
                for b in range(n_batches):
                    optimizer.zero_grad()
                    batch_data = {k: v[b * batch_size:(b + 1) * batch_size]
                                  for k, v in sampled_data.items()}
                    losses = trainer.forward(model, batch_data, taus, 5)
                    loss = losses["total_loss"]
                    accelerator.backward(loss)
                    optimizer.step()
                    total_loss += loss.item()
                    batch_losses.append(loss.item())

                    # Log batch-level metrics
                    writer.add_scalar('Loss/train_step', loss.item(), e * n_batches + b)

                # Log batch statistics
                writer.add_scalar('Loss/batch_mean', np.mean(batch_losses), e)
                writer.add_scalar('Loss/batch_std', np.std(batch_losses), e)

        except Exception as err:
            print(err)
            continue

        if use_lfbgs:
            losses = trainer.forward(model, sampled_data, taus, 1)
            total_loss = losses["total_loss"]

        # Log epoch-level metrics
        writer.add_scalar('Loss/train_epoch', total_loss, e)
        if scheduler is not None:
            writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], e)

        if verbose and isinstance(pbar, tqdm):
            losses = {k: v.item() for k, v in losses.items()}
            pbar.set_postfix(losses)
            # Log individual loss components
            for loss_name, loss_value in losses.items():
                writer.add_scalar(f'Losses/{loss_name}', loss_value, e)

        if scheduler is not None:
            scheduler.step(total_loss)

        if total_loss < best_loss:
            best_loss = total_loss
            best_model_weights = model.state_dict()
            writer.add_scalar('Best_loss', best_loss, e)

    # Close TensorBoard writer
    writer.close()
    model.load_state_dict(best_model_weights)
    return best_loss, model


def main():
    # Create runs directory if it doesn't exist
    os.makedirs('runs', exist_ok=True)

    # Rest of your main function remains the same
    taus = CONFIG["general"]["quantiles"]
    n_points = 10000
    q_storm = QStorm(40, [1054, 1054, 256, 64, 32], 32, 4, 'relu')
    sampler = StormSampler(accelerator.device, taus)
    optimizer = torch.optim.Adam(q_storm.parameters(), lr=0.001)
    scheduler = None
    trainer = CallStormTrainer()

    q_storm, optimizer, scheduler = accelerator.prepare(q_storm, optimizer, scheduler)

    best_loss, q_storm = train_storm(
        model=q_storm,
        optimizer=optimizer,
        trainer=trainer,
        scheduler=scheduler,
        n_epochs=100,
        batch_size=10000,
        n_points=n_points,
        taus=taus,
        sampler=sampler,
        use_lfbgs=False)

    print(f"Training completed with best loss: {best_loss}")

    save_path = "models/q_storm.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(q_storm, save_path)


if __name__ == "__main__":
    main()
