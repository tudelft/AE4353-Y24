# plot_utils.py

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset


def plot_multiple_sample_predictions(model: nn.Module, val_set: Dataset, num_samples: int = 4):
    """
    Plot ground truth vs. predicted values for multiple trajectories from the validation set.

    Parameters:
        model (nn.Module): Trained model.
        val_set (Dataset): Validation dataset.
        num_samples (int): Number of trajectories to plot (default: 4).
    """
    model.eval()
    total = len(val_set)
    indices = torch.linspace(0, total - 1, steps=num_samples).int().tolist()

    with torch.no_grad():
        for i, idx in enumerate(indices):
            x_traj, u_true = val_set[idx]
            u_pred = model(x_traj)

            time = range(len(x_traj))
            fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

            axs[0].plot(time, u_true[:, 0], label="Left - Ground Truth", color="blue")
            axs[0].plot(time, u_pred[:, 0], label="Left - Prediction", color="orange")
            axs[0].set_ylabel("Left Output")
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(time, u_true[:, 1], label="Right - Ground Truth", color="blue")
            axs[1].plot(time, u_pred[:, 1], label="Right - Prediction", color="orange")
            axs[1].set_ylabel("Right Output")
            axs[1].set_xlabel("Time Step")
            axs[1].legend()
            axs[1].grid(True)

            plt.suptitle(f"Model Fit on Validation Sample #{idx}")
            plt.tight_layout()
            plt.show()
