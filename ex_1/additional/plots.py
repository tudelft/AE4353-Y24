import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset
# Number of worst predictions to plot
top_k = 5


def plot_predictions(
    model: nn.Module,
    val_set: Dataset,
    top_indices: torch.Tensor,
    step: int,
) -> None:
    """
    Plot the worst predictions based on validation loss.

    Parameters:
        model (nn.Module): Trained PyTorch model.
        val_set (Dataset): Validation dataset.
        top_indices (torch.Tensor): Indices of top-k worst predictions.
        step (int): Current training step.
    """
    fig, axs = plt.subplots(2, len(top_indices), figsize=(12, 6))

    for i, idx in enumerate(top_indices):
        # Fetch trajectory and ground truth values
        x_traj, u = val_set[idx]
        # Predict using the model
        u_pred = model(x_traj)

        # Plot ground truth and predicted values for both components of the output
        axs[0, i].plot(range(len(x_traj)), u[:, 0], label="l-gt")
        axs[0, i].plot(range(len(x_traj)), u_pred[:, 0], label="l-pred")
        axs[1, i].plot(range(len(x_traj)), u[:, 1], label="r-gt")
        axs[1, i].plot(range(len(x_traj)), u_pred[:, 1], label="r-pred")
        # Add text annotation for each subplot
        axs[0, i].text(
            0.5,
            0.5,
            f"traj {idx}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[0, i].transAxes,
        )
        # Add legend to the last plot for both components
        if i == len(top_indices) - 1:
            axs[0, i].legend()
            axs[1, i].legend()

    plt.suptitle(f"{top_k} Worst Predictions at Step={step}")
    plt.tight_layout()
    plt.show()


def plot_percentiles(
    model: nn.Module,
    val_set: Dataset,
    val_loss: torch.Tensor,
    percentiles: list[int],
) -> None:
    """
    Plot predictions at specified percentiles of the validation loss distribution.

    Parameters:
        model (nn.Module): Trained PyTorch model.
        val_set (Dataset): Validation dataset.
        val_loss (torch.Tensor): Computed validation loss for each sample.
        percentiles (list[int]): List of percentiles to plot.
    """
    # Sort losses to find indices for each percentile
    _, sorted_indices = torch.sort(val_loss)
    # Compute positions for each percentile in the sorted indices
    positions = [(p / 100.0) * (len(val_loss) - 1) for p in percentiles]
    fig, axs = plt.subplots(2, len(percentiles), figsize=(12, 6))

    # Iterate over each percentile and plot the corresponding predictions
    for i, (idx, percentile) in enumerate(zip(positions, percentiles)):
        idx = int(idx)
        # Retrieve trajectory and ground truth values
        x_traj, u = val_set[sorted_indices[idx]]
        # Predict using the model
        u_pred = model(x_traj)

        # Plot ground truth and predicted values for both components of the output
        axs[0, i].plot(range(len(x_traj)), u[:, 0], label="l-gt")
        axs[0, i].plot(range(len(x_traj)), u_pred[:, 0], label="l-pred")
        axs[1, i].plot(range(len(x_traj)), u[:, 1], label="r-gt")
        axs[1, i].plot(range(len(x_traj)), u_pred[:, 1], label="r-pred")
        # Add text annotation and title for each subplot
        axs[0, i].text(
            0.5,
            0.5,
            f"traj {sorted_indices[idx]}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[0, i].transAxes,
        )
        axs[0, i].set_title(f"{percentile}th Percentile")
        # Add legend to the last plot for both components
        if i == len(positions) - 1:
            axs[0, i].legend()
            axs[1, i].legend()

    plt.tight_layout()
    plt.show()