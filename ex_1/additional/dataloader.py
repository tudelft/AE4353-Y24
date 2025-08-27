import torch
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
import numpy as np


class TrajectoryDataset(Dataset):
    """
    A custom PyTorch Dataset class to handle 2D and 3D trajectory datasets for model training.

    Parameters:
        dataset (dict): A dictionary containing the dataset.
        indices (list[int]): Indices of trajectories to include in this dataset.
        d (int): Dimension of the dataset. Must be either 2 or 3.
        flatten (bool): If True, flattens the dataset such that each sample in a trajectory is treated separately.

    Attributes:
        indices (list[int]): Indices of trajectories to include.
        trajectory_length (int): Length of each trajectory.
        flatten (bool): If True, flattens the dataset.
        data (list[dict]): List of dictionaries containing 'X' (inputs) and 'U' (controls) tensors.
    """

    def __init__(
        self,
        dataset: dict[str, torch.Tensor],
        indices: list[int],
        d: int,
        flatten: bool = False,
    ) -> None:
        self.indices = indices
        self.trajectory_length = len(dataset["t"][0])
        self.flatten = flatten
        self.data = []

        # Define input keys for 2D and 3D cases
        if d == 2:
            input_keys_2d = ["y", "z", "vy", "vz", "theta", "omega"]

            # Assemble dataset for 2D trajectories
            for i in tqdm(indices, desc=">>> Assembling Dataset (2D)"):
                input_tensor = []
                for key in input_keys_2d:
                    # Flatten tensors with multiple dimensions
                    if dataset[key][i].ndim == 1:
                        input_tensor.append(dataset[key][i])
                    else:
                        input_tensor.extend(torch.unbind(dataset[key][i], dim=-1))
                # Stack inputs and controls
                X = torch.stack(input_tensor, dim=1)
                U = torch.stack([dataset["ul"][i], dataset["ur"][i]], dim=1)
                self.data.append({"X": X, "U": U})

        elif d == 3:
            input_keys_3d = [
                "dx",
                "dy",
                "dz",
                "vx",
                "vy",
                "vz",
                "phi",
                "theta",
                "psi",
                "p",
                "q",
                "r",
                "omega",
            ]

            # Assemble dataset for 3D trajectories
            for i in tqdm(indices, desc=">>> Assembling Dataset (3D)"):
                input_tensor = []
                for key in input_keys_3d:
                    # Flatten tensors with multiple dimensions
                    if dataset[key][i].ndim == 1:
                        input_tensor.append(dataset[key][i])
                    else:
                        input_tensor.extend(torch.unbind(dataset[key][i], dim=-1))
                # Stack inputs and controls
                X = torch.stack(input_tensor)
                U = torch.stack([*torch.unbind(dataset["u"][i], dim=-1)])
                self.data.append({"X": X, "U": U})

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        if self.flatten:
            return len(self.indices) * self.trajectory_length
        else:
            return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample or trajectory from the dataset.

        Parameters:
            index (int): Index of the sample or trajectory to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Input (X) and control (U) tensors.
        """
        if self.flatten:
            # Calculate trajectory and sample index for flattened datasets
            trajectory_idx = index // self.trajectory_length
            sample_idx = index % self.trajectory_length
            return (
                self.data[trajectory_idx]["X"][sample_idx],
                self.data[trajectory_idx]["U"][sample_idx],
            )
        else:
            return self.data[index]["X"], self.data[index]["U"]


def prepare_dataset(
    file_path: str, d: int
) -> tuple[TrajectoryDataset, TrajectoryDataset]:
    """
    Loads and prepares the trajectory dataset for training and validation.

    Parameters:
        file_path (str): Path to the dataset file (in .npz format).
        d (int): Dimension of the dataset. Must be either 2 or 3.

    Returns:
        tuple[TrajectoryDataset, TrajectoryDataset]: Training and validation datasets.
    """
    assert d in [2, 3], "dimension must be 2 or 3"

    # Load the dataset
    with np.load(file_path) as full_dataset:
        # Total number of trajectories
        num = len(full_dataset["t"])

        # Split dataset into training and validation sets
        train_indices = list(range(int(0.8 * num)))
        val_indices = list(set(range(num)) - set(train_indices))
        print(num, ">>> Total Trajectories")
        print(len(train_indices), " >>> Training Set Trajectories")
        print(len(val_indices), " >>> Validation Set Trajectories")

        # Define keys to load based on dimension
        if d == 2:
            keys_to_load = ["t", "y", "z", "vy", "vz", "theta", "omega", "ul", "ur"]

        elif d == 3:
            keys_to_load = [
                "t",
                "dx",
                "dy",
                "dz",
                "vx",
                "vy",
                "vz",
                "phi",
                "theta",
                "psi",
                "p",
                "q",
                "r",
                "omega",
                "u",
                "omega_min",
                "omega_max",
                "k_omega",
                "Mx_ext",
                "My_ext",
                "Mz_ext",
            ]

        # Convert loaded dataset to PyTorch tensors
        dataset = {
            key: torch.tensor(full_dataset[key], dtype=torch.float32)
            for key in tqdm(keys_to_load, desc=">>> Loading Dataset")
        }

    # Create training and validation datasets
    train_set = TrajectoryDataset(dataset, train_indices, d, flatten=True)
    val_set = TrajectoryDataset(dataset, val_indices, d, flatten=False)

    return train_set, val_set