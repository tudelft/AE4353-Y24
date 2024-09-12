import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os


class PolImgDataset(Dataset):
    def __init__(self, dataset_path, prefix="", h5=False, augment=False):
        self.h5 = h5
        self.augment = augment
        if self.h5:
            self.file = h5py.File(
                os.path.join(dataset_path, prefix + "dataset.h5"), "r"
            )
            self.maps = self.file["maps"]
            self.vector = self.file["labels"]
            self.angles = torch.tensor(self.file["angles"][:])
        else:
            self.maps = torch.tensor(
                np.load(os.path.join(dataset_path, prefix + "maps.npy"))
            )
            self.vector = torch.tensor(
                np.load(os.path.join(dataset_path, prefix + "labels.npy"))
            )
            angles = torch.rad2deg(torch.atan2(self.vector[:, 1], self.vector[:, 0]))
            self.angles = torch.remainder(angles, 360)

    def __len__(self):
        return len(self.vector)

    def __getitem__(self, idx):
        if self.h5:
            maps, vector, angles = (
                torch.tensor(self.maps[idx]),
                torch.tensor(self.vector[idx]),
                torch.tensor(self.angles[idx]),
            )
        else:
            maps, vector, angles = self.maps[idx], self.vector[idx], self.angles[idx]

        if self.augment:
            # TODO: Implement data augmentation
            pass

        return maps, vector, angles


if __name__ == "__main__":
    dataset = PolImgDataset(
        "/workspaces/msc-ai-course/data/polarization_dataset", h5=False, augment=True
    )
    sample = dataset[5]
    print(sample)
