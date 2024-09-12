import h5py
from ex_2.dataset import PolImgDataset


real_dataset = PolImgDataset(
    "/workspaces/msc-ai-course/data/polarization_dataset", "", h5=False
)

with h5py.File(
    "/workspaces/msc-ai-course/data/polarization_dataset/dataset.h5", "w"
) as f:
    f.create_dataset("maps", data=real_dataset.maps.numpy())
    f.create_dataset("labels", data=real_dataset.vector.numpy())
    f.create_dataset("angles", data=real_dataset.angles.numpy())
