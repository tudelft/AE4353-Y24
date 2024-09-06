1. Uncomment the "runArgs" section in `.devcontainer/devcontainer.json`.
2. Uncomment the lines for the `nvidia` channel and the `pytorch-cuda` package in the `env.yml` file.
3. Install Nvidia Container Toolkit and configure Docker by following this [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). You only need to do the steps under "Installing with Apt" and "Configuring Docker" ***without*** the "Rootless mode" section.
    > ⚠️ Windows users: First verify you can do `nvidia-smi` in WSL. Then please install and configure Nvidia Container Toolkit in WSL.
    >
    > ⚠️ Windows users: For step 2 under "Configuring Docker", instead of doing `sudo systemctl restart docker` do `sudo service docker restart`.

You can proceed with the installation of the devcontainer. Once the installation is complete, you can check the access to GPUs by executing in the devcontainer terminal:
```bash
nvidia-smi
```
You should see GPU stats.

And check PyTorch CUDA runtime is installed in Python console:
```python
import torch
torch.cuda.is_available() # should return True
```
