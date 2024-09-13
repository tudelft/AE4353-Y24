1. Uncomment the `"runArgs"` section in `.devcontainer/devcontainer.json`.

2. Uncomment the `nvidia` channel and `pytorch-cuda` package lines in `env.yml`.

3. Install the Nvidia Container Toolkit and configure Docker by following the instructions in this [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Focus on the sections **"Installing with Apt"** and **"Configuring Docker"**, and **do not** follow the **"Rootless mode"** section.

    > ⚠️ **Windows Users:** First, verify that you can run `nvidia-smi` in the **Ubuntu** terminal. Next, install and configure the Nvidia Container Toolkit in the **Ubuntu** terminal.
    >
    > ⚠️ **Important:** When following the **"Configuring Docker"** step, use **`sudo service docker restart`** instead of `sudo systemctl restart docker`.
    
4. After installing the devcontainer, verify GPU access by running the following command in the Visual Studio Code (devcontainer) terminal:

    ```bash
    nvidia-smi
    ```

    You should see GPU statistics displayed.

5. Verify that PyTorch CUDA runtime is installed and configured correctly by running the following code in a Python console:

    ```python
    import torch
    print(torch.cuda.is_available())  # This should print 'True' if CUDA support is enabled
    ```
    
    If the output is `True`, it means that PyTorch is correctly set up to use CUDA. If it prints `False`, CUDA support may not be properly configured.

