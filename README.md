# [AE4353] Artificial Intelligence for Aerospace Control and Operations
> Welcome to the repository of the 2024/2025 [AE4353] Artificial Intelligence for Aerospace Control and Operations course! 🚀


## Table of Contents
- [About](#about)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [GitHub Copilot](#github-copilot)
    - [Remote Development](#remote-development)
    - [Windows](#windows)
    - [Linux (Ubuntu)](#linux-ubuntu)
    - [MacOS](#macos)
- [Usage](#usage)
- [License](#license)
- [Contact Information](#contact-information)


## About
This repository contains all the resources for the weekly exercises and competitions associated with the course. Each week, exercises and their solutions will be published here.

To ensure a consistent and easy-to-setup coding environment for everyone, we will be using [VSCode Devcontainers](https://code.visualstudio.com/docs/devcontainers/containers). This will be the official environment throughout the course, including for the final digital exam.

To get started, please refer to this README file for detailed instructions and guidance on setting up your environment correctly. Once your environment is configured, you can begin exploring the code and working on the exercises. If you have any questions or need assistance, feel free to reach out. Happy coding and learning! 🌟


## Getting Started
Welcome to the setup guide! This section will cover the prerequisites, including setting up GitHub Copilot, and provide step-by-step instructions for configuring your development environment. You can choose to set up your environment either remotely using GitHub Codespaces or Google Colab, or locally on Windows, Linux (Ubuntu), or MacOS. Follow the guidelines for your operating system to ensure you have all the necessary tools and configurations. Once your setup is complete, proceed to the [Usage](#usage) section to begin working with the project.

### Prerequisites
- [Visual Studio Code](https://code.visualstudio.com/)

### GitHub Copilot
GitHub Copilot is an AI-powered assistant that helps you write code faster and more efficiently. It provides intelligent code suggestions and completions based on your context, enhancing your coding experience and boosting productivity. 

> ⚠️ We encourage you to use tools like this to aid in learning concepts and practicing coding. However, it’s important not to rely solely on these tools — ensure you put in the effort to understand and practice the material yourself! Please note that such tools will ***NOT*** be permitted during the final exam.

If you do not have it yet, please sign up for the Student Developer Pack on GitHub using this [link](https://education.github.com/pack). Once you have signed up, wait for GitHub to authenticate your request. Once authenticated, you will have access to GitHub Copilot.

If you already have access to GitHub Copilot, it comes pre-installed when you open the devcontainer. Simply log in with your GitHub account and you can start using GitHub Copilot in Visual Studio Code.

### Remote Development
For those who prefer or need a cloud-based development environment, we are pleased to offer a comprehensive guide for setting up and using GitHub Codespaces. This platform allows you to work remotely without the need to install software locally. You can find detailed instructions [here](codespaces_guide.md).

> ⚠️ For GitHub Codespaces, it is _**highly**_ advisable to have a GitHub Pro account, which can be obtained by signing up for the [Student Developer Pack](https://education.github.com/pack). Without this account, you may incur charges for usage. With a GitHub Pro account, you are entitled to up to 90 hours of usage per month with a 2-core setup and 20 GB of storage at no additional cost. Please note that any usage beyond these limits will result in extra fees, so we recommend monitoring your usage carefully.
>
> 💡 You may safely disregard the [Usage](#usage) section in this README, as the guide includes its own detailed usage instructions.

### Windows
To get started with this project on Windows, follow the steps below:

1. Ensure that Visual Studio Code is equipped with the necessary extensions. Install the following extensions in Visual Studio Code: [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), [Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker), and [WSL](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl).

2. Install [Windows Subsystem for Linux (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/install-manual) with the [Ubuntu 20.04](https://www.microsoft.com/store/productId/9MTTCL66CPXJ?ocid=pdpshare) distribution. This will set up an Ubuntu terminal environment, which is necessary for the following steps.

3. Install [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) in WSL2 by following these steps:
    
    a. Open the WSL2 terminal by typing `Ubuntu` in the Windows search bar and selecting the Ubuntu app. From this step onwards, input every subsequent command in this terminal.
    
    b. Set up Docker's `apt` repository:
    ```bash
    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    ```

    > ⚠️ If the above code block execution failed, try to execute the commands one line at a time.

    c. Install the Docker packages:
    ```bash
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin     
    ```

    d. Verify that Docker Engine is installed successfully:
    ```bash
    docker --version
    ```

    e. Check if Docker is running:
    ```bash
    sudo service docker status
    ```

    f. Start the service (if not running):
    ```bash
    sudo service docker start
    ```

    g. Verify that everything is installed and running correctly by running the `hello-world` image:
    ```bash
    sudo docker run hello-world
    ```
    This downloads a test image and runs it in a container. You should see a confirmation message. If you see this, you have successfully installed and started Docker Engine.

4. Follow the [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) for Docker Engine to set the necessary permissions for running Docker on Visual Studio Code. 

5. If this is your first time using WSL2, follow these steps to install Git and set up SSH:

    a. Install Git:
    - Follow these [instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) in the "Installing on Linux" section to install Git on Ubuntu.
    - Verify the installation by running the following command:
      ```bash
      git --version
      ```

    b. Generate a new SSH key by following these [instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent).

    c. Add the SSH key to your GitHub account using these [instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account?platform=linux).

    
Well done! Your setup on Windows is complete, and you are ready to start working on the project. Next, proceed to the [Usage](#usage) section to learn how to run the code.

### Linux (Ubuntu)
To get started with this project on Linux (Ubuntu), follow the steps below:

1. Ensure that Visual Studio Code is equipped with the necessary extensions. Install the following extensions in Visual Studio Code: [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) and [Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker).

2. Install [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) by following these steps:
    
    a. Open the terminal.
    
    b. Set up Docker's `apt` repository:
    ```bash
    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    ```

    > ⚠️ If the above code block execution failed, try to execute the commands one line at a time.

    c. Install the Docker packages:
    ```bash
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin     
    ```

    d. Verify that Docker Engine is installed successfully:
    ```bash
    docker --version
    ```

    e. Check if Docker is running:
    ```bash
    sudo service docker status
    ```

    f. Start the service (if not running):
    ```bash
    sudo service docker start
    ```

    g. Verify that everything is installed and running correctly by running the `hello-world` image:
    ```bash
    sudo docker run hello-world
    ```
    This downloads a test image and runs it in a container. You should see a confirmation message. If you see this, you have successfully installed and started Docker Engine.

3. Follow the [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) for Docker Engine to set the necessary permissions for running Docker on Visual Studio Code. 

4. If Git is not installed on your machine or you have not set up an SSH key, please refer to step 5 in the [Windows](#windows) section for instructions on how to do so.

Well done! Your setup on Linux (Ubuntu) is complete, and you are ready to start working on the project. Next, proceed to the [Usage](#usage) section to learn how to run the code.

### MacOS
To get started with this project on MacOS, follow the steps below:

1. Ensure that Visual Studio Code is equipped with the necessary extensions. Install the following extensions in Visual Studio Code: [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

2. Install [Docker Desktop](https://docs.docker.com/desktop/install/mac-install/).

3. If Git is not installed on your machine or you have not set up an SSH key, please refer to step 5 in the [Windows](#windows) section for instructions on how to do so. For guidance on MacOS, please refer to the instructions specific to MacOS provided in the links.

Well done! Your setup on MacOS is complete, and you are ready to start working on the project. Next, proceed to the [Usage](#usage) section to learn how to run the code.


## Usage
To use this project, follow the steps below:

> 💡 Windows users should do the steps 0-2 in WSL2.

0. If you do not have a ssh key under `$HOME/.git`, follow the [steps](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) to create a key pair and upload to Github.

1. Ensure your Git credentials are available in DevContainers by configuring them according to the guidelines provided [here](https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials#_using-ssh-keys). This will allow seamless access to repositories without repeated authentication prompts while working inside DevContainers.
   
2. Clone the repository and create a private mirror by following the steps outlined [here](private_repo.md). This will help you set up a secure, private copy of the repository for your use.

3. Set up a dataset folder by following these steps:
- `cd` into a directory to host the dataset folder.

    On Linux/MacOS we recommend the `Downloads` folder:
    ```bash
    cd ~/Downloads
    ```
    > ⚠️ For MacOS users, avoid putting the dataset folder in `Desktop` or `Documents` if you have iCloud Drive '[Sync this Mac](https://support.apple.com/en-us/109344)' on.

    On WSL, we recommend the home folder:
    ```bash
    cd ~
    ```
- Create a dataset folder at the directory:
   ```bash
   mkdir ae4353_dataset
   ```

- Navigate to the `ae4353_dataset` directory and copy the output of the pwd command into your clipboard:
    ```bash
    cd ae4353_dataset
    pwd
    ```
    The `pwd` command will print the absolute path to the terminal. Copy this path.

- Go back to the directory of the repository, and start visual studio code:
  ```bash
    cd ~/AE4353-Y24
    code .
    ```
  You will see the files in the repository. 

- Modify the `.devcontainer/devcontainer.json` file:
	- Open the `.devcontainer/devcontainer.json` file.
	- Locate the "mounts" section.
	- Uncomment the line under the "mounts" section.
	- Replace `<your-external-data-directory>` with the absolute path you copied from the `pwd` command.
 
    > 💡 Make sure there is at least 10GB of free space available to host the datasets.
    > 
    > 💡 For WSL2 users, if you create the dataset folder within WSL2, you can access your WSL2 files through File Explorer in Windows and enter `\\wsl$` in the address bar. Navigate to `Ubuntu-<version>\home\<your-username>` (replace `<version>` with your WSL2 distribution version and `<your-username>` with your Linux username).
    >
    > 💡 For WSL2 users, if you want to create the dataset directory outside WSL2 and in your Windows file space, you can access it under the path `/mnt/<disk>/` within WSL2. An example of `<your-external-data-directory>` could be `/mnt/c/Users/Downloads/Dataset_Folder`.

- Close Visual Studio Code.

4. If your machine has a CUDA-enabled GPU, you can follow the steps in [CUDA Guide](cuda_guide.md) in order to use GPUs in the devcontainer.

5. Download the required dataset(s) for the exercise/competition from [SurfDrive](https://surfdrive.surf.nl/files/index.php/s/QzvOHJx2o4KIESI) and move the file(s) to the above dataset folder.
    > 💡 Keep the file in its original format (e.g. `.npz`) without unzipping or extracting any contents unless being instructed explicitly.

6. Open Visual Studio Code, click "File -> Open Folder," and select the repository folder. For WSL2 users, first connect to the WSL2 backend by pressing `F1` and selecting `WSL: Connect to WSL`.
   
7. Set up the development environment:
- Ensure you have the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension installed.

- When you open the folder in Visual Studio Code, a prompt will appear in the bottom-right corner asking if you want to `Reopen in Container`. Click this prompt to build the Docker container, which will open a new Visual Studio Code window inside the container. This container provides a pre-configured environment with all the necessary packages to run the scripts.

- Alternatively, open the Command Palette in Visual Studio Code (`⇧⌘P` on macOS or `Ctrl+Shift+P` on Windows/Linux) and select `Reopen in Container`.

    > 💡 Container build may take up to 10 minutes and require up to 10GB of storage.
    >
    > 💡 During the build process of the container, click the bottom-right corner of the window to view detailed progress logs.
    >
    > 💡 Once the installation is complete and the container is open, if you encounter an "Invalid Python Interpreter" error, open the Command Palette (`⇧⌘P` on MacOS or `Ctrl+Shift+P` on Windows/Linux) and select `Developer: Reload Window`. Ensure that the Jupyter notebook kernel starts with the prefix `AE4353` to confirm everything is functioning correctly.

Good job! You can now start working on the project using the pre-configured environment.


## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for more details.


## Contact Information
For any questions or inquiries, please contact us at:

- Yilun Wu: [y.wu-9@tudelft.nl](mailto:y.wu-9@tudelft.nl)
- Kevin Malkow: [k.malkow@student.tudelft.nl](mailto:k.malkow@student.tudelft.nl)

We will be happy to answer your questions and assist you! 🙂
