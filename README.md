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
Welcome to the setup guide! This section will cover the prerequisites, including setting up GitHub Copilot, and provide step-by-step instructions for configuring your development environment. You can choose to set up your environment either remotely using GitHub Codespaces, or locally on Windows, Linux (Ubuntu), or MacOS. Follow the guidelines for your operating system to ensure you have all the necessary tools and configurations. Once your setup is complete, proceed to the [Usage](#usage) section to begin working with the project.

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

> 💡 You may safely disregard the [Usage](#usage) section in this README, as the guide includes its own detailed usage instructions.

### Windows
To get started with this project on Windows, follow the steps below:

1. Ensure that Visual Studio Code is equipped with the necessary extensions. Install the following extensions in Visual Studio Code: [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), [Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker), and [WSL](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl).

2. Install [Windows Subsystem for Linux (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/install-manual) along with the [Ubuntu 20.04](https://www.microsoft.com/store/productId/9MTTCL66CPXJ?ocid=pdpshare) distribution. Follow the guide fully up to and including "Step 6: Install your Linux distribution of choice." After installation, open the **Ubuntu** terminal by typing `Ubuntu` into the Windows search bar and selecting the Ubuntu distribution you installed (e.g., Ubuntu 20.04).

	This sets up an Ubuntu terminal environment that will be necessary for the following steps. _**From this point forward, ensure you always open the Ubuntu terminal as described above and use it for all subsequent commands!**_

	> ⚠️ **Important:** Always use the **Ubuntu** terminal, **NOT** the WSL2 terminal, as using the WSL2 terminal can cause issues when setting up the development environment. Additionally, if you see `root@hostname:~#` in your **Ubuntu** terminal, exit root mode by running `exit`, as staying in root mode can also lead to setup issues.

3. Install [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) in the **Ubuntu** terminal by following these steps:
        
    a. Set up Docker's `apt` repository:
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

    b. Install the Docker packages:
    ```bash
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin     
    ```

    c. Verify that Docker Engine is installed successfully:
    ```bash
    docker --version
    ```

    d. Check if Docker is running:
    ```bash
    sudo service docker status
    ```

    e. Start the service (if not running):
    ```bash
    sudo service docker start
    ```

    f. Verify that everything is installed and running correctly by running the `hello-world` image:
    ```bash
    sudo docker run hello-world
    ```
    This downloads a test image and runs it in a container. You should see a confirmation message. If you see this, you have successfully installed and started Docker Engine.

4. Follow the [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) for Docker Engine to set the necessary permissions for running Docker on Visual Studio Code. 

5. If you're setting up WSL2 with Ubuntu for the first time, follow these steps to install Git and configure SSH using the **Ubuntu** terminal:

    a. **Install Git:**
    - Follow these [instructions](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) in the "Installing on Linux" section to install Git on Ubuntu.
    - Verify the installation by running the following command:
      ```bash
      git --version
      ```
   b. **Generate a New SSH Key:**
   	1. Open your Ubuntu terminal.
	2. Run the following command to generate a new SSH key pair:
	    ```bash
	    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
	    ```
	    - Replace `"your_email@example.com"` with the email address associated with your GitHub account.
	    - You will be prompted to specify a file location. Press `Enter` to accept the default location (`/home/your_username/.ssh/id_rsa`).
	    - Optionally, you can set a passphrase for added security. If you don’t want a passphrase, just press `Enter` to skip.

	    Example of output:
	
	    ```bash
	    Generating public/private rsa key pair.
	    Enter file in which to save the key (/home/your_username/.ssh/id_rsa): [Press Enter]
	    Enter passphrase (empty for no passphrase): [Type a passphrase or press Enter]
	    Enter same passphrase again: [Type passphrase again or press Enter]
    	```
   c. **Add Your SSH Key to the SSH Agent:**

   	After creating the SSH key, you need to add it to the SSH agent so it can be used for authentication.
	1. Start the SSH agent:

	    ```bash
	    eval "$(ssh-agent -s)"
	    ```

    	This will start the agent and print out the process ID (PID) if it's running correctly.

	2. Add your private key to the agent:
	
	    ```bash
	    ssh-add ~/.ssh/id_rsa
	    ```

    	Make sure the path points to the correct location of your private key (default is `~/.ssh/id_rsa`).

   d. **Copy the SSH Key to Your Clipboard:**

	Next, copy the public SSH key so you can add it to GitHub.
	
	1. Use the following command to copy the SSH key to your clipboard:
	
	    ```bash
	    cat ~/.ssh/id_rsa.pub
	    ```

	2. Select and copy the entire output, which should look something like this:
	
	    ```text
	    ssh-rsa AAAAB3... your_email@example.com
	    ```
     
   e. **Add Your SSH Key to GitHub:**

	1. Log in to your [GitHub account](https://github.com/).
	2. Navigate to **Settings** > **SSH and GPG keys**.
	3. Click **New SSH Key**.
	4. In the "Title" field, add a descriptive label for the new key (e.g., "AE4353_SSH_Key").
	5. Paste your public key (the SSH key you just copied in the previous step) in the "Key" field.
	6. Click **Add SSH Key**.

   f. **Test Your SSH Connection:**

	Now that you've added your key to GitHub, test your connection:

	1. Run the following command:
	
	    ```bash
	    ssh -T git@github.com
	    ```
	
	2. You should see a message similar to this:
	
	    ```bash
	    Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.
	    ```

	This confirms that your SSH key was added successfully, and you're now able to clone, pull, and push repositories via SSH.
    
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

1. If you haven't already created and added an SSH key, please follow the instructions starting from Step 5b in the [Windows](#windows) section for detailed guidance on how to do so.
    > ⚠️ Windows users should use the **Ubuntu** terminal for these steps.

2. Ensure your Git credentials are set up correctly for use in DevContainers by following the guidelines provided [here](https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials). This process includes:

   - **Configuring SSH Keys:** Make sure your SSH keys are correctly set up on your local machine.
   - **Sharing SSH Keys with DevContainers:** The guide explains how to forward your local SSH agent to the DevContainer, allowing the DevContainer to use your local SSH keys.

   By following these steps, you will be able to push changes to your repositories without encountering a `git@github.com: Permission denied (publickey)` error when using the `git push origin` command in the Visual Studio Code terminal inside the container. Properly configuring your Git credentials ensures smooth and uninterrupted access to your repository without needing to repeatedly authenticate/create and add an SSH Key.

    > ⚠️ Windows users should use the **Ubuntu** terminal for this and follow the instructions for Linux.

    > 💡 **Note:** If you followed the guide for setting up your SSH key in the [Windows](#windows) section, your `<your ssh key>` will be `id_rsa`.


3. Clone the public `AE4353-Y24` repository and create a private version for your use by following the steps outlined [here](private_repo.md).
	> ⚠️ Windows users should use the **Ubuntu** terminal for this.
   
4. Set up and mount a dataset folder by following these steps:

	a. `cd` into a directory to host the dataset folder.

	- On Linux/MacOS we recommend the `Downloads` folder:
	
		```bash
		cd ~/Downloads
		```
	
		> ⚠️ **For MacOS users:** If you have iCloud Drive's [Sync this Mac](https://support.apple.com/en-us/109344) feature enabled, avoid placing the dataset folder in the `Desktop` or `Documents` directories.

	- On WSL2, we recommend the home folder:

		```bash
		cd ~
		```
  
	b. Create a dataset folder at the directory:

	```bash
	mkdir ae4353_dataset
	```
	
   	c. Navigate to the `ae4353_dataset` directory and then use the `pwd` command to display the absolute path of the directory:

	```bash
	cd ae4353_dataset
	pwd
	```

   	Copy this path.

   	d. Open Visual Studio Code and select "File -> Open Folder" to choose the repository folder.

	  Alternatively, on Windows/Linux, you can navigate to the repository directory in your **Ubuntu** terminal and start Visual Studio Code with the following commands:
		
	```bash
	cd ~/AE4353-Y24
	code .
	```
 
   	e. Find the `.devcontainer/devcontainer.json` file and modify it as follows:
	
 	1. Open the `.devcontainer/devcontainer.json` file.
	2. Locate the "mounts" section.
	3. Uncomment the line under the "mounts" section.
	4. Replace `<your-external-data-directory>` with the absolute path you copied from the `pwd` command.
 
		> 💡 For WSL2 users: If you create the dataset folder within WSL2, you can access these files from Windows File Explorer by entering `\\wsl$` in the address bar. Navigate to `Ubuntu-<version>\home\<your-username>`, replacing `<version>` with your Ubuntu distribution version and `<your-username>` with your Linux username.
		
		> 💡 For WSL2 users: If you create the dataset folder in your Windows file space (outside WSL2), you can access it from WSL2 using the path `/mnt/<disk>/`. For example, if your dataset folder is located in `C:\Users\Downloads\Dataset_Folder`, you can access it in WSL2 at `/mnt/c/Users/Downloads/Dataset_Folder`.

 	f. Close Visual Studio Code.

5. Download the required dataset(s) for the exercise/competition from [SurfDrive](https://surfdrive.surf.nl/files/index.php/s/QzvOHJx2o4KIESI) and move the file(s) to the above dataset folder.
	> ⚠️ Unzip the `AE4353-Datasets-2024.zip` file and move its contents into the `ae4353_dataset` folder that you created earlier.

6. **(Optional)** If your machine has a CUDA-enabled GPU and you want to use it within the devcontainer, follow the instructions in the [CUDA Guide](cuda_guide.md) to set up GPU support. Enabling GPU support can speed up training times from hours to minutes, making your deep learning work a lot quicker!

7. **Open Visual Studio Code and Set Up the Development Environment:**

	a. Open Visual Studio Code and select "File -> Open Folder" to choose the repository folder.
		
	Alternatively, on Windows/Linux, you can navigate to the repository directory in your **Ubuntu** terminal and start Visual Studio Code with the following commands:
					
	```bash
	cd ~/AE4353-Y24
	code .
	```
 
	> ⚠️ **For WSL2 Users:** After opening Visual Studio Code, make sure you are connected to the WSL2 backend:
	>
	> - Press `Ctrl+Shift+P` to open the Command Palette.
	> - Type `WSL: Connect to WSL` and select it from the dropdown menu to establish the connection.


	b. When you open the folder in Visual Studio Code, a prompt will appear in the bottom-right corner asking if you want to `Reopen in Container`. Click this prompt to build the dev container, which will open a new Visual Studio Code window inside the container.

 	> 💡 Alternatively, open the Command Palette in Visual Studio Code (`⇧⌘P` on macOS or `Ctrl+Shift+P` on Windows/Linux) and select `Reopen in Container`.
		
	> 💡 **Good to Know:** Building the container may take up to 10 minutes and require up to 10GB of storage. You can view the progress by clicking on the bottom-right corner of the window to see detailed logs.
   
	> 💡 **Troubleshooting Tip:** After the container is installed and open, if you see an "Invalid Python Interpreter" error:
	>
	> - Open the Command Palette (`⇧⌘P` on macOS or `Ctrl+Shift+P` on Windows/Linux).
	> - Select `Developer: Reload Window`.

	c. Once your development environment has finished building, you can confirm its functionality by opening a Jupyter notebook (`.ipynb`) of your choice (e.g., `ex_1.ipynb`). Click on `Select Kernel` in the top right corner of the notebook, then select `Python Environments`. Check that the kernel listed starts with the prefix `AE4353` to ensure your environment is configured properly. If the kernel is correctly prefixed, your development environment has been set up successfully!


8. **Save, Commit, and Push Your Work:**

	Always save your work locally (using `Ctrl+S`). When you finish your session, commit and push your changes to your private repository to ensure your work is backed up on GitHub. To do this, use the **Visual Studio Code (devcontainer) terminal** and follow these steps:

	> ⚠️ We strongly recommend committing and pushing your changes to GitHub regularly. This way, if something goes wrong locally, you won't lose your progress!

   1. Check the status of your changes:
      ```bash
      git status
      ```

   2. Add all changes to the staging area:
      ```bash
      git add .
      ```

   3. Commit your changes with a message:
      ```bash
      git commit -m "Your commit message here"
      ```

   4. Push your changes to the repository:
      ```bash
      git push origin
      ```

	> 💡 **Troubleshooting Tip:** If you encounter a "permission denied" error while running `git push origin`, you may need to revisit Step 2 and ensure that your Git credentials are correctly forwarded to the DevContainer.
	
	> 💡 Visual Studio Code may prompt you to configure your Git username and email. If prompted, you can follow the instructions to enter the following commands in the terminal:
	>
	> ```bash
	> git config --global user.name "Your Name"
	> git config --global user.email "you@example.com"
	> ```
 
	> 💡 **Good to Know:** For further information about Git commands, please check this [Git cheatsheet](https://education.github.com/git-cheat-sheet-education.pdf).

Good job! 🎉 You can now start working on the project using the pre-configured environment.


## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for more details.


## Contact Information
For any questions or inquiries, please contact us at:

- Yilun Wu: [y.wu-9@tudelft.nl](mailto:y.wu-9@tudelft.nl)
- Kevin Malkow: [k.malkow@student.tudelft.nl](mailto:k.malkow@student.tudelft.nl)

We will be happy to answer your questions and assist you! 🙂
