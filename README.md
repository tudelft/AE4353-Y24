# [AE4353] Artificial Intelligence for Aerospace Control and Operations
> Welcome to the repository of the 2025/2026 [AE4353] Artificial Intelligence for Aerospace Control and Operations course! 🚀


## Table of Contents
- [About](#about)
- [Getting Started](#getting-started)
	- [Prerequisites](#prerequisites)
		- [Virtual Studio Code](#vscode)
		- [WSL2](#wsl2)
		- [Anaconda](#anaconda)
		- [Git](#git)
	- [GitHub Copilot](#github-copilot)
	- [Remote Development](#remote-development)
- [Setting Up](#setup)
	- [Workspace](#workspace)
		- [Cloning Repository](#clone)
		- [Environment Setup](#environment)
		- [Data](#data)
- [Usage](#usage) 
- [License](#license)
- [Contact Information](#contact-information)

## About
This repository contains all the resources for the weekly exercises and competitions associated with the course. Each week, exercises and their solutions will be published here.

To ensure a consistent and easy-to-setup coding environment for everyone, we will be using a combination of [Virtual Studio Code](https://code.visualstudio.com/) and [Kaggle](https://www.kaggle.com/). These will be the official environments throughout the course.

To get started, please refer to this README file for detailed instructions and guidance on setting up your environment correctly. Once your environment is configured, you can begin exploring the code and working on the exercises. If you have any questions or need assistance, feel free to reach out (see [contact information](#contact-information) to find out how!). Happy coding and learning! 🌟

## Getting Started

### Pre-requisites

These are the required tools for this course. Please make sure to install them before starting the exercises.

#### [1. Virtual Studio Code](https://code.visualstudio.com/)

Visual Studio Code (VS Code) is a lightweight and powerful code editor. We will use it for writing and editing code during the course.  

- [Install](https://code.visualstudio.com/download)
- [Documentation](https://code.visualstudio.com/docs)

#### [2. Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)

Anaconda is a distribution of Python that makes it easy to manage packages ande nvironments. It ensures everyone works with the same dependencies and setup. For this course we will use Miniconda (The primary difference is that Anaconda comes with pre-installed packages, however since we are creating our own environment, we will have a fixed set of packages ready to be installed).

- [Install](https://www.anaconda.com/docs/getting-started/miniconda/install)
- [Documentation](https://www.anaconda.com/docs/main)


#### [3. Windows Subsystem for Linux (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/about) - <span style="color:red;"> (windows only)

WSL2 allows you to run a Linux environment directly on Windows. This provides a more consistent development environment for the course. If you are working on Linux, there is no need for you to install this. 

- [Install](https://learn.microsoft.com/en-us/windows/wsl/install)
- [(online) Documentation](https://learn.microsoft.com/en-us/windows/wsl/)

Make sure to follow the installation instructions provided, and if you run into any issues, walk through the [installation troubleshooting](https://learn.microsoft.com/en-us/windows/wsl/troubleshooting#installation-issues) page, and cross-reference the error code you got with the instructions provided to get past them.

Once you have WSL2 installed, you will want to configure your VS Code to use the WSL bash. This can be done  as follows:

1. Open VS Code and open the terminal by either navigating to the terminal tab on the top left taskbar, or by typing: 
```
ctrl + Shift + ` 
``` 
2. Next, in the top right corner of the terminal (the terminal should open in the bottom row of the VS Code window), there is the option to select a "launch profile" (the symbol is a downward arrow: ⌄). Clicking on this should open a small tab of different options. 

3. ***Select Ubuntu(WSL)***. This will launch your terminal in the WSL environment - allowing you to use it's bash features directly through VS Code.

#### [4. Git](https://git-scm.com/about/branching-and-merging)

Git is a version control system used to track and manage changes in code. It allows you to collaborate and keep your work organized.

***
Make sure to Install this within WSL if you are using Windows (this means installing th Linux version! within WSL). To do this the Navigate to the install link provided below, and select the Linux/Unix option. From there type the following into your WSL terminal:
```
apt-get install git
```
This is the same information provided in the install link from Git, however we still provide the link so that if you want to read a bit into it, you can.
***

- [Install](https://git-scm.com/downloads)
- [Documentation](https://git-scm.com/docs/user-manual)
### GitHub Copilot
GitHub Copilot is an AI-powered assistant that helps you write code faster and more efficiently. It provides intelligent code suggestions and completions based on your context, enhancing your coding experience and boosting productivity. 

> ⚠️ We encourage you to use tools like this to aid in learning concepts and practicing coding. However, it’s important not to rely solely on these tools — ensure you put in the effort to understand and practice the material yourself! Please note that such tools will ***NOT*** be permitted during the final exam.

If you do not have it yet, please sign up for the Student Developer Pack on GitHub using this [link](https://education.github.com/pack). Once you have signed up, wait for GitHub to authenticate your request. Once authenticated, you will have access to GitHub Copilot.

If you already have access to GitHub Copilot, it comes pre-installed when you open the devcontainer. Simply log in with your GitHub account and you can start using GitHub Copilot in Visual Studio Code.

### Remote Development <span style="color:red;"> (UNTOUCHED - NEEDS CHANGES)
For those who prefer or need a cloud-based development environment, we are pleased to offer guides for GitHub Codespaces and Kaggle. These platforms let you work remotely without needing local software installations.

- **GitHub Codespaces**: Easy to set up and user-friendly, offering free (limited) access to a CPU instance. It’s great for general coding but does not provide GPU/TPU access.

- **Kaggle**: Slightly more complex to set up but provides free (limited) access to GPUs and TPUs. This is especially useful for faster training if you don’t have a CUDA-enabled GPU on your laptop.

For detailed instructions, please refer to the guides below:

- [GitHub Codespaces Guide](codespaces_guide.md)
- [Kaggle Guide](kaggle_guide.md)

> ⚠️ For GitHub Codespaces, it is _**highly**_ advisable to have a GitHub Pro account, which can be obtained by signing up for the [Student Developer Pack](https://education.github.com/pack). Without this account, you may incur charges for usage. With a GitHub Pro account, you are entitled to up to 90 hours of usage per month with a 2-core setup and 20 GB of storage at no additional cost. Please note that any usage beyond these limits will result in extra fees, so we recommend monitoring your usage carefully.
>
> 💡 You may safely disregard the [Usage](#usage) section in this README, as the remote development guides include their own detailed usage instructions.

## Setting Up
The next step is setting up the actual workspace you will be working in. This will be fully elaborated here below.

### Workspace
1. ***SSH Key Setup:*** An SSH key is is essentially a secure way of connecting your computer to GitHub without needing to enter your account credentials. This will be useful for connecting your workspace to GitHub. This is how you can do it:

   a. **Generate a New SSH Key:**
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
   b. **Add Your SSH Key to the SSH Agent:**

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

   c. **Copy the SSH Key to Your Clipboard:**

	Next, copy the public SSH key so you can add it to GitHub.
	
	1. Use the following command to copy the SSH key to your clipboard:
	
	    ```bash
	    cat ~/.ssh/id_rsa.pub
	    ```

	2. Select and copy the entire output, which should look something like this:
	
	    ```text
	    ssh-rsa AAAAB3... your_email@example.com
	    ```
     
   d. **Add Your SSH Key to GitHub:**

	1. Log in to your [GitHub account](https://github.com/).
	2. Navigate to **Settings** > **SSH and GPG keys**.
	3. Click **New SSH Key**.
	4. In the "Title" field, add a descriptive label for the new key (e.g., "AE4353_SSH_Key").
	5. Paste your public key (the SSH key you just copied in the previous step) in the "Key" field.
	6. Click **Add SSH Key**.

   e. **Test Your SSH Connection:**

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

2. ***Creating the workspace:***

	a. First we will clone the public `AE4353-Y25` repository and create a private version for your personal use by following the steps outlined [here](private_repo.md).

	> ⚠️ Windows users should use the **Ubuntu** terminal for this (and following steps!)

	b. The next step is to ***Set up the environment***. The reason we use an environment is to allow us to download all the dependencies (packages we will need to complete exercises) in one go. 
	
	First, make sure you are within the repository you just cloned. Once there, run the following command to set up the environment:

	```
	conda env create -f env.yml
	```
	This will install all dependencies and packages needed to complete the exercises that go along with this course. For some further documentation on managing environments, click [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). You can further check if the environment was correctly created by running: 
	```
	conda env list
	```
	If you see the environment `AE4353` there, then you have correctly created the environment!

	Now that you have the environment, you need to be able to activate and de-activate it (depending on what you need at a given time). To activate your environment simply run: 
	```
	conda activate AE4353
	```
	Similarly, you can de-activate the environment by replacing `activate` with `deactivate`.

3. ***Data***

	what we need for this: 
		
		- how to access the data
		- how to store/save the data
		- how to use the data

## Usage
what this section will contain:
steps needed to access the assignments/workspace on every run once the environment is correctly set up.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for more details.

## Contact Information

For any questions or inquiries, please contact us at:

- Quentin Missinne: [Q.Missinne@tudelft.nl](mailto:Q.Missinne@tudelft.nl)
- Dequan Ou: [D.Ou@tudelft.nl](mailto:D.Ou@tudelft.nl)
- Reinier Vos: [R.W.Vos@tudelft.nl](mailto:R.W.Vos@tudelft.nl)

We will be happy to answer your questions and assist you! 🙂