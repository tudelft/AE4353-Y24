# [AE4353] Artificial Intelligence for Aerospace Control and Operations
> Welcome to the repository of the 2025/2026 [AE4353] Artificial Intelligence for Aerospace Control and Operations course! 🚀


## Table of Contents
- [About](#about)
- [Getting Started](#getting-started)
	- [Prerequisites](#prerequisites)
		- [Virtual Studio Code](#vscode)
	- [GitHub Copilot](#github-copilot)
- [Setting Up](#setup)
	- [Workspace](#workspace)
		- [Cloning Repository](#clone)
		- [Environment Setup](#environment)
		- [Data](#data)
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

#### [2. Git](https://git-scm.com/about/branching-and-merging)

Git is a version control system used to track and manage changes in code. It allows you to collaborate and keep your work organized.

- [Install](https://git-scm.com/downloads)
- [Documentation](https://git-scm.com/docs/user-manual)

#### [3. Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)

Anaconda is a distribution of Python that makes it easy to manage packages and environments. It ensures everyone works with the same dependencies and setup. For this course we will use Miniconda (The primary difference is that Anaconda comes with pre-installed packages, however since we are creating our own environment, we will have a fixed set of packages ready to be installed).

- [Install](https://www.anaconda.com/docs/getting-started/miniconda/install)
- [Documentation](https://www.anaconda.com/docs/main)

### GitHub Copilot
GitHub Copilot is an AI-powered assistant that helps you write code faster and more efficiently. It provides intelligent code suggestions and completions based on your context, enhancing your coding experience and boosting productivity. 

> ⚠️ We encourage you to use tools like this to aid in learning concepts and practicing coding. However, it’s important not to rely solely on these tools — ensure you put in the effort to understand and practice the material yourself! Please note that such tools will ***NOT*** be permitted during the final exam.

If you do not have it yet, please sign up for the Student Developer Pack on GitHub using this [link](https://education.github.com/pack). Once you have signed up, wait for GitHub to authenticate your request. Once authenticated, you will have access to GitHub Copilot.

If you already have access to GitHub Copilot, it comes pre-installed when you open the devcontainer. Simply log in with your GitHub account and you can start using GitHub Copilot in Visual Studio Code.

## Setting Up
The next step is setting up the actual workspace you will be working in. This will be fully elaborated here below.

### Workspace
1. ***Creating the workspace:***

	> ⚠️ Windows users should do this in the Git Bash. Linux users can do this directly through the bash terminal.

	a. First we will clone the public `AE4353-Y25` repository. In order to do this open the Git/Bash terminal and type the following.

	```
	git clone https://github.com/tudelft/AE4353-Y25.git
	```

	This will create your workspace for the course where you will find all the exercises.

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

	The data for this course can be found at this [link](https://surfdrive.surf.nl/files/index.php/s/uStySKYBKHBXcjP). In order to access it you will need to enter the password: `Ae4353`. Once you've downloaded this data, please extract/unzip the folder and store the data within the data folder found in `/your/workspace/path/AE4353-Y25/data/`.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for more details.

## Contact Information

For any questions or inquiries, please contact us at:

- Quentin Missinne: [Q.Missinne@tudelft.nl](mailto:Q.Missinne@tudelft.nl)
- Dequan Ou: [D.Ou@tudelft.nl](mailto:D.Ou@tudelft.nl)
- Reinier Vos: [R.W.Vos@tudelft.nl](mailto:R.W.Vos@tudelft.nl)

We will be happy to answer your questions and assist you! 🙂