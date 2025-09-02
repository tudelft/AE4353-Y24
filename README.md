# [AE4353] Artificial Intelligence for Aerospace Control and Operations
> Welcome to the repository of the 2025/2026 [AE4353] Artificial Intelligence for Aerospace Control and Operations course! 🚀

## Table of Contents
- [About](#about)
- [GitHub Copilot](#github-copilot)
- [Pre-requisites](#pre-requisites)
    - [Virtual Studio Code](#virtual-studio-code)
- [Windows](#windows)
    - [Setting Up](#setting-up)
        - [Git](#git)
        - [Miniconda](#miniconda)
- [Linux](#linux)
    - [Setting Up](#setting-up-1)
        - [Git](#git-1)
        - [Miniconda](#miniconda-1)
    - [Data](#data)
- [License](#license)
- [Contact Information](#contact-information)

## About
This repository contains all the resources for the weekly exercises and competitions associated with the course. Each week, exercises and their solutions will be published here.

To ensure a consistent and easy-to-setup coding environment for everyone, we will be using a combination of [Virtual Studio Code](https://code.visualstudio.com/) and [Kaggle](https://www.kaggle.com/). These will be the official environments throughout the course.

To get started, please refer to this README file for detailed instructions and guidance on setting up your environment correctly. Once your environment is configured, you can begin exploring the code and working on the exercises. If you have any questions or need assistance, feel free to reach out (see [contact information](#contact-information) to find out how!). Happy coding and learning! 🌟

### GitHub Copilot
GitHub Copilot is an AI-powered assistant that helps you write code faster and more efficiently. It provides intelligent code suggestions and completions based on your context, enhancing your coding experience and boosting productivity. 

> ⚠️ We encourage you to use tools like this to aid in learning concepts and practicing coding. However, it’s important not to rely solely on these tools — ensure you put in the effort to understand and practice the material yourself! Please note that such tools will ***NOT*** be permitted during the final exam.

If you do not have it yet, please sign up for the Student Developer Pack on GitHub using this [link](https://education.github.com/pack). Once you have signed up, wait for GitHub to authenticate your request. Once authenticated, you will have access to GitHub Copilot.

If you already have access to GitHub Copilot, it comes pre-installed when you open the devcontainer. Simply log in with your GitHub account and you can start using GitHub Copilot in Visual Studio Code.

## Getting Started

### Pre-requisites
These are the required tools for this course. Please make sure to install them before starting the exercises.

#### [1. Virtual Studio Code](https://code.visualstudio.com/)
Visual Studio Code (VS Code) is a lightweight and powerful code editor. We will use it for writing and editing code during the course.  

- [Install](https://code.visualstudio.com/download)
- [Documentation](https://code.visualstudio.com/docs)

## Setting Up
The next step is setting up the actual workspace you will be working in. This will be fully elaborated here below.

> ⚠️ Windows and Linux have different setups, so please make sure you follow the setup which works for your distribution!

### Windows:
1. ***Git***

    a. Install the Git Bash for Windows using this [Installation link](https://git-scm.com/downloads), and selecting the Windows distribution. Once the installation file is downloaded, run the installation (by selecting the downloaded file), and accept the recommended installation location and permissions.

    b. Once you've completed the tasks through the installation wizard, open the Git Bash by either searching for it in your applications (can be done by pressing the `Windows key` and typing Git Bash).

    c. Inside the Git terminal, you can now perform Git operations. This will allow you to download the course repository and update it as we progress through the course. 
    
    1. Clone the repository (create a local copy) by running the following command:
        ```
        git clone https://github.com/tudelft/AE4353-Y25.git
        ```
            
    2. Now you have a local version of the repository which can be updated directly along with any changes we upload to the repository with one simple command. First navigate to the repository folder (from the Git terminal simply type `cd AE4353-Y25/`), then type:
        ```
        git pull
        ```
    	
		This will automatically update your local version with any changes or additions we have made.

2. ***Miniconda***

    Now that you have a version of the repository, we need to create the working environment. This is essentially a way of downloading all the packages required for the exercises in a controlled way.

    a. Use the following [download link](https://www.anaconda.com/download). Here you will need to press the big green `Get Started` button, which will ask you to create an account. Make sure to do this with a ***valid email*** as you will need to verify the account. Once this is done you will have a choice between downloading the `distribution` and downloading `miniconda`. Please select `miniconda`. 

    b. With Miniconda downloaded, you can now navigate to the `Anaconda Prompt` (in the same way you did for the Git Bash), and open it.

    c. In Anaconda Prompt, navigate to your repository folder (using `cd AE4353-Y25/`) then enter the following command to create the environment:
    ```
    conda env create -f env.yml
    ```

    This will take a few minutes. Once it’s done, run `conda env list` in the terminal to check if the environment installed correctly. If `AE4353` shows there, then you are ready to go!
    
> Once you have completed these steps, you can immediately go to the [Usage](#usage) section to see how to use the setup you just created!

### Linux:
Installations on Linux are all done directly through bash. Therefore the first thing to do for the installation is opening the command window (either by navigating to it through the applications page or by typing `ctrl + alt + t`).

1. ***Git***

    The following instructions are all found on this [installation link](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). Please refer to it if you encounter any problems. 

    a. Through the bash terminal install Git:
    ```
    apt-get install git-all
    ```
        
    b. Once you've installed git and its dependencies, you can immediately clone the repository:
    ```
    git clone https://github.com/tudelft/AE4353-Y25.git
    ```
        
    c. For any future updates we may make to the repository, you can directly update your local copy by navigating to the repository (in the bash terminal type `cd AE4353-Y25/`) and running:
    ```
    git pull
    ```

2. ***Miniconda***

    a. In order to install Miniconda, follow the instructions shown on this [installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2). This should lead you to a clean install of Miniconda.

    b. Once you've installed Miniconda, you will be able to use the conda commands in your bash terminal. This will allow you to create the environment for this course. To do this enter the following command in your terminal:
    ```
    conda env create -f env.yml
    ```
    
	You can check this installation by using the `conda env list` command. If the environment `AE4353` shows up, then you have completed the installation process!

## Usage
Now that you've completed the installations you can start working on the exercises! In this section we will cover how you can use Virtual Studio Code (VS Code) as your interface for this course. 

> 💡 This section is explained in the `Practical session 1` slides with images of each step. Please check them for further clarification on the steps below.

1. Open VS Code. Once in, you will see several options in the center of the window. Select the `open folder...` option. From there select the repository you just created.

2. Once in the environment, you will see on the left-hand side the ***working directory***. On the top right you will see a `kernel` option. Click on it, then select the `jupyter kernel` option (as this is a VS Code pre-requirement to load in a Python environment). Once you've done this, you should see the `AE4353` environment option. Select it!

3. Now that you've set up the environment in your VS Code, you are ready to get coding! Open the first exercise notebook `Ex_1.ipynb` and get coding!

## Data
The data for this course can be found at this [link](https://surfdrive.surf.nl/files/index.php/s/uStySKYBKHBXcjP). In order to access it you will need to enter the password: `Ae4353`. Once you've downloaded this data, please extract/unzip the folder and store the data within the data folder found in `/your/workspace/path/AE4353-Y25/data/`.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for more details.

## Contact Information
For any questions or inquiries, please contact us at:

- Quentin Missinne: [Q.Missinne@tudelft.nl](mailto:Q.Missinne@tudelft.nl)
- Dequan Ou: [D.Ou@tudelft.nl](mailto:D.Ou@tudelft.nl)
- Reinier Vos: [R.W.Vos@tudelft.nl](mailto:R.W.Vos@tudelft.nl)

We will be happy to answer your questions and assist you! 🙂
