# Getting Started with Kaggle
This guide will help you get started with Kaggle, a cloud-based platform that allows you to run Python code directly in your browser, removing the need for a local setup. Kaggle offers powerful computing resources, including GPUs and TPUs, making it an excellent tool for data science, machine learning, and deep learning projects. It is especially useful for those who may not have access to high-performance hardware.

Please note, that Kaggle provides limited free usage of these resources. While you will not be billed for exceeding usage limits, access to GPUs or TPUs may be temporarily restricted if you reach your quota.

We hope you find this guide helpful! Follow the steps below to get started with Kaggle.

## Prerequisites
Before you start, make sure you have:

1. **Your Personal Private `AE4354-Y24` Repository**: Clone the repository and create a private version by following the steps outlined [here](private_repo.md), if you have not done so already.

2. **A GitHub Personal Access Token**: You’ll need a personal access token to use GitHub with Kaggle. Follow the instructions [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic) to create one.

## Getting Started
Follow these steps to set up Kaggle and start coding:

### Step 1: Sign Up with Kaggle
1. Go to the [Kaggle website](https://www.kaggle.com/).
2. Click on "Sign Up" and follow the instructions to create your account ***`with your TUDelft email`***.

> ⚠️ Your Kaggle account must be associated with your TU Delft email address in order to join the course competition, which is exclusive to TU Delft students.

### Step 2: Import a Notebook into Kaggle
1. **Download Your Notebook**:
   - Find and download the Kaggle notebook from your GitHub repository to your computer. For example, it should be located under `ex_2/ex_2_kaggle.ipynb` for Exercise 2.

2. **Create a New Notebook in Kaggle**:
   - Log in to Kaggle.
   - Click on `+ Create` in the top-left corner of the homepage and select `New Notebook`.
   - Name your notebook. For example, `ex_2_kaggle` for Exercise 2.

3. **Import Your Notebook**:
   - In your new notebook, click on `File` in the menu and select `Import Notebook`.
   - Upload the notebook file you downloaded.

4. **Verify Your Account**:
   - Ensure your account is verified with a phone number. If you’re not prompted to verify, you can find the verification link in the `Session options` on the right side of the page.
   - Verification is required to access Kaggle’s internet and GPU/TPU features.

5. **Enable Internet Access**:
   - Once verified, turn on the **Internet** option in the `Session options` on the right side.

### Step 3: Load the Dataset in Kaggle
To work with the dataset in your Kaggle notebook, follow these steps:

1. **Check Dataset Availability**:
   - After opening your notebook, look under `Input` → `DATASETS` on the right side of the notebook interface.
   - You should see the corresponding dataset there (e.g. `polarization` for Exercise 2). If it’s there, you’re ready to use it.

2. **Add the Dataset if Not Visible**:
   - If you don’t see the dataset:
     1. Click on `Add Input`.
     2. In the search bar, paste the corresponding URL:
       - For Exercise 2: [https://www.kaggle.com/datasets/dnaylw/polarization](https://www.kaggle.com/datasets/dnaylw/polarization).
       - For Exercise 4: [https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/).
     3. Select the dataset from the search results and add it to your notebook.

### Step 4: Switching Between CPU and GPU
1. **Use CPU for Initial Setup**:
   - Perform setup and exploration tasks using a `CPU`. In the `Settings` bar, select `Accelerator` and choose `None`. This will save your GPU quota for when you need it.

2. **Switch to GPU P100 for Training**:
   - When you’re ready to run your training, switch to `GPU P100` to speed up the process.
   - To switch, go to the `Settings` bar, select `Accelerator`, and choose `GPU P100`.

> 💡 After switching the accelerator, the runtime of the notebook will be restarted. You would need to re-execute the entire notebook from the beginning.

> 💡 Kaggle offers unlimited hours of CPU usage, 30 hours per week GPU usage, and 20 GB of auto-saved disk space under `/kaggle/working`. Use CPU for setup and GPU for training to make the most of your usage quota. For more details on Kaggle’s GPU usage limits, see the [Kaggle Documentation](https://www.kaggle.com/docs/notebooks#accelerators). Ensure you monitor your resource usage to avoid hitting limits and budget the resource usage wisely among working on exercises and the competition.

## Common Errors
- **Kernel Crashes**: Restart the kernel by going to the `Run` dropdown and clicking `Factory reset`.
- **Dataset Not Found**: Use the command `!ls /kaggle/input` to list available datasets and adjust your paths accordingly. See the notebooks for further details.

Well done! Your Kaggle setup is complete, and you are ready to start working on the project. Follow the instructions in your notebook to proceed.
