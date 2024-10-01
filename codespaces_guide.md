# Getting Started with GitHub Codespaces
This guide will help you set up a GitHub Codespace, a cloud-based development environment that you can access directly from your browser. This allows you to code from anywhere, without the need for a local setup.

Please note, that GitHub Codespaces provides a limited number of usage hours and storage per month for free with a GitHub Pro account, which also includes those with the GitHub Student Developer Pack. Given that deep learning can be quite computationally intensive, this option is particularly well-suited for those who may have a laptop that is not ideally equipped for such demanding tasks.

We hope that you will find this guide helpful! Please follow the steps below to get started.

## Prerequisites
Before you start, make sure you have:

1. **A GitHub Account**: If you do not have one, sign up for free [here](https://github.com/).

2. **Access to the Student Developer Pack**: Apply for the [GitHub Student Developer Pack](https://education.github.com/pack) to receive free Codespaces hours and other benefits.

   > ⚠️ Make sure you have the Student Developer Pack **_before_** continuing to avoid additional costs!

3. **Your Personal Private `AE4353-Y24` Repository**: If you haven’t already created a private version of the `tudelft/AE4353-Y24` repository, follow these steps. Note that if you’ve already created a private repository, you do not need to create a new one specifically for Codespaces:

   1. Go to GitHub and create a new repository.
   2. Click on the `Import repository` option at the top of the page.
   3. Paste the URL of the public repository: `https://github.com/tudelft/AE4353-Y24`.
   4. Select `Private` at the bottom and name the repository `AE4353-Y24`.
   
   After completing these steps, you will have a private copy of the `tudelft/AE4353-Y24` repository under your GitHub username as `<your_username>/AE4353-Y24`.

## Getting Started
Follow these steps to set up your GitHub Codespace and start coding:

1. Navigate to the [GitHub Codespaces page](https://github.com/features/codespaces).

2. Click on `Get started for free` and log in to your GitHub account.

3. Click the green button labeled `New codespace` in the top right corner to start setting up your Codespace.

4. Configure the Codespace as follows:
   - _Repository_: `<your_username>/<personal-repo-name>`
   - _Branch_: `main`
   - _Dev Container Configuration_: `ae4354`
   - _Region_: Select `Europe West` if you are in Europe.
   - _Machine Type_: Choose `2-core`.

   > ⚠️ **For GitHub Pro accounts, including those with the Student Developer Pack, 180 core hours of usage and 20GB of storage are provided for free each month.** For a 2-core setup, this typically allows around 90 hours of usage per month or approximately 3 hours per day. **Please be mindful that any usage beyond this limit _will_ be billed to your account, so use with caution.** However, this amount of time should generally be sufficient to complete the tasks at hand. For more information, please refer to [GitHub Codespaces billing](https://docs.github.com/en/billing/managing-billing-for-github-codespaces/about-billing-for-github-codespaces).

5. Click `Create codespace`. After doing so, a Visual Studio Code interface will appear. The DevContainer environment will be automatically set up, and all necessary Visual Studio Code extensions will be installed. Please allow some time for this process to complete.

   > 💡 During the environment setup, you can click on `Building codespace` in the bottom-right corner of the window to view detailed progress logs.
   >
   > 💡 **If an error window appears indicating that your environment is not configured correctly and is running in recovery mode, please delete the Codespace instance you created and try creating a new 2-core instance using the steps above.** If the issue persists, continue this process until you successfully configure a 2-core setup. Usually, this resolves after one or two attempts.

6. Add the public repository as `upstream` to track changes from it while disabling push access to prevent accidental updates to it:

   ```bash
   git remote add upstream https://github.com/tudelft/AE4353-Y24.git
   git remote set-url --push upstream DISABLE
   ```

Well done! Your GitHub Codespaces setup is complete, and you are ready to start working on the project. Next, proceed to the [Usage](#usage) section to learn how to run the code.

## Usage
To use your configured GitHub Codespace, please follow the steps below:

1. Navigate to the Jupyter notebook you wish to use. When you open it, the kernel may not be pre-selected. In the top right corner of the notebook, click on `Select Kernel`, then go to `Python Environments` and select the recommended kernel, which should be `AE4353 ...`.

2. Download the required dataset(s) for the exercise or competition from [SurfDrive](https://surfdrive.surf.nl/files/index.php/s/QzvOHJx2o4KIESI) and move the file(s) to the `data` folder. You can do this by either dragging and dropping the files or by right-clicking on the `data` folder and selecting `Upload`.

    > 💡 **Keep the file in its original format (e.g., `.npz`) without unzipping or extracting any contents unless explicitly instructed.**
    >
    > 💡 Uploading the file may take a few minutes. You can monitor the progress in the bar at the bottom.

3. After you finish working in your Codespace, make sure to commit and push your changes. Then, stop your Codespace to avoid unnecessary usage and costs. For detailed instructions on managing your Codespace, refer to the [Managing Your Codespace](#managing-your-codespace) section.

   > ⚠️ **Important:** Always commit and push your changes after each session to avoid losing your work. Use the following commands in the terminal:
   > 
   > ```bash
   > git status
   > git add .
   > git commit -m "<your-message-here>"
   > git push
   > ```
   > 
   > Keeping your work saved and up-to-date ensures that nothing is lost and everyone stays happy!

4. If you want to update your private repository with changes from the public repository, use the following commands in the terminal:

   1. If there are new commits in the public repository, you can fetch them:

      ```bash
       git fetch upstream
      ```
      
   3. Rebase your work on top of the latest changes from the public repository:

      ```bash
      git rebase upstream/main
      ```

   4. If there are any merge conflicts, resolve them as needed. This ensures your private repository remains up-to-date while maintaining your own changes.

Good job! You are now ready to start working on the project using the pre-configured environment in your GitHub Codespace. Happy coding! 🚀

## Managing Your Codespace
Once your Codespace is running, it may be necessary to manage it to avoid additional costs. Here are some key actions you can take:

- **Stopping**: If you want to pause your work and avoid further billing, you can stop your Codespace. Click the three dots next to your Codespace in the Codespaces list and select `Stop codespace`. This action saves your current state and pauses any associated costs, allowing you to resume where you left off later.

- **Deleting**: If you no longer need a Codespace, you can delete it to free up resources and stop any further charges. Click the three dots next to your Codespace and select `Delete`. Be sure to save all your work before performing this action, as deleting a Codespace is permanent and cannot be undone.

## Additional Resources
For more detailed information about GitHub Codespaces, refer to the following resource:

- [GitHub Codespaces Documentation](https://docs.github.com/en/codespaces).
