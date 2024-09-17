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

     - Go to GitHub and create a new repository.
     - Click on the `Import repository` option at the top of the page.
     - Paste the URL of the public repository: `https://github.com/tudelft/AE4353-Y24`.
     - Select `Private` at the bottom and name the repository `AE4353-Y24`.
   
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

6. **Create and Add an SSH Key:** To create and add an SSH key in the GitHub Codespaces (Visual Studio Code) terminal, follow the instructions starting from Step 5bii in the Windows section of the [README](README.md) for detailed guidance.

7. **Configure Your Repository’s Settings:**
   1. Update your repository’s remote settings:
      ```bash
      git remote rename origin upstream                                              # Rename the existing 'origin' remote to 'upstream'
      git remote add origin git@github.com:<your_username>/<personal-repo-name>.git  # Add your new private repository as 'origin'
      git remote set-url --push upstream DISABLE                                     # Disable push access to the public repository
      ```
      This configuration allows you to push changes to your private repository while maintaining a reference to the public repository. By renaming the original remote and adding your personal repo as the new `origin`, you ensure that your work is kept private and secure, while still being able to fetch updates from the public repository.

   2. Push to your new repository:
       ```bash
       git push origin
       ```  
     
   3. Set the `origin` as the default push destination.
       ```bash
       git branch --set-upstream-to=origin/main
       ```

       This command configures your local `main` branch to push changes to your private repository by default. It ensures that when you use `git push`, your commits will be sent to your private repo instead of the public one.

   Later, if there are new commits in the public repository (`tudelft/AE4353-Y24`), you can incorporate these changes into your private repository by pulling from the `upstream` and rebasing your work on top of it:
   
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
   If there are any conflicts during the rebase, resolve them as needed. This process ensures your private repository stays up-to-date with the public repository while maintaining your own changes.


Well done! Your GitHub Codespaces setup is complete, and you are ready to start working on the project. Next, proceed to the [Usage](#usage) section to learn how to run the code.

## Usage
To use your configured GitHub Codespace, please follow the steps below:

1. Navigate to the Jupyter notebook you wish to use. When you open it, the kernel may not be pre-selected. In the top right corner of the notebook, click on `Select Kernel`, then go to `Python Environments` and select the recommended kernel, which should be `AE4353 ...`.

2. Download the required dataset(s) for the exercise or competition from [SurfDrive](https://surfdrive.surf.nl/files/index.php/s/QzvOHJx2o4KIESI) and move the file(s) to the `data` folder. You can do this by either dragging and dropping the files or by right-clicking on the `data` folder and selecting `Upload`.

    > 💡 **Keep the file in its original format (e.g., `.npz`) without unzipping or extracting any contents unless explicitly instructed.**
    >
    > 💡 Uploading the file may take a few minutes. You can monitor the progress in the bar at the bottom.

3. When you have finished using your Codespace, please ensure that you stop it to avoid unnecessary usage and costs. For more details on how to manage your Codespace, refer to the [Managing Your Codespace](#managing-your-codespace) section.

Good job! You are now ready to start working on the project using the pre-configured environment in your GitHub Codespace. Happy coding! 🚀

## Managing Your Codespace
Once your Codespace is running, it may be necessary to manage it to avoid additional costs. Here are some key actions you can take:

- **Stopping**: If you want to pause your work and avoid further billing, you can stop your Codespace. Click the three dots next to your Codespace in the Codespaces list and select `Stop codespace`. This action saves your current state and pauses any associated costs, allowing you to resume where you left off later.

- **Deleting**: If you no longer need a Codespace, you can delete it to free up resources and stop any further charges. Click the three dots next to your Codespace and select `Delete`. Be sure to save all your work before performing this action, as deleting a Codespace is permanent and cannot be undone.

## Additional Resources
For more detailed information about GitHub Codespaces, refer to the following resource:

- [GitHub Codespaces Documentation](https://docs.github.com/en/codespaces).
