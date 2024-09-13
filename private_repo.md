To keep your work private and save it on GitHub, consider creating a personal repository for your code related to exercises and competitions. The `tudelft/AE4353-Y24` repository is public, and GitHub does not support private forks of public repositories. Instead, you can create a private repository that mirrors the public one, allowing you to work privately while still keeping your code synchronized with the public repository.

 1. Clone this repository. For Windows users, please do the clone in the **Ubuntu** terminal!
    ```bash
    cd ~
    git clone git@github.com:tudelft/AE4353-Y24.git
    ```

    > ⚠️ **For MacOS users:** If you have iCloud Drive syncing enabled, avoid cloning the repository into the `Desktop` or `Documents` folders. If you need to clone into one of these locations, rename the cloned folder to `AE4353-Y24.nosync` to prevent automatic syncing.

 2. Go to [GitHub's repository creation page](https://help.github.com/articles/creating-a-new-repository/) and create a new private repository with a name of your choice (e.g., `<personal-repo-name>`).
 
    > ⚠️ **Important:** When creating the repository, ensure that you do **NOT** check the "Add a README file" option. Set both the .gitignore template and License to **None**. Additionally, make sure the repository visibility is set to **Private**.
    > 
    > 💡 If you cannot create a private repository, you can request unlimited private repos by applying for the [GitHub Student Pack](https://education.github.com/pack).
     
3. Update your repository’s remote settings:

   ```bash
   cd AE4353-Y24                                                                  # Navigate to your local repository
   git remote rename origin upstream                                              # Rename the existing 'origin' remote to 'upstream'
   git remote add origin git@github.com:<your_username>/<personal-repo-name>.git  # Add your new private repository as 'origin'
   git remote set-url --push upstream DISABLE                                     # Disable push access to the public repository
   ```

   This configuration allows you to push changes to your private repository while maintaining a reference to the public repository. By renaming the original remote and adding your personal repo as the new `origin`, you ensure that your work is kept private and secure, while still being able to fetch updates from the public repository.
 
 4. Push to your new repository:
    ```bash
    git push origin
    ```
   
5. Set the `origin` as the default push destination.
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
