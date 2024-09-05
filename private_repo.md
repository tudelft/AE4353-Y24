You might want to create a private personal repo to push your code written for the exercises and competition. This repository is made public for everyone and Github does not allow the creation of private forks for public repositories. You could, however, make a private repo which mirrors this repo for your personal use. 

 1. Clone this repository. For Windows users, please do the clone in WSL2!
    ```bash
    cd ~
    git clone git@github.com:tudelft/AE4353-Y24.git
    ```

 2. [Create a new private repository on Github](https://help.github.com/articles/creating-a-new-repository/) and name it as you like: <personal-repo-name>.
    > If you are unable to create a private repo, you can request unlimited private repos as a student by getting
    > the [student pack](https://education.github.com/pack) from Github.

 3. Rename the old URL and add the personal repo URL as an upstream. Disable pushing to the public repo for which you do not have push access.
    ```bash
    cd AE4353-Y24
    git remote rename origin upstream
    git remote add origin git@github.com:<your_username>/<personal-repo-name>.git
    git remote set-url --push upstream DISABLE
    ```
 
 4. Push to your new repository.
    ```bash
    git push origin
    ```

 5. Set origin as default push destination.
    ```bash
    git branch --set-upstream-to=origin/main
    ```
    
   
Later, when we push new commits to the public repo, you can pull changes from `upstream` and rebase on top of your work.
```bash
git fetch upstream
git rebase upstream/main
```
And solve the conflicts if any.
