# AE4353 Uninstallation Guide:
This is the manual to clear the course data/add-ons for the course AE4353. 

**Please make sure to read the complete section before inputting the lines of code into your system, as some of the sections have warnings for each command directly after the command line.**

The uninstallation guide is broken up into the following sections:
1. Docker
2. WSL2 (Only for windows users)
3. VS Code and Extensions
4. Local Repository/Files

Feel free to un-install the components you see fit, keeping the ones you determine are useful for future use. Please note, this is a general guide, and should be used as such. If you are using Docker for another project, following the command that completely clears out all docker files, images and containers is not a great idea - so **please follow the instructions accordingly**.

Any further questions about the uninstallation of the course material can be posted on the discussion page: https://github.com/tudelft/AE4353-Y24/discussions

## Uninstall Manual:

For this manual, the word sudo is not included, but could be necessary in part where a base user does not have permissions. This is not a problem for those who have followed the course in Windows, as it is a rootless setup. Therefore, for those who have a Linux setup, please use sudo where necessary.

#

### Docker:

#### Docker Containers and Images:

using WSL2 (either the extension in VS-code or as the loose terminal), run the following to see your terminal:

**1. check if docker is running:**

```
service status docker
```
if the status shows active, then proceed to the next step, if not:

```
sudo service start docker
```

**2. Run the following to see which docker images you have:**
```
sudo docker images -a
```

**3. Remove all detached (dangling) and un-used docker images:**

dangling images are ones that may have been created but are not connected to a container, but still take up space. remove them by running the following command:

``` 
sudo docker image prune
```

to remove **all unused images** (not just dangling) images:

```
sudo docker image prune -a
```

**4. Removing other docker images:**

Now that you have cleaned up the docker images that are loose and un-used, you may want to remove images tied to containers and that are used. To do this follow the next instructions:

Check the docker images you have:

```
sudo docker images -a
```

From these docker images, specific containers may still be running. These first need to be stopped in order to remove them. Therefore check which containers are still running:

```
sudo docker ps
```

If you plan on removing this container, first stop it (replace ```<CONTAINER_ID>```from the list of running containers you got above):

```
sudo docker stop <CONTAINER_ID>
sudo docker rm <CONTAINER_ID>
```

you can remove all stopped containers with the following command:

```
sudo docker container prune
```

Next, you will need to clean out docker images. This can be done as follows:

Remove individual images using the specific IMAGE_ID shown (replace ```<IMAGE_ID>``` with the specific IMAGE_ID shown for the docker image from the command above):

```
sudo docker rmi <IMAGE_ID>
```

#### Uninstalling Docker (WSL2):

If you uninstall **Docker Desktop**, the images and containers stored inside WSL2 may still exist inside the WSL distribution. To fully remove the Docker Data:

1. **Uninstall Docker Desktop** (Control panel -> "Add or Remove Programs").

2. **Manually** Delete Docker Data:

```
rm -r ~/.docker /var/lib/docker
```

#### Uninstalling Docker (Linux - Ubuntu / Debian):
For Linux, uninstalling Docker **does not** remove images and containers. Therefore, to fully clear everything:

1. **uninstall  Docker:**
``` 
sudo apt remove --purge docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

2. If you have already cleared all images, contianers and volumes **you want removed**, then the following is not necessary. However if you want a clean slate and completely remove all docker images, containers and volumes:

```
sudo rm -r /var/lib/docker
```

to remove docker config files:

```
sudo rm -r ~/.docker
```

#### Uninstalling Docker (macOS):
1. **Delete Docker Desktop** from your applications
2. remove all docker data (terminal):

```
rm -r ~/docker /var/lib/docker
```
#

### WSL2 (Only for Windows users!):

#### 1. Remove all Linux Distributions:

Before removing WSL2, you need to first uninstall all installed Linux distributions:

1. **Open PowerShell as Administrator**

2. **List all installed WSL distributions:**
```
wsl --list --verbose
```

3. **Unregister (delete) each distribution:**

replace each ```<distro-name>``` with your distribution name (e.g., ```Ubuntu-22.04```):

```
wsl --unregister <distro-name>
```

**WARNING: This will *Permenently delete* the Linux filesystem, including all files and configurations inside WSL.**

#### 2. Disable WSL2 and Uninstall its components:
1. Disable WSL features:
```
wsl --shutdown
```

2. disable WSL:
```
dism.exe /online /disable-feature /featurename:Microsoft-Windows-Subsystem-Linux /norestart
```

3. Restard your computer (to apply changes):
```
sutdown /r /t 0
```

#### 3. Verify WSL is fully uninstalled:
after restarting, check if WSL is gone:

```
wsl --list
```
if you get an error along the lines of *"WSL is not recognized"* then WSL was successfully removed.

#

### VS Code (Optional):

For those who intend to keep coding (hopefully everyone!), Visual Studio Code is an extremely useful tool to use and develop your programming abilities. There is extremely good documentation on how to install / uninstall extensions (devcontainers, docker, git, ...). 

Any and all information on Visual Studio Code use the following link: https://code.visualstudio.com/docs

Information on how to remove VS Code and its extensions can be found there.

#

### AE4353 - dataset / local repository:

As you pushed your code during the course to GitHub, any and all your progress is stored online, and therefore (if you don't plan on working on the dataset or course exercises any longer), the local files can be deleted.

#### Windows and macOS:

As its done for any file you previously stored on your distribution, locate the files in the file sytem UI and simply hit delete, then empty your trash folder.

#### Linux:

Locate where you stored the course files by using the search command in terminal:
```
find / -type d -iname "AE4353-Y24" 2>/dev/null
```

Once you find the path you were looking for, copy it and then in your terminal type the following (replace ```<PATH>``` with your path you copied from above):

```
cd <PATH>
cd ..
```

Following this, remove the repository from your laptop:

```
rm -r AE4353-Y24/
```

This will remove the full AE4353-Y24 repository. 

If you have other files relating to the course, you can remove them in a similar way (i.e. find the folder, go one folder above the folder which you want to delete, then remove it).