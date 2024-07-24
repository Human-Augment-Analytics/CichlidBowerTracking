# Cichlid Bower Tracking

<!-- omit in toc -->
## Table of Contents
 - [How to Setup Conda Environment](#how-to-setup-conda-environment)
   - [Windows WSL2 (Ubuntu x86)](#windows-wsl2-ubuntu-x86)
 - [How to Setup Rclone Remote](#how-to-setup-rclone-remote)
   - [Windows WSL2 (Ubuntu x86)](#windows-wsl2-ubuntu-x86-1)
 - [How to Use with PACE ](#how-to-use-with-pace)
   - [ICE Cluster](#ice-cluster)
 - [References](#references)

## How to Setup Conda Environment

This section explains how to setup the environment used by this repo. If your preferred method isn't listed here, please feel free to add the step-by-step process here.

### Windows WSL2 (Ubuntu x86)

1. Open the WSL2 command line in `cmd.exe`.
2. Create a directory for miniconda3 by entering the command `mkdir -p ~/miniconda3` [1](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).
3. Get `miniconda.sh` from the official server by entering `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh` [1](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).
4. Enter the command `bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3` [1](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).
5. Delete `miniconda.sh` by entering the command `rm -rf ~/miniconda3/miniconda.sh` [1](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).
6. Enter the command `~/miniconda3/bin/conda init bash` and then close the WSL2 command line tab in `cmd.exe` [1](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).
7. Re-open the WSL2 command line in `cmd.exe` and open `~/.bashrc` in the nano editor by entering the command `nano ~/.bashrc` [2](https://medium.com/@sawepeter6/conda-command-not-found-ac28bea24291).
8. Add `export PATH=~/miniconda3/bin:$PATH` to the end of the `~/.bashrc` file using nano, and save the changes.
9. Exit the nano editor and run the command `source ~/.bashrc` to enact the changes made.
10. Open a Windows Powershell tab in `cmd.exe` and run the command `wsl --shutdown` [4](https://stackoverflow.com/questions/67923183/miniconda-on-wsl2-ubuntu-20-04-fails-with-condahttperror-http-000-connection).
11. Reopen the WSL2 commmand line in `cmd.exe` and enter the command `conda --version`; if you get a version number as output, miniconda is correctly installed.
12. Traverse to the desired directory in your file system and clone this repo using `git clone https://github.com/Human-Augment-Analytics/CichlidBowerTracking.git`. 
13. Setup the environment by entering the command `conda env create -f cichlidbowertracking.yml` [3](https://stackoverflow.com/a/59686678).
14. If you get the error `CondaSSLError: Encountered an SSL error. Most likely a certificate verification issue.` try running the previous command again and it will likely pickup with the setup exactly where it left off before the error.

## How to Setup Rclone Remote

This section explains how to setup the rclone remote used to connect to the Dropbox. If your preferred method isn't listed here, please feel free to add the step-by-step process here.

### Windows WSL2 (Ubuntu x86)

1. Install rclone using the Ubuntu command line, e.g., `sudo apt install`.
2. Download the `rclone` file from `BioSci-McGrath/Apps/CichlidPiData/__CredentialFiles/` Dropbox directory.
3. Move the downloaded `rclone` file from your Windows Downloads folder to `/home/<wsl-user>/.config/rclone/`.
4. Run the command `rclone config` and rename the rclone remote from `cichlidVideo` to `CichlidPiData`.
5. Reconnect to the rclone remote by running `rclone config reconnect CichlidPiData:` and following the provided prompts.

## How to Use with PACE 

### ICE Cluster

1. Download Georgia Tech's VPN from [https://vpn.gatech.edu/global-protect/login.esp](https://vpn.gatech.edu/global-protect/login.esp) for your specific OS and turn it on (have your Duo Sign-in device ready) [5](https://vpn.gatech.edu/global-protect/login.esp).
2. Download VS Code onto your device.
3. Set up SSH support in VS Code using the instructions at [https://code.visualstudio.com/docs/remote/ssh](https://code.visualstudio.com/docs/remote/ssh) and access the ICE cluster in VS Code.
4. Install/Import rclone to your home path (`~`) and add the following lines to your `~/.bashrc`:

```bash
PATH=$HOME/rclone:$PATH
export PATH
```

5. Install a conda distro and create the "CichlidBowerTracking" environment, see [How to Setup Conda Environment](#how-to-setup-conda-environment) for more guidance.
6. Change your working directory to `~/ondemand` and clone this repo using the command `git clone https://github.com/Human-Augment-Analytics/CichlidBowerTracking.git`.
7. To run any scripts in this repo, you should make sure the stored value of the `USING_PACE` constant in `CichlidBowerTracking/cichlid_bower_tracking/misc/pace_vars.py` is `True`.
8. Also, make sure your working directory is set to `CichlidBowerTracking/cichlid_bower_tracking` before running any scripts in this repo.

## References
1. "Miniconda: Quick command line install," docs.anaconda.com. [https://docs.anaconda.com/free/miniconda/#quick-command-line-install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).
2. P. Sawe, "Conda Command Not Found," medium.com. [https://medium.com/@sawepeter6/conda-command-not-found-ac28bea24291](https://medium.com/@sawepeter6/conda-command-not-found-ac28bea24291).
3. L. Gonzalez, "How to make new anaconda env from yml file," stackoverflow.com. [https://stackoverflow.com/a/59686678](https://stackoverflow.com/a/59686678).
4. "Miniconda on WSL2 (Ubuntu 20.04) fails with CondaHTTPError: HTTP 000 CONNECTION FAILED," stackoverflow.com. [https://stackoverflow.com/questions/67923183/miniconda-on-wsl2-ubuntu-20-04-fails-with-condahttperror-http-000-connection](https://stackoverflow.com/questions/67923183/miniconda-on-wsl2-ubuntu-20-04-fails-with-condahttperror-http-000-connection).
5. "Remote Development using SSH," code.visualstudio.com. [https://code.visualstudio.com/docs/remote/ssh](https://code.visualstudio.com/docs/remote/ssh).