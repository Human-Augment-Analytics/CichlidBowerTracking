# Cichlid Bower Tracking

<!-- omit in toc -->
## Table of Contents
 - [How to Install](#how-to-install)
   - [Windows WSL2 (Ubuntu x86)](#windows-wsl2-ubuntu-x86)
 - [References](#references)

## How to Install

This section explains how to setup the environment used by this repo. If your preferred method isn't listed here, please feel free to add the step-by-step process here.

### Windows WSL2 (Ubuntu x86)

1. Open the WSL2 command line in `cmd.exe`.
2. Create a directory for miniconda3 by entering the command `mkdir -p ~/miniconda3` [1](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).
3. Get `miniconda.sh` from the official server by entering `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh` [1](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).
4. Enter the command `bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3` [1](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).
5. Delete `miniconda.sh` by entering the command `rm -rf ~/miniconda3/miniconda.sh` [1](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).
6. Open `~/.bashrc` in the nano editor by entering the command `nano ~/.bashrc` [2](https://medium.com/@sawepeter6/conda-command-not-found-ac28bea24291).
7. Add `export PATH=~/miniconda3/bin:$PATH` to the end of the `~/.bashrc` file using nano, and save the changes.
8. Exit the nano editor and run the command `source ~/.bashrc` to enact the changes made.
9. Open a Windows Powershell tab in `cmd.exe` and run the command `wsl --shutdown` [4](https://stackoverflow.com/questions/67923183/miniconda-on-wsl2-ubuntu-20-04-fails-with-condahttperror-http-000-connection).
10. Reopen the WSL2 commmand line in `cmd.exe` and enter the command `conda --version`; if you get a version number as output, miniconda is correctly installed.
11. Traverse to the desired directory in your file system and clone this repo using `git clone https://github.com/charlesrclark1243/CichlidBowerTracking.git`. 
12. Setup the environment by entering the command `conda env create -f cichlidbowertracking.yml` [3](https://stackoverflow.com/a/59686678).
13. If you get the error `CondaSSLError: Encountered an SSL error. Most likely a certificate verification issue.` try running the previous command again and it will likely pickup with the setup exactly where it left off before the error.

## References
1. "Miniconda: Quick command line install," docs.anaconda.com. [https://docs.anaconda.com/free/miniconda/#quick-command-line-install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).
2. P. Sawe, "Conda Command Not Found," medium.com. [https://medium.com/@sawepeter6/conda-command-not-found-ac28bea24291](https://medium.com/@sawepeter6/conda-command-not-found-ac28bea24291).
3. L. Gonzalez, "How to make new anaconda env from yml file," stackoverflow.com. [https://stackoverflow.com/a/59686678](https://stackoverflow.com/a/59686678).
4. "Miniconda on WSL2 (Ubuntu 20.04) fails with CondaHTTPError: HTTP 000 CONNECTION FAILED," stackoverflow.com. [https://stackoverflow.com/questions/67923183/miniconda-on-wsl2-ubuntu-20-04-fails-with-condahttperror-http-000-connection](https://stackoverflow.com/questions/67923183/miniconda-on-wsl2-ubuntu-20-04-fails-with-condahttperror-http-000-connection).