## README - Deeplabcut setup for Cichlid Bower Repository

### Dockerfile & docker-compose.yml
These files  are for local setup to setup a docker container 
and run the deeplabcut GUI for dataset curation and generation of a training set.
Follow the setup doc here: [deeplabcut gui docker container setup](#dlc-docker-setup)

### Deeplabcut_1_Notebook.ipynb
This file can be run in google colab to train a model. Simply follow the instructions
in the [deeplabcut google colab walkthrough](#google-colab-walkthrough)
 

### DLC docker setup
1. Install [VcXsrv](https://sourceforge.net/projects/vcxsrv/)
    - This is an application that acts as a server and will launch GUI apps.
	- You can use other applications like Xming, MobaXterm, etc. 
	Use the link to download, install, and run the application using the default options.
	- To use VcXsrv, run the XLaunch application (you can find it by searching XLaunch
	in the Windows Search Bar) and you will be prompted with a window asking you 
	to configure various settings before launching the X Server.
	- Keep clicking **Next** to accept all the defaults and click **Finish** once it shows up.
2. You should now have an X-server running which should be visible in your taskbar 
    - ![My taskbar in Windows](walkthrough_images/XMING_taskbar.png)
	

 
### Google Colab Walkthrough
