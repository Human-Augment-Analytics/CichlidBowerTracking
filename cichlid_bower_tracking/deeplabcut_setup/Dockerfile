FROM deeplabcut/deeplabcut:latest-gui

# https://askubuntu.com/questions/1444943/nvidia-gpg-error-the-following-signatures-couldnt-be-verified-because-the-publi
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


# Install additional dependencies if needed
RUN apt-get update && apt-get install -y \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*
# RUN apt-get install -y python3-tk && rm -rf /var/lib/apt/lists/*

# Set the display environment variable
ENV DISPLAY=host.docker.internal:0.0
