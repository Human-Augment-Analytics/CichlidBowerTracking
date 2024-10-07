# Training DeepLabCut on ICE compute node

This guide will walk you through the specific steps and pitfalls when training a `DeepLabCut` model on HAAG's Cichlid data that are annotated with keypoints (body parts) on an ICE compute node.

These steps follow [the general instructions from DLC](https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html) for multi-animal pose estimation and tracking. The entire DLC pipeline includes data creation, GUI, training, evaluation and inference (animal tracking). These steps pertain to training only, with the annotation data already created, and are meant to quickly get to the training steps with most default settings kept unchanged.

Please see these references for other aspects of DeepLabCut:

- Instructions for multi-animal pose estimation training: https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html#create-training-dataset
- Paper "Multi-animal pose estimation, identification and tracking with DeepLabCut" (https://www.nature.com/articles/s41592-022-01443-0#Fig1)
- Paper "DeepLabCut: markerless pose estimation of user-defined body parts with deep learning" (https://www.nature.com/articles/s41593-018-0209-y)
- The second DLC paper utilizes Part Affinity Fields, which was introduced in this paper "Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields" (https://arxiv.org/abs/1611.08050)

## DLC Installation

- Download or copy the content of this file into a local `DEEPLABCUT.yaml`: https://github.com/DeepLabCut/DeepLabCut/blob/main/conda-environments/DEEPLABCUT.yaml
- Use `conda env create` to install DLC and dependencies

```
conda env create -f DEEPLABCUT.yaml
```

## Download the annotation data

- Download this folder onto the scratch folder on your ICE node (permission needed): https://drive.google.com/drive/folders/1vH_g827bM39VrxGgzLm59u9qPR_9C0co. One way to do this is by launching an Interactive Desktop session on ICE via Open OnDemand. Follow this link: https://ondemand-ice.pace.gatech.edu/pun/sys/dashboard/batch_connect/sys/bc_desktop_rh9/session_contexts/new. Once inside the interactive desktop, open a browser and download the data folder from Google Drive (may need to log in).

## Request a compute node with GPUs to test-run `deeplabcut`

- To make sure that DLC can get to the training steps, log into a compute node and start a terminal (either SSH or via an interactive desktop (link above) or VS Code session: https://ondemand-ice.pace.gatech.edu/pun/sys/dashboard/batch_connect/sys/bc_codeserver_rh9/session_contexts/new). Request GPUs to prepare for training.
- Folder structure of the downloaded folder should look like this:

```.../dlc_model-student-2023-07-26
  ├── config.yaml
  ├── dlc-models
  ├── labeled-data
  ├── training-datasets
  └── videos
```

- Activate the environment (assuming you named it `DEEPLABCUT` in the YAML file): `conda activate DEEPLABCUT`

## Call `deeplabcut.create_multianimaltraining_dataset`


- Inspect the `config.yaml` file. Make sure to update the project path to point to your newly downloaded project folder (`dlc_model-student-2023-07-26`), and set the `engine` to `pytorch`. Review the rest of the YAML settings, which have also been set up for fish pose estimation (body parts, skeleton, etc.)

  ```
  # Project path (change when moving around)
  project_path: /…/dlc_model-student-2023-07-26


  # Default DeepLabCut engine to use for shuffle creation (either pytorch or tensorflow)
  engine: pytorch

  ```
- *IMPORTANT*: Some of the folders in `labeled-data` are corrupted (according to Adam Thomas). In order to tell DLC to ignore these files, edit the `config.yaml` file and comment out these data files within the `video_sets` field:

```
# Annotation data set configuration (and individual video cropping parameters)
video_sets:
  # ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc21_3_Tk53_021220_0004_vid
  # : crop: 0, 1296, 0, 972
  # ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc21_6_Tk53_030320_0002_vid
  # : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc23_1_Tk33_021220_0004_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc23_8_Tk33_031720_0001_vid
  : crop: 0, 1296, 0, 972
  # ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc24_4_Tk47_030320_0002_vid
  # : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc26_2_Tk63_022520_0001_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc28_1_Tk3_022520_0004_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc29_3_Tk9_030320_0001_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc32_5_Tk65_030920_0002_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc34_3_Tk43_030320_0001_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc35_11_Tk61_051220_0002_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc36_2_Tk3_030320_0001_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc37_2_Tk17_030320_0001_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc40_2_Tk3_030920_0002_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc41_2_Tk9_030920_0001_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc43_11_Tk41_060220_0001_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc44_7_Tk65_050720_0002_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc45_7_Tk47_050720_0002_vid
  : crop: 0, 1296, 0, 972
  # ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc46_2_Tk53_030920_0002_vid
  # : crop: 0, 1296, 0, 972
  # ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc51_2_Tk53_031720_0001_vid
  # : crop: 0, 1296, 0, 972
  # ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc54_5_Tk53_051220_0002_vid
  # : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc55_2_Tk47_051220_0002_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc56_2_Tk65_051220_0002_vid
  : crop: 0, 1296, 0, 972
  # ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc58_4_Tk53_060220_0001_vid
  # : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc59_4_Tk61_060220_0001_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc62_3_Tk65_060220_0001_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc63_1_Tk9_060220_0001_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc64_1_Tk51_060220_0002_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc65_4_Tk9_072920_0002_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc76_3_Tk47_072920_0002_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc80_1_Tk41_072920_0002_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc81_1_Tk51_072920_0002_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc82_b2_Tk63_073020_0002_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc86_b1_Tk47_073020_0002_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc90_b1_Tk3_081120_0001_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc91_b1_Tk9_081120_0001_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc94_b1_Tk31_081120_0001_vid
  : crop: 0, 1296, 0, 972
  ? /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/labeled-data/MC_singlenuc96_b1_Tk41_081120_0001_vid
  : crop: 0, 1296, 0, 972
```
- *NOTE*: If there are already-created training datasets (which DLC calls shuffles) inside the project folder, these training shuffles might be marked as using Tensorflow engine or using PyTorch engine. In this guide, we want to ensure that DLC uses the PyTorch engine. So let’s erase the previously-created training shuffles and create a brand-new one.
From the project folder, for example `dlc_model-student-2023-07-26`, erase the folders that say `dlc-models-pytorch` or `dlc-models` and the `training-datasets` folder, so that the tree looks like this (must keep `labeled-data` and `config.yaml` and `videos`)
```
(base) [… dlc_model-student-2023-07-26]$ tree -L 1
.
├── config.yaml
├── labeled-data
└── videos
```
Then let’s run this Python script (be sure that the Python interpreter is the correct environment, and that `config_path` points to the correct place on your machine):
```
config_path = "…/dlc_model-student-2023-07-26/config.yaml"
deeplabcut.create_multianimaltraining_dataset(config_path)
```
Output should look successful like this:
```
Loading DLC 3.0.0rc5...
DLC loaded in light mode; you cannot use any GUI (labeling, relabeling and standalone GUI)
Utilizing the following graph: [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8], [4, 5], [4, 6], [4, 7], [4, 8], [5, 6], [5, 7], [5, 8], [6, 7], [6, 8], [7, 8]]
Creating training data for: Shuffle: 1 TrainFraction:  0.95
100%|██████████████████████████████████████████████████████████████████████████████████████| 1472/1472 [00:39<00:00, 37.30it/s]
The training dataset is successfully created. Use the function 'train_network' to start training. Happy training!
```
- Confirming the previous step: The project folder structure now looks like this. DLC created the `dlc-models-pytorch` folder and `training-datasets` folder, since its latest version supports a PyTorch engine.

  ```.
  ├── config.yaml
  ├── dlc-models-pytorch
  ├── labeled-data
  ├── training-datasets
  └── videos
  ```
Now if we inspect the `training-datasets` folder, we should have this tree, which shows the newly-created training shuffle.
```
(base) [… dlc_model-student-2023-07-26]$ cd training-datasets/
(base) [… training-datasets]$ tree -L 3
.
└── iteration-0
    └── UnaugmentedDataSet_dlc_modelJul26
        ├── CollectedData_student.csv
        ├── CollectedData_student.h5
        ├── dlc_model_student95shuffle1.pickle
        ├── Documentation_data-dlc_model_95shuffle1.pickle
        └── metadata.yaml
```
Go into this `metadata.yaml` file to confirm that there’s only one shuffle, and it’s marked as using PyTorch engine:
`… dlc_model-student-2023-07-26/training-datasets/iteration-0/UnaugmentedDataSet_dlc_modelJul26/metadata.yaml`
```
# This file is automatically generated - DO NOT EDIT
# It contains the information about the shuffles created for the dataset
---
shuffles:
  dlc_modelJul26-trainset95shuffle1:
    train_fraction: 0.95
    index: 1
    split: 1
    engine: pytorch
```
Then DLC would know to use PyTorch engine for this training shufffle.


## Call `deeplabcut.train_network`

- Run this code (be sure to modify `config_path`) to see if DLC gets to the first training step, then exit.

  ```dlc_training_script.py
  import deeplabcut

  config_path = ".../dlc_model-student-2023-07-26/config.yaml"

  # Execute the training function
  deeplabcut.train_network(config_path, shuffle=1, trainingsetindex=0,)
  ```
- If the script could not get to a first training step and threw error, inspect this YAML file that controls some training settings: `.../dlc_model-student-2023-07-26/dlc-models-pytorch/iteration-0/dlc_modelJul26-trainset95shuffle1/train/pytorch_config.yaml`

  - If there is the CUDA OutOfMemory error, you need to request a compute node with larger GPU memory, or reduce the batch size by half. You'll also need to decide how many epochs to train via the settings below.

  ```
  ...
  train_settings:
  batch_size: 1
  dataloader_workers: 0
  dataloader_pin_memory: false
  display_iters: 100
  epochs: 180
  seed: 42
  ```

  - The `eval_interval` setting under `Runner` controls how many epochs will the model attempts an evaluation after:

  ```
  runner:
  type: PoseTrainingRunner
  gpus:
  - 0
  key_metric: test.mAP
  key_metric_asc: true
  eval_interval: 10
  optimizer:
      type: AdamW
      params:
      lr: 0.0001
  ```

  Initially it is set to 10. However, evaluating DLC too early seems to always lead the program to crash. My theory is that internally, DLC's neural network outputs one "heatmap" for each of the body parts (tail fin, etc.) that we have defined for the cichlid. The heatmap, which represents the video frame being processed, indicates the probability that a given point in that video frame could be the corresponding body part. In other words, the local "peaks" in a heatmap likely indicate where a body part (keypoint) should be. At the beginning, the model has not been trained very well, so the heatmaps look more or less random, and there will be too many possible keypoint locations, leading to a crash.

  One workaround is to postpone evaluation until after training, hoping that by that time, the model will have learned to produce good heatmaps with only few local maxima. For example, we could set the `eval_interval` to a larger number than the `epochs` settings to ensure that evaluation is never done while training.

  ```
  runner:
  type: PoseTrainingRunner
  gpus:
  - 0
  key_metric: test.mAP
  key_metric_asc: true
  eval_interval: 300
  optimizer:
      type: AdamW
      params:
      lr: 0.0001
  ```

  - The training code does not appear to automatically use all available GPUs. I raise a GitHub issue with DeepLabCut - please refer for further details, if you want to force `deeplabcut` to use all available GPUs: https://github.com/DeepLabCut/DeepLabCut/issues/2744. Otherwise, it defaults to using one GPU. Oddly enough, I found that multi-GPU training did not seem to result in a boost in training speed, so I trained with one GPU.
- If, with the `train_network()` function, DLC can get to the first training step, you should still monitor a few more training steps to ensure there are no memory errors.

## Training for an extended period

- Once you're convinced that the model can be trained properly, there are options for running the training to completion: using interactive VS Code or terminal session or a batch job.
- On an interactive session, you can use `tmux` to launch the training script above and have `tmux` keep track of the process such that you can temporarily close the interactive session or exit the VS Code browser window and later go back into the training process that `tmux` still manages.

  - Create a new `tmux` session by running this command in the terminal:

  ```
  tmux new-session -t dlc-training
  ```
  Then launch any script you want as in any terminal. `Ctrl + B`, then `D` to detach from `tmux` while the script is still running.
  Later, attach again to view the script by running in the terminal:

  ```
  tmux attach-session -t dlc-training
  ```
- If using a batch job, here's an SBATCH request that you can use. Assuming `test_training_script.py` is the name of the Python script that contains `.train_network()` function call above. Create the file `dlc_training.sbatch` with this content:

  ```
  #!/bin/bash
  #SBATCH -JDLC_model_training1
  #SBATCH -N1 --ntasks-per-node=1
  #SBATCH --gres=gpu:H100:1
  #SBATCH --cpus-per-task=4
  #SBATCH --mem=64GB
  #SBATCH -t0-08:00:00     # If requesting 8 hours, enough to run 90 epochs. If 180 epochs, may need 16 hours.
  #SBATCH -oDLC_Training_Test1_Report-%j.out                
  #SBATCH --mail-type=BEGIN,END,FAIL

  cd $SLURM_SUBMIT_DIR                     # Change to working directory

  source /home/hice1/tnguyen868/anaconda3/bin/activate /path/to/DLC/environment/folder
  export PATH="/path/to/DLC/environment/folder/bin:$PATH"

  srun python test_training_script.py

  ```
  Then request the batch job by changing to the same folder where `test_training_script.py` is located, and run:

  ```
  sbatch dlc_training.sbatch
  ```
  Please refer to [PACE Documentation](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042503) for details on `.sbatch` syntax.

  Specifically, you check the status of your job by running from a terminal shell: `squeue -u <username>`

  You can peek at your output file while the batch job is running to see the console output so far, for example:
  ```
  tail -f DLC_Training_Test1_Report-698516.out
  ```

  The output so far might look like this:
  ```
  This is good for small batch sizes (e.g., when training on a CPU), where you should keep ``freeze_bn_stats=true``.
  If you're using a GPU to train, you can obtain faster performance by setting a larger batch size (the biggest power of 2 where you don't geta CUDA out-of-memory error, such as 8, 16, 32 or 64 depending on the model, size of your images, and GPU memory) and ``freeze_bn_stats=false`` for the backbone of your model.
  This also allows you to increase the learning rate (empirically you can scale the learning rate by sqrt(batch_size) times).
  
  Using 1472 images and 78 for testing
  
  Starting pose model training...
  --------------------------------------------------
  Number of iterations: 500, loss: 0.00740, lr: 0.0001
  Number of iterations: 1000, loss: 0.00521, lr: 0.0001
  ```