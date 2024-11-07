# Training DeepLabCut on ICE compute node

This guide will walk you through the specific steps and pitfalls when training and evaluating a `DeepLabCut` model on HAAG's Cichlid data that are annotated with keypoints (body parts) on an ICE compute node.

These steps follow [the general instructions from DLC](https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html) for multi-animal pose estimation and tracking. The entire DLC pipeline includes data creation, GUI, training, evaluation and inference (animal tracking). These steps assume that annotation data have already been created, and are meant to quickly guide you through training and evaluation. Thus, most settings are kept as default.

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
   - 70
      - 90
  snapshots:
    max_snapshots: 15
    save_epochs: 10
    save_optimizer_state: false
  train_settings:
    batch_size: 8
    dataloader_workers: 0
    dataloader_pin_memory: false
    display_iters: 500
    epochs: 100
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

## Continue training (optional)
  If your compute node reaches time limit and you have to stop training, you can continue training with this Python code (please locate your desired snapshot to continue training from, and change the path accordingly):
  ```
  deeplabcut.train_network(config_path, shuffle=1, trainingsetindex=0, snapshot_path="<project folder> /dlc_model-student-2023-07-26/dlc-models-pytorch/iteration-0/dlc_modelJul26-trainset95shuffle1/train/snapshot-100.pt")
  ```

## Evaluating the network
  Once the network is trained, you can evaluate with this Python code (modify your path to `config.yaml` accordingly):
  ```
  config_path = "<project folder> /dlc_model-student-2023-07-26/config.yaml"
  deeplabcut.evaluate_network(config_path, plotting=True)
  ```
  From here on, it’s important to understand the files that DeepLabCut is creating for us to save the results of evaluation or visualization.
  
  After this code, you’ll see `evaluation-results-pytorch` folder, where the evaluation results are stored in `.csv` and `.h5` files and visualization is created.
  ```
  .
  ├── config.yaml
  ├── dlc-models-pytorch
  ├── evaluation-results-pytorch
  ├── labeled-data
  ├── training-datasets
  └── videos
  ```
  The visualization results here are the same training and testing image frames (provided in original dataset), but now they show both the ground-truth keypoints (body parts) and model-predicted keypoints. (Ground-truth keypoints are marked with “+”s, while model-predicted keypoints are round.)
  
  The visualized images are saved in a folder that has a path similar to this: ` <project folder>/dlc_model-student-2023-07-26/evaluation-results-pytorch/iteration-0/dlc_modelJul26-trainset95shuffle1/LabeledImages_DLC_Resnet50_dlc_modelJul26shuffle1_snapshot_200`
  
  The console output will show evaluation results, which might look like this. Basically, this means that on test images, the model-predicted keypoints are roughly 6 pixels away from ground-truth keypoints.
  ```
  Evaluation results for DLC_Resnet50_dlc_modelJul26shuffle1_snapshot_100-results.csv (pcutoff: 0.01):
  train rmse                        3.97
  train rmse_pcutoff                3.97
  train mAP                        86.50
  train mAR                        89.37
  train rmse_detections             3.85
  train rmse_detections_pcutoff     3.85
  test rmse                         5.89
  test rmse_pcutoff                 5.89
  test mAP                         70.26
  test mAR                         76.16
  test rmse_detections              6.66
  test rmse_detections_pcutoff      6.66
  ```
  
  **(As of October 15, 2024)** If you don’t get these accurate evaluation results (around 6 pixels), then model might not have been trained properly. As of October 15, 2024, DeepLabCut version rc4 seems to have an bug that you have to fix – please refer to my GitHub issue raised with DeepLabCut repo: https://github.com/DeepLabCut/DeepLabCut/issues/2751 DeepLabCut version rc5 (latest version) seems to introduce a fix for this. As the DLC team fixes the code, this note will be updated.

## Inference on new videos – `analyze_videos()`
  Once your model achieved solid accuracies after training, you can proceed to analyze new videos. You should test this with a short video first (maybe 30 seconds long).
  
  To analyze a new video, modify the path to your video and execute this code. Note that the `<video folder>` does not need to be the same as your project folder.
  ```
  config_path = "<project folder>/dlc_model-student-2023-07-26/config.yaml"
  scorername = deeplabcut.analyze_videos(config_path,['<video folder>/0001_vid_30secs.mp4'], videotype='.mp4')
  ```
  **(As of October 15)** If there are errors, please refer to some issues I raised on DeepLabCut GitHub: 
  - Analyze_videos failed for empty frame. https://github.com/DeepLabCut/DeepLabCut/issues/2754
  - Np shape errors led to failure to create annotated videos. https://github.com/DeepLabCut/DeepLabCut/issues/2755
  
  This note will be updated as the DeepLabCut team investigates and fixes these bugs.
  
  This code will create some files, such as:
  ```
  <video folder>/0001_vid_30secsDLC_Resnet50_dlc_modelJul26shuffle1_snapshot_200_full.pickle
  <video folder>/0001_vid_30secsDLC_Resnet50_dlc_modelJul26shuffle1_snapshot_200_meta.pickle
  <video folder>/0001_vid_30secsDLC_Resnet50_dlc_modelJul26shuffle1_snapshot_200.h5
  <video folder>/0001_vid_30secsDLC_Resnet50_dlc_modelJul26shuffle1_snapshot_200_assemblies.pickle
  <video folder>/0001_vid_30secsDLC_Resnet50_dlc_modelJul26shuffle1_snapshot_200_el.pickle
  ```
  The file names are a concatenation of your video name, the model name, the shuffle index and the snapshot index. These files contain the model’s predictions. Inspecting the `.h5` or `.pickle` files, you will see the predictions for each frame. For each frame, the model makes prediction for each animal (up to a maximum number of animals) in that frame. For each animal, the model might predict some body-part (keypoints) of that animal on the image frame. For each keypoint, the model might predict x-coordinate, y-coordinate, and confidence. 
  
  The `_assemblies.pickle` file might contain an additional column for the `affinity` value for the assembled skeleton (“skeleton” meaning the model has grouped a set of keypoints and decided they are likely to belong to the same animal) – the `afinity` value seems to indicate how strongly the model believes in the assembled set of keypoints as a skeleton.
  
  The `_meta.pickle` holds some metadata of the video inference.

## Create videos with body-parts visualized
  Run this code after `.analyze_videos()`:
  ```
  deeplabcut.create_video_with_all_detections(config_path, ['<video folder>/0001_vid_30secs.mp4'], videotype='.mp4')
  ```
  Look for a new video with name that might look like this: `<video folder> /0001_vid_30secsDLC_Resnet50_dlc_modelJul26shuffle1_snapshot_200_full.mp4`. The video should have keypoints that “follow” the fish like this: https://photos.app.goo.gl/Ehx7fAtejxgtvLau5 

## Run `.stitch_tracklets()`
  Run this to associate the tracklets (small tracks identified by the model) into longer tracks that might better help us track the same fish individuals. 
  
  My initial interpretation, based on my inspection of DeepLabCut code, is that the previous `.analyze_videos()` function call uses SORT algorithm to process consecutive frames, make predictions on how a “skeleton” might move next, and associate those predictions with the actual “skeleton” detected in the next frames. This way, the model identified small tracks that belong to the same individual fish.
  
  But it appears that if the individual fish cross paths or occlude one another, those small tracks might be disrupted and are no longer associated with the right individuals. By running a max-flow algorithm, DeepLabCut is able to stitch the disrupted tracks into longer tracks, helping us better track individual fish.
  ```
  config_path = "/home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/config.yaml"
  deeplabcut.stitch_tracklets(config_path,['/home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/temp/0001_vid_30secs.mp4'], videotype='.mp4', shuffle=1, trainingsetindex=0)
  ```
  New files might be created, with names like this:
  ```
  <video folder>/0001_vid_30secsDLC_Resnet50_dlc_modelJul26shuffle1_snapshot_200_el.h5
  ```
## `create_labeled_video()`
  After `.stitch_tracklets()`, DeepLabCut can now generate a new version of the same video that you provided, but that has keypoints “drawn across multiple frames” so that viewers can identify tracks along which the individual fish move – like a “trailing visual effect.” An example is https://photos.app.goo.gl/3bN8xEpQMhXw8ziY7 
  ```
  config_path = "<project folder>/dlc_model-student-2023-07-26/config.yaml"
  
  deeplabcut.create_labeled_video(
      config_path,
      ['<video folder>/0001_vid_30secs.mp4'],
      color_by="individual",
      keypoints_only=False,
      trailpoints=10,
      draw_skeleton=False,
      track_method="ellipse",
  )
  
  ```
  
## Training a transformer re-identification model

In this folder are some add-on scripts for training a transformer re-ID model, which is implemented for DeepLabCut TensorFlow engine but has not been implemented in DeepLabCut v3.0 PyTorch engine (as of November 2024): https://github.com/Human-Augment-Analytics/CichlidBowerTracking/tree/master/cichlid_bower_tracking/transformer_re-ID_for_DLC_pytorch_engine

Please download these two files into a folder: `generate_bodypart_features_file.py` and `transformer_reID_for_pytorch_engine.py`. (See further explanations below.)

Then in the same folder, you can run your own script like below, calling `transformer_reID()` function not as imported from `deeplabcut` but from `transformer_reID_for_pytorch_engine`.

Please note that the `_assemblies.pickle` file must exist in your video folder first (by running `deeplabcut.analyze_videos()`).
```
from transformer_reID_for_pytorch_engine import transformer_reID

config_path = "/home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/config.yaml"
transformer_reID(config_path, ['/home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/temp/0001_vid_30secs.mp4'], n_tracks=10, videotype="mp4")
# n_tracks should be the same as the max number of individuals (here, the default has been set to 10 fish
```
You should see console output like this (simplified to avoid clutter, with run-time warnings removed). Transformer model will be saved to the path as mentioned in console output. Then `deeplabcut.stitch_tracklets()` will be run to automatically stitch tracklets, this time using the “weights” from the transformer re-ID model for stitching.
```
Analyzing videos with <snapshot folder> /snapshot-200.pt
Forward-passing this video to extract features at the backbone: /home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/temp/0001_vid_30secs.mp4
Video metadata: 
  Overall # of frames:    900
  Duration of video [s]:  30.00
  fps:                    30.0
  resolution:             w=1296, h=972

Body-part features from the backbone of DeepLabCut model have been saved to <filename>_bpt_features.pickle.dat

Creating the triplet dataset...
Creating the triplet dataset... DONE

Training transformer re-identification model...
Epoch 10, train acc: 0.58
Epoch 10, test acc 0.52
Epoch 20, train acc: 0.61
Epoch 20, test acc 0.54
…
Epoch 90, train acc: 0.62
Epoch 90, test acc 0.51
Epoch 100, train acc: 0.62
Epoch 100, test acc 0.53

Transformer re-ID checkpoint saved: <model folder>/dlc_transreid_100.pth

Now running deeplabcut.stitch_tracklets()
…
```
### Explanations
#### Recap
To recap, when you have a video to do DeepLabCut inference on by `deeplabcut.analyze_videos()` (as described above). Several files will be created.

Note that the files are automatically named based on your video name and the information abou your model and snapshot (as detected by DeepLabCut from the `config.yaml` file). (Below, `<filename>` will be `<video name>` concatenated with `<DLC scorer name>`. For example, if your `<video name>` is `0001_vid_30secs` and your “DeepLabCut scorer” name is `DLC_Resnet50_dlc_modelJul26shuffle1_snapshot_200`. Then <filename> would be `0001_vid_30secsDLC_Resnet50_dlc_modelJul26shuffle1_snapshot_200`)

When `.analyze_videos()` is run, these files are created:
-	`<filename>_full.pickle`: storing data on the predicted body parts, assembled into individual fish according to DLC’s algorithm
-	`<filename>.h5`: storing the same data in h5 format
-	`<filename>_assemblies.pickle`: storing essentially the same data in slightly different format 
-	`<filename>_meta.pickle`: the metadata of the video inference run

By default,`.analyze_videos()` would automatically call:
-	`convert_detections2tracklets()`, which would create `<filename>_el.pickle`. This file re-organizes the fish assemblies by tracklets that might persist across a number of frames.
-	`stitch_tracklets()`, which would load the files above as input and generate `<filename>_el.h5`, in which tracklets are stitched into longer tracks based on a max-flow algorithm that aims to connect tracklets with some measure of smallest cost from one tracklet to the next
#### `_bpt_features.pickle` file generated in Tensorflow engine
Transformer re-identification training has been implemented in the DLC TensorFlow engine but not yet in the PyTorch engine. 

For some background, for transformer re-ID training, the activations that the DLC model computed at the end of its ResNet backbone for each body-part keypoint are kept. The last convolutional layer of ResNet backbone in DLC has 2048 channels by default, so for each keypoint of the fish, there’s a 2048-dimensional vector for that keypoint to be kept. Each fish has 9 keypoints, so for each fish, DeepLabCut keeps 9 of those 2048-dimensional vectors. Please refer the multi-animal paper cited above for details.

DeepLabCut authors hypothesized that each of those features encoded some visual representation of the area around each keypoint in the original image. Thus, they could perhaps serve as good features for re-identifcation task.

To that end, during the inference run on a new video and besides generating the files mentioned above, DLC TensorFlow engine also extracts the 2048-dimensional features for each detected keypoints and stores in a separate file: `<filename>_bpt_features.pickle` 

(Actually, there are three files: `_bpt_features.pickle.dat`, `_bpt_features.pickle.bak`, and `bpt_features.pickle.dir`- that’s because of the way the `shelves` library handles pickle files. But as a  user of that library, we can just think of them as one pickle file.)

Then when you run `deeplabcut.transformer_reID()` (implemented in the TensorFlow engine, as described in DLC documentation), the code would open this `_bpt_features.pickle` file, load the 2048-dimensional features, generate triplet dataset from them, and save the triplets into `<filename>_triplet_vector.npy`. The code then builds a dataloader, a transformer model, optimizer, etc. as usual to train a re-ID model, based on this `_triplet_vector.npy` file.

#### PyTorch engine

For the PyTorch engine, this transformer re-ID pipeline has not been implemented yet. It’s not known when it’s going to be. But the add-on code above could replicate this pipeline. Most components of the transformer re-ID pipeline in the Tensorflow engine are actually available in the PyTorch engine, such as triplet dataset creation, dataloader, transformer model, traning code, etc.

The missing piece in the PyTorch engine is that during inference, the 2048-dimensional features are not kept and saved into the `_bpt_features.pickle` file. If this file can be generated, then the remaining components of transformer re-ID for Tensorflow engine can be re-used for PyTorch engine.

The add-on code in ` generate_bodypart_features_file.py` above does just that. In order to minimize changes to DeepLabCut code, this file does not insert any extra steps while video inference is running to extract 2048-dimensional ResNet features. 

Instead, the code loads up a new instance of the trained DeepLabCut model, set it to evaluation mode, re-run the video again as a forward pass, but stop after the ResNet backbone layer to extract the necessary features and ultimately save into a `_bpt_features.pickle` file. Once the right file has been created, then the other components can be adapted (triplet dataset creation, model building, training, etc.) almost intact from DeepLabCut’s code.

