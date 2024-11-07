## Training a transformer re-identification model

In this folder are some add-on scripts for training a transformer re-ID model, which is implemented for DeepLabCut TensorFlow engine but has not been implemented in DeepLabCut v3.0 PyTorch engine (as of November 2024).

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

