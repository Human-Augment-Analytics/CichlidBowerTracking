import sys
sys.path.append("/home/hice1/tnguyen868/scratch/dlc_dataset/cichlid-behavior-detection/")

import deeplabcut
import torch
import numpy as np
import os
import pickle
import shelve
from deeplabcut.core import trackingutils
from deeplabcut.refine_training_dataset.stitch import TrackletStitcher
from pathlib import Path
from deeplabcut.pose_tracking_pytorch.tracking_utils.preprocessing import query_feature_by_coord_in_img_space
import cv2
from PIL import Image

np.random.seed(0)

# so that tensors show their shape in debugging
if __name__ == "__main__":
    normal_repr = torch.Tensor.__repr__ 
    torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}" 


# These functions originally comes from DeepLabCut v3 rc4. First credit goes to DeepLabCut: https://github.com/DeepLabCut/DeepLabCut/
# create_tracking_dataset()
# create_triplets_dataset()
# create_train_using_pickle()
# save_train_triplets()

# Secondly, the credit goes to Adam Thomas (Cichlid CV team - summer 2024), who modified these functions to implement transformer re-identification training with cropped images of the animals as extra features. Adam's original version is here: https://github.com/athomas125/cichlid-behavior-detection/tree/main/behavior_detection

# Adam's version was implemented for DeepLabCut's TensorFlow engine. The experiment was replicated by Thuan Nguyen (Cichlid CV team - Fall 2024) on DeepLabCut’s newer PyTorch engine. (The TensorFlow engine is being deprecated.) 

# Please contact these members for any questions.



def save_train_triplets_with_image(feature_fname, triplets, out_name):
    ret_vecs = []

    feature_dict = shelve.open(feature_fname, protocol=pickle.DEFAULT_PROTOCOL)

    nframes = len(feature_dict.keys())

    zfill_width = int(np.ceil(np.log10(nframes)))

    final_dim_size = image_preprocess_weights.transforms().crop_size[0]**2 * OUTPUT_COLOR_CHANNELS #squaring crop size cause square crop
    example = triplets[0][0]
    example_frame = "frame" + str(example[1]).zfill(zfill_width)
    kpt_feature_shape = query_feature_by_coord_in_img_space(
        feature_dict, example_frame, example[0]
    ).flatten().shape[0]
    
    
    # Initialize a memory-mapped array to avoid OOM issues
    ret_vecs_shape = (len(triplets), 3, kpt_feature_shape + final_dim_size)
    ret_vecs = np.memmap(out_name, dtype='float32', mode='w+', shape=ret_vecs_shape)
    
    frame_map = {}

    for i, triplet in enumerate(triplets):
        anchor, pos, neg = triplet[0], triplet[1], triplet[2]

        anchor_coord, anchor_frame = anchor
        pos_coord, pos_frame = pos
        neg_coord, neg_frame = neg

        anchor_frame = "frame" + str(anchor_frame).zfill(zfill_width)
        pos_frame = "frame" + str(pos_frame).zfill(zfill_width)
        neg_frame = "frame" + str(neg_frame).zfill(zfill_width)

        if (
            anchor_frame in feature_dict
            and pos_frame in feature_dict
            and neg_frame in feature_dict
        ):
            # only try to find these features if they are in the dictionary

            anchor_vec = query_feature_by_coord_in_img_space(
                feature_dict, anchor_frame, anchor_coord
            ).flatten()
            pos_vec = query_feature_by_coord_in_img_space(
                feature_dict, pos_frame, pos_coord
            ).flatten()
            neg_vec = query_feature_by_coord_in_img_space(
                feature_dict, neg_frame, neg_coord
            ).flatten()

            ret_vecs[i, :, :kpt_feature_shape] = np.array([anchor_vec, pos_vec, neg_vec])
            triplet_ind = 0
            for coord, frame in [anchor, pos, neg]:
                frame_str = "frame" + str(frame).zfill(zfill_width)
                if frame_str not in frame_map:
                    frame_map[frame_str] = {}
                
                # using this so we have a hashable type (also using a mask here)
                bounding_coords = (max(coord[coord[:,0] >= 0,0]), min(coord[coord[:,0] >= 0,0]), max(coord[coord[:,1] >= 0,1]), min(coord[coord[:,1] >= 0,1]))
                if bounding_coords not in frame_map[frame_str]:
                    frame_map[frame_str][bounding_coords] = []
                frame_map[frame_str][bounding_coords].append((i, triplet_ind))
                triplet_ind += 1

            # ret_vecs.append([anchor_vec, pos_vec, neg_vec])

    # ret_vecs = np.array(ret_vecs)

    # with open(out_name, "wb") as f:
        # np.save(f, ret_vecs)

    import cv2
    cap = cv2.VideoCapture(video_path)
    
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_ind = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_str = "frame" + str(frame_ind).zfill(zfill_width)
        if frame_str in frame_map:
            for key in frame_map[frame_str]:
                frame_crop = frame[max(key[3]-50, 0):min(key[2]+50, height),max(key[1]-50, 0):min(key[0]+50, width),:]
                vit_vec = image_preprocess_func(frame_crop, image_preprocess_weights).flatten()
                for i, ind in frame_map[frame_str][key]:
                    ret_vecs[i, ind, kpt_feature_shape:] = vit_vec
                    ret_vecs.flush() 
        frame_ind += 1
            
    # clear everything to disk
    ret_vecs.flush() 
    # close array
    del ret_vecs


def create_train_using_pickle_with_image(feature_fname, path_to_pickle, out_name, n_triplets=1000):
    from deeplabcut.pose_tracking_pytorch.create_dataset import generate_train_triplets_from_pickle
    triplets = generate_train_triplets_from_pickle(
        path_to_pickle, n_triplets=n_triplets
    )
    save_train_triplets_with_image(feature_fname, triplets, out_name)


def create_triplets_dataset_with_image(
    videos, dlcscorer, track_method, n_triplets=1000, destfolder=None
):
    # 1) reference to video folder and get the proper bpt_feature file for feature table
    # 2) get either the path to gt or the path to track pickle

    for video in videos:
        vname = Path(video).stem
        videofolder = str(Path(video).parents[0])
        if destfolder is None:
            destfolder = videofolder
        feature_fname = os.path.join(
            destfolder, vname + dlcscorer + "_bpt_features.pickle"
        )

        method = trackingutils.TRACK_METHODS[track_method]
        track_file = os.path.join(destfolder, vname + dlcscorer + f"{method}.pickle")
        out_fname = os.path.join(destfolder, vname + dlcscorer + "_triplet_vector.npy")
        create_train_using_pickle_with_image(
            feature_fname, track_file, out_fname, n_triplets=n_triplets
        )


def create_tracking_dataset_with_image(
    config,
    videos,
    track_method,
    videotype="",
    shuffle=1,
    trainingsetindex=0,
    # gputouse=None,
    # save_as_csv=False,
    destfolder=None,
    # batchsize=None,
    # cropping=None,
    # TFGPUinference=True,
    # dynamic=(False, 0.5, 10),
    modelprefix="",
    # robust_nframes=False,
    n_triplets=1000,
):

    # from deeplabcut.pose_tracking_pytorch.create_dataset import create_triplets_dataset
    from deeplabcut.utils.auxfun_multianimal import get_track_method
    from deeplabcut.utils import auxiliaryfunctions
    from deeplabcut.pose_estimation_pytorch.task import Task
    from deeplabcut.pose_estimation_pytorch.apis.utils import (
        get_model_snapshots,
        get_inference_runners,
        get_scorer_name,
        get_scorer_uid,
        list_videos_in_folder,
        parse_snapshot_index_for_analysis,
    )
    from pathlib import Path
    from deeplabcut.core.engine import Engine

    # Read the configuration settings from the provided config file.
    cfg = auxiliaryfunctions.read_config(config)

    # Load the project configuration
    project_path = Path(cfg["project_path"])
    train_fraction = cfg["TrainingFraction"][trainingsetindex]
    model_folder = project_path / auxiliaryfunctions.get_model_folder(
        train_fraction,
        shuffle,
        cfg,
        modelprefix=modelprefix,
        engine=Engine.PYTORCH,
    )
    train_folder = model_folder / "train"

    # Read the inference configuration, load the model
    model_cfg_path = train_folder / Engine.PYTORCH.pose_cfg_name
    model_cfg = auxiliaryfunctions.read_plainconfig(model_cfg_path)
    pose_task = Task(model_cfg["method"])

    pose_cfg_path = model_folder / "test" / "pose_cfg.yaml"
    pose_cfg = auxiliaryfunctions.read_plainconfig(pose_cfg_path)

    snapshot_index = detector_snapshot_index = None
    snapshot_index, detector_snapshot_index = parse_snapshot_index_for_analysis(
        cfg, model_cfg, snapshot_index, detector_snapshot_index,
    )

    snapshot = detector_snapshot = None
    snapshot = get_model_snapshots(snapshot_index, train_folder, pose_task)[0]

    if track_method is None:
        track_method = auxfun_multianimal.get_track_method(cfg, track_method=track_method)

    dlc_scorer = get_scorer_name(
        cfg,
        shuffle,
        train_fraction,
        snapshot_uid=get_scorer_uid(snapshot, detector_snapshot),
        modelprefix=modelprefix,
    )

    from generate_bodypart_features_file import extract_backbone_features_for_transformer_reID_training

    for video in videos:
        
        extract_backbone_features_for_transformer_reID_training(
            config=config,
            video_path=video,
            batch_size=8,
            shuffle=1,
            trainingsetindex=0,
            destfolder=None,
            modelprefix='',
            snapshot_index=None,
            detector_snapshot_index=None,
            device=None,
            transform=None
        )

    print("\nCreating the triplet dataset...\n")
    create_triplets_dataset_with_image(
        videos, dlc_scorer, track_method, n_triplets=1000, destfolder=None
    )
    print("\nCreating the triplet dataset... DONE\n")

    return dlc_scorer


## from Adam: train_dlctransreid_with_image.py
import random

try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Unsupervised identity learning requires PyTorch. Please run `pip install torch`."
    )
import numpy as np
import os
import glob
from pathlib import Path
from deeplabcut.pose_tracking_pytorch.config import cfg
from deeplabcut.pose_tracking_pytorch.datasets import make_dlc_dataloader
from behavior_detection.transformer_training.make_model_dlc_and_image import make_dlc_model
# from behavior_detection.transformer_training.make_model_dlc_and_image import make_dlc_model_just_image
from deeplabcut.pose_tracking_pytorch.solver import make_easy_optimizer
from deeplabcut.pose_tracking_pytorch.solver.scheduler_factory import create_scheduler
from deeplabcut.pose_tracking_pytorch.loss import easy_triplet_loss
from deeplabcut.pose_tracking_pytorch.processor import do_dlc_train


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def split_train_test(npy_list, train_frac, npy_shape, dtype='float32', train_map_path='train_data.mmap', test_map_path='test_data.mmap', seed=42, chunk_size=1024):
    if seed is not None:
        np.random.seed(seed)
        
    print(f"Creating train and test maps at {train_map_path}, {test_map_path}")
    
    total_train_samples = 0
    total_test_samples = 0
    
    print('Calculating total samples for train and test datasets')
    for npy in npy_list:
        mmap = np.memmap(npy, dtype=dtype, mode='r', shape=npy_shape)
        n_samples = mmap.shape[0]
        num_train = int(n_samples * train_frac)
        total_train_samples += num_train
        total_test_samples += n_samples - num_train

    train_shape = (total_train_samples,) + npy_shape[1:]
    test_shape = (total_test_samples,) + npy_shape[1:]
    
    print('Creating memmap files')
    train_mmap = np.memmap(train_map_path, dtype=dtype, mode='w+', shape=train_shape)
    test_mmap = np.memmap(test_map_path, dtype=dtype, mode='w+', shape=test_shape)
    
    train_start_idx = 0
    test_start_idx = 0
    
    print('Writing data to memmap files')
    for npy in npy_list:
        mmap = np.memmap(npy, dtype=dtype, mode='r', shape=npy_shape)
        n_samples = mmap.shape[0]
        indices = np.random.permutation(n_samples)
        num_train = int(n_samples * train_frac)
        
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]
        
        # Write train data in chunks
        for i in range(0, len(train_indices), chunk_size):
            chunk_indices = train_indices[i:i+chunk_size]
            train_end_idx = train_start_idx + len(chunk_indices)
            train_mmap[train_start_idx:train_end_idx] = mmap[chunk_indices]
            train_start_idx = train_end_idx
            
        # Write test data in chunks
        for i in range(0, len(test_indices), chunk_size):
            chunk_indices = test_indices[i:i+chunk_size]
            test_end_idx = test_start_idx + len(chunk_indices)
            test_mmap[test_start_idx:test_end_idx] = mmap[chunk_indices]
            test_start_idx = test_end_idx

    # Flush to ensure data is written to disk
    train_mmap.flush()
    test_mmap.flush()
    
    return train_mmap, test_mmap

def train_tracking_transformer_with_image(
    path_config_file,
    dlcscorer,
    videos,
    videotype="",
    train_frac=0.8,
    modelprefix="",
    n_triplets=1000,
    train_epochs=100,
    batch_size=64,
    ckpt_folder="",
    destfolder=None,
    feature_dim=2048,
    num_kpts=9,
    feature_extractor=None,
    just_image = False,
    feature_extractor_in_dim=None,
    feature_extractor_out_dim=None,
    npy_list_filenames = None,
    train_map_path = 'train_data.mmap',
    test_map_path = 'test_data.mmap'
):
    npy_list = []
    if npy_list_filenames is None:
        from deeplabcut.utils import auxiliaryfunctions
        videos = auxiliaryfunctions.get_list_of_videos(videos, videotype)
        for video in videos:
            videofolder = str(Path(video).parents[0])
            if destfolder is None:
                destfolder = videofolder
            video_name = Path(video).stem
            # video_name = '.'.join(video.split("/")[-1].split(".")[:-1])
            files = glob.glob(os.path.join(destfolder, video_name + dlcscorer + "*vector.npy"))

            # assuming there is only one match
            npy_list.append(files[0])
    else:
        if type(npy_list_filenames) == list:
            for f in npy_list_filenames:
                npy_list.append(f)
        else:
            # assuming just a single filename
            npy_list.append(npy_list_filenames)

    print('getting lists')
    npy_shape = (n_triplets, 3, num_kpts*feature_dim + np.prod(feature_extractor_in_dim))
    train_list, test_list = split_train_test(npy_list, train_frac, npy_shape, train_map_path=train_map_path, test_map_path=test_map_path)
    print("got lists")

    train_loader, val_loader = make_dlc_dataloader(train_list, test_list, batch_size)
    print("got dataloaders")
    
    if just_image:
        model = make_dlc_model_just_image(cfg, 
                                        feature_dim,
                                        num_kpts,
                                        feature_extractor,
                                        feature_extractor_in_dim,
                                        feature_extractor_out_dim)
        print("got dlc model just images")
    else:
        # make my own model factory
        model = make_dlc_model(cfg, 
                            feature_dim,
                            num_kpts,
                            feature_extractor,
                            feature_extractor_in_dim,
                            feature_extractor_out_dim)
        print("got dlc model that includes features")
    
    # make my own loss factory
    triplet_loss = easy_triplet_loss()

    optimizer = make_easy_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)
    print('got optimizer and scheduler')
    
    num_query = 1

    cfg["log_period"] = 50
    cfg["checkpoint_period"] = 10

    do_dlc_train(
        cfg,
        model,
        triplet_loss,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        num_kpts,
        feature_dim,
        num_query,
        total_epochs=train_epochs,
        ckpt_folder=ckpt_folder,
    )


# This function originally comes from DeepLabCut v3 rc4. I modified it slightly to implement transformer re-identification training for the PyTorch engine.
def transformer_reID_with_image(
    config,
    videos,
    videotype="",
    shuffle=1,
    trainingsetindex=0,
    batch_size=8,
    track_method="ellipse",
    n_tracks=None,
    n_triplets=1000,
    train_epochs=100,
    train_frac=0.8,
    modelprefix="",
    destfolder=None,
):
    """
    Enables tracking with transformer.

    Substeps include:

    - Mines triplets from tracklets in videos (from another tracker)
    - These triplets are later used to tran a transformer with triplet loss
    - The transformer derived appearance similarity is then used as a stitching loss when tracklets are
    stitched during tracking.

    Outputs: The tracklet file is saved in the same folder where the non-transformer tracklet file is stored.

    Parameters
    ----------
    config: string
        Full path of the config.yaml file as a string.

    videos: list
        A list of strings containing the full paths to videos for analysis or a path to the directory, where all the videos with same extension are stored.

    videotype: string, optional
        Checks for the extension of the video in case the input to the video is a directory.\n Only videos with this extension are analyzed.
        If left unspecified, videos with common extensions ('avi', 'mp4', 'mov', 'mpeg', 'mkv') are kept.

    shuffle : int, optional
        which shuffle to use

    trainingsetindex : int. optional
        which training fraction to use, identified by its index

    track_method: str, optional
        track method from which tracklets are sampled

    n_tracks: int
        number of tracks to be formed in the videos.
        TODO: handling videos with different number of tracks

    n_triplets: (optional) int
        number of triplets to be mined from the videos

    train_epochs: (optional), int
        number of epochs to train the transformer

    train_frac: (optional), fraction
        fraction of triplets used for training/testing of the transformer

    Examples
    --------

    Training model for one video based on ellipse-tracker derived tracklets
    >>> deeplabcut.transformer_reID(path_config_file,[''/home/alex/video.mp4'],track_method="ellipse")

    --------

    """
    import deeplabcut
    import os
    from deeplabcut.utils import auxiliaryfunctions

    # calling create_tracking_dataset_with_image, train_tracking_transformer, stitch_tracklets

    cfg = auxiliaryfunctions.read_config(config)

    DLCscorer, _ = deeplabcut.utils.auxiliaryfunctions.GetScorerName(
        cfg,
        shuffle=shuffle,
        trainFraction=cfg["TrainingFraction"][trainingsetindex],
        modelprefix=modelprefix,
    )

    create_tracking_dataset_with_image(
        config,
        videos,
        track_method,
        videotype=videotype,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        n_triplets=n_triplets,
        destfolder=destfolder,
    )

    (
        trainposeconfigfile,
        testposeconfigfile,
        snapshotfolder,
    ) = deeplabcut.return_train_network_path(
        config,
        shuffle=shuffle,
        modelprefix=modelprefix,
        trainingsetindex=trainingsetindex,
    )

    print("Training transformer re-identification model...")
    # train_tracking_transformer_with_image(
    #     config,
    #     DLCscorer,
    #     videos,
    #     videotype=videotype,
    #     train_frac=train_frac,
    #     modelprefix=modelprefix,
    #     train_epochs=train_epochs,
    #     ckpt_folder=snapshotfolder,
    #     destfolder=destfolder,
    # )
    train_tracking_transformer_with_image(
        config,
        DLCscorer,
        videos,
        videotype="",
        train_frac=0.95,
        modelprefix="",
        n_triplets=1000,
        train_epochs=100,
        batch_size=batch_size,
        ckpt_folder=destfolder,
        destfolder=None,
        feature_dim=FEATURE_DIM,
        num_kpts=NUM_KPTS,
        feature_extractor=feature_extractor,
        just_image = False,
        feature_extractor_in_dim=in_shape,
        feature_extractor_out_dim=out_shape,
        npy_list_filenames = training_data_filename,
        train_map_path = train_map_path,
        test_map_path = test_map_path
    )
    # train_tracking_transformer_with_image(
    #     config,
    #     DLCscorer,
    #     video_path,
    #     videotype="",
    #     train_frac=train_frac,
    #     modelprefix=modelprefix,
    #     n_triplets=n_triplets,
    #     train_epochs=train_epochs,
    #     batch_size=n,
    #     ckpt_folder=snapshotfolder,
    #     feature_dim=FEATURE_DIM,
    #     num_kpts=NUM_KPTS,
    #     feature_extractor=feature_extractor,
    #     just_image=True,
    #     feature_extractor_in_dim=in_shape,
    #     feature_extractor_out_dim=out_shape,
    #     npy_list_filenames=training_data_filename,
    #     train_map_path = train_map_path,
    #     test_map_path = test_map_path
    # )

    transformer_checkpoint = os.path.join(
        snapshotfolder, f"dlc_transreid_{train_epochs}.pth"
    )

    if not os.path.exists(transformer_checkpoint):
        raise FileNotFoundError(f"checkpoint {transformer_checkpoint} not found")
    else:
        print(f"\nTransformer re-ID checkpoint saved: {transformer_checkpoint}\n")
    
    print(f"\nNow running deeplabcut.stitch_tracklets()\n")
    deeplabcut.stitch_tracklets(
        config,
        videos,
        videotype=videotype,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        track_method=track_method,
        modelprefix=modelprefix,
        n_tracks=n_tracks,
        transformer_checkpoint=transformer_checkpoint,
        destfolder=destfolder,
    )



def ViT_H_14_Preprocess(frame, weights):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Apply the ViT transforms
    vit_input = weights.transforms()(pil_image)
    
    return vit_input


from torchvision.models import vit_h_14, ViT_H_14_Weights
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
from PIL import Image
import os
from behavior_detection.transformer_training.ViTFeatureExtractor import ViTFeatureExtractor

def get_vit_14(device):
    weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
    pretrained_model = vit_h_14(weights=weights)

    feature_extractor = ViTFeatureExtractor(
        image_size=pretrained_model.image_size,
        patch_size=pretrained_model.patch_size,
        num_layers=len(pretrained_model.encoder.layers),
        num_heads=pretrained_model.encoder.layers[0].num_heads,
        hidden_dim=pretrained_model.hidden_dim,
        mlp_dim=pretrained_model.mlp_dim
    )

    feature_extractor.load_state_dict(pretrained_model.state_dict(), strict=False)

    feature_extractor = feature_extractor.to(device)
    
    img_channels = 3 
    in_shape = torch.Size([img_channels, pretrained_model.image_size, pretrained_model.image_size])
    out_shape = pretrained_model.hidden_dim
    return feature_extractor, in_shape, out_shape


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")

    from torchvision.models import ViT_H_14_Weights
    func = ViT_H_14_Preprocess
    weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
    
    image_preprocess_func = func
    image_preprocess_weights = weights
    OUTPUT_COLOR_CHANNELS = 3

    FEATURE_DIM = 2048
    NUM_KPTS = 9
    feature_extractor, in_shape, out_shape = get_vit_14(device)
    training_data_filename = "/home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/temp4/0001_vid_test4DLC_Resnet50_dlc_modelJul26shuffle1_snapshot_200_triplet_vector.npy"
    train_map_path = "/home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/temp4/train_map.mmap"
    test_map_path = "/home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/temp4/test_map.mmap"

    config_path = "/home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/config.yaml"
    video_path = '/home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/temp4/0001_vid_test4.mp4'
    transformer_reID_with_image(config_path, ['/home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/temp4/0001_vid_test4.mp4'], n_tracks=10, videotype="mp4", batch_size=4, train_epochs=50, destfolder="/home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/temp4")
