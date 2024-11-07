
import deeplabcut
import torch
from generate_bodypart_features_file import extract_backbone_features_for_transformer_reID_training

# so that tensors show their shape in debugging
if __name__ == "__main__":
    normal_repr = torch.Tensor.__repr__ 
    torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}" 


def create_tracking_dataset(
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

    from deeplabcut.pose_tracking_pytorch.create_dataset import create_triplets_dataset
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
    create_triplets_dataset(
        videos, dlc_scorer, track_method, n_triplets=1000, destfolder=None
    )
    print("\nCreating the triplet dataset... DONE\n")

    return dlc_scorer


# This function originally comes from DeepLabCut v3 rc4. I modified it slightly to implement transformer re-identification training for the PyTorch engine.
def transformer_reID(
    config,
    videos,
    videotype="",
    shuffle=1,
    trainingsetindex=0,
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

    # calling create_tracking_dataset, train_tracking_transformer, stitch_tracklets

    cfg = auxiliaryfunctions.read_config(config)

    DLCscorer, _ = deeplabcut.utils.auxiliaryfunctions.GetScorerName(
        cfg,
        shuffle=shuffle,
        trainFraction=cfg["TrainingFraction"][trainingsetindex],
        modelprefix=modelprefix,
    )

    create_tracking_dataset(
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
    deeplabcut.pose_tracking_pytorch.train_tracking_transformer(
        config,
        DLCscorer,
        videos,
        videotype=videotype,
        train_frac=train_frac,
        modelprefix=modelprefix,
        train_epochs=train_epochs,
        ckpt_folder=snapshotfolder,
        destfolder=destfolder,
    )

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


