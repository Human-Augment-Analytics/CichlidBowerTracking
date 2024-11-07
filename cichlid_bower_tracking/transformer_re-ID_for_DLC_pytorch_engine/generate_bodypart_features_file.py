from __future__ import annotations

import deeplabcut
import torch

from deeplabcut.pose_estimation_pytorch.apis.utils import (
    get_model_snapshots,
    get_scorer_name,
    get_scorer_uid,
    parse_snapshot_index_for_analysis,
)
from deeplabcut.utils import auxfun_multianimal, auxiliaryfunctions, VideoReader

from deeplabcut.pose_estimation_pytorch.data.preprocessor import (
    build_bottom_up_preprocessor,
    # build_top_down_preprocessor,
)
from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms
from deeplabcut.pose_estimation_pytorch.models import PoseModel

# from deeplabcut.pose_estimation_pytorch.runners.snapshots import (
#     Snapshot,
#     TorchSnapshotManager,
# )
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.utils import resolve_device

from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import VideoIterator

import copy
import logging
import pickle
# import time
from pathlib import Path
# from typing import Any

import albumentations as A
import numpy as np
# import pandas as pd
from tqdm import tqdm

from deeplabcut.core.engine import Engine
# from deeplabcut.pose_estimation_pytorch.apis.convert_detections_to_tracklets import (
#     convert_detections2tracklets,
# )
# from deeplabcut.pose_estimation_pytorch.post_processing.identity import assign_identity
# from deeplabcut.pose_estimation_pytorch.runners import InferenceRunner
# from deeplabcut.refine_training_dataset.stitch import stitch_tracklets

# import torch.nn as nn

import deeplabcut.pose_estimation_pytorch.modelzoo.utils as modelzoo_utils
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.pose_estimation_pytorch.models.backbones import BACKBONES, BaseBackbone
from deeplabcut.pose_estimation_pytorch.models.criterions import (
    CRITERIONS,
    LOSS_AGGREGATORS,
)
from deeplabcut.pose_estimation_pytorch.models.heads import BaseHead, HEADS
from deeplabcut.pose_estimation_pytorch.models.necks import BaseNeck, NECKS
from deeplabcut.pose_estimation_pytorch.models.predictors import PREDICTORS
from deeplabcut.pose_estimation_pytorch.models.target_generators import (
    TARGET_GENERATORS,
)

class PoseModelReturningBackboneFeatures(PoseModel):
    """
    PoseModel variant that returns the last layer of the backbone features
    along with the outputs of the model heads.
    """
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        """
        Forward pass of the PoseModelReturningBackboneFeatures.

        Args:
            x: Input images

        Returns:
            A tuple containing:
                - The last layer of the backbone features (torch.Tensor)
                - The outputs of head groups (dict[str, dict[str, torch.Tensor]])
        """
        if x.dim() == 3:
            x = x[None, :]  # Ensure input has batch dimension

        # Get backbone features
        features = self.backbone(x)

        # Pass features through neck if available
        if self.neck:
            features = self.neck(features)

        # Get head outputs
        outputs = {}
        for head_name, head in self.heads.items():
            outputs[head_name] = head(features)
        
        # Return both backbone features and head outputs
        return features, outputs

    @staticmethod
    def build(
        cfg: dict,
        weight_init: None | WeightInitialization = None,
        pretrained_backbone: bool = False,
    ):
        """
        Args:
            cfg: The configuration of the model to build.
            weight_init: How model weights should be initialized. If None, ImageNet
                pre-trained backbone weights are loaded from Timm.
            pretrained_backbone: Whether to load an ImageNet-pretrained weights for
                the backbone. This should only be set to True when building a model
                which will be trained on a transfer learning task.

        Returns:
            the built pose model
        """
        cfg["backbone"]["pretrained"] = pretrained_backbone
        backbone = BACKBONES.build(dict(cfg["backbone"]))

        neck = None
        if cfg.get("neck"):
            neck = NECKS.build(dict(cfg["neck"]))

        heads = {}
        for name, head_cfg in cfg["heads"].items():
            head_cfg = copy.deepcopy(head_cfg)
            if "type" in head_cfg["criterion"]:
                head_cfg["criterion"] = CRITERIONS.build(head_cfg["criterion"])
            else:
                weights = {}
                criterions = {}
                for loss_name, criterion_cfg in head_cfg["criterion"].items():
                    weights[loss_name] = criterion_cfg.get("weight", 1.0)
                    criterion_cfg = {
                        k: v for k, v in criterion_cfg.items() if k != "weight"
                    }
                    criterions[loss_name] = CRITERIONS.build(criterion_cfg)

                aggregator_cfg = {"type": "WeightedLossAggregator", "weights": weights}
                head_cfg["aggregator"] = LOSS_AGGREGATORS.build(aggregator_cfg)
                head_cfg["criterion"] = criterions

            head_cfg["target_generator"] = TARGET_GENERATORS.build(
                head_cfg["target_generator"]
            )
            head_cfg["predictor"] = PREDICTORS.build(head_cfg["predictor"])
            heads[name] = HEADS.build(head_cfg)

        model = PoseModelReturningBackboneFeatures(cfg=cfg, backbone=backbone, neck=neck, heads=heads)

        if weight_init is not None:
            logging.info(f"Loading pretrained model weights: {weight_init}")

            # TODO: Should we specify the pose_model_type in WeightInitialization?
            backbone_name = cfg["backbone"]["model_name"]
            pose_model_type = modelzoo_utils.get_pose_model_type(backbone_name)

            # load pretrained weights
            if weight_init.customized_pose_checkpoint is None:
                _, _, snapshot_path, _ = modelzoo_utils.get_config_model_paths(
                    project_name=weight_init.dataset,
                    pose_model_type=pose_model_type,
                )
            else:
                snapshot_path = weight_init.customized_pose_checkpoint

            logging.info(f"The pose model is loading from {snapshot_path}")
            snapshot = torch.load(snapshot_path, map_location="cpu")
            state_dict = snapshot["model"]

            # load backbone state dict
            model.backbone.load_state_dict(filter_state_dict(state_dict, "backbone"))

            # if there's a neck, load state dict
            if model.neck is not None:
                model.neck.load_state_dict(filter_state_dict(state_dict, "neck"))

            # load head state dicts
            if weight_init.with_decoder:
                all_head_state_dicts = filter_state_dict(state_dict, "heads")
                conversion_tensor = torch.from_numpy(weight_init.conversion_array)
                for name, head in model.heads.items():
                    head_state_dict = filter_state_dict(all_head_state_dicts, name)

                    # requires WeightConversionMixin
                    if not weight_init.memory_replay:
                        head_state_dict = head.convert_weights(
                            state_dict=head_state_dict,
                            module_prefix="",
                            conversion=conversion_tensor,
                        )

                    head.load_state_dict(head_state_dict)

        return model



def _forward_pass_batch(frame_batch, model, pose_preprocessor):
    batch_contexts = []
    preprocessed_batch: torch.Tensor = None
    for each_frame in frame_batch:
        # Preprocessing
        context = {}
        preprocessed_frame, context = pose_preprocessor(each_frame, context)
        if preprocessed_batch is None:
            preprocessed_batch = preprocessed_frame
        else:
            preprocessed_batch = torch.cat([preprocessed_batch, preprocessed_frame], dim=0)
        batch_contexts.append(context)

    # Forward pass
    with torch.no_grad():
        features_batch, _ = model(preprocessed_batch.to(next(model.parameters()).device))
    return features_batch, _

def _get_features_dict(raw_coords, features, stride):
    from deeplabcut.pose_tracking_pytorch import (
        load_features_from_coord,
        convert_coord_from_img_space_to_feature_space,
    )

    coords_img_space = np.array(
        [coord[:, :2] for coord in raw_coords]
    )  # only first two columns are useful

    coords_feature_space = convert_coord_from_img_space_to_feature_space(
        coords_img_space,
        stride,
    )

    bpt_features = load_features_from_coord(
        features.astype(np.float16), coords_feature_space
    )
    return {"features": bpt_features, "coordinates": coords_img_space}

def _extract_features_for_batch(frame_indices_of_batch, features_batch, feature_dict, assemblies, stride, strwidth):
    for idx_in_batch, idx in enumerate(frame_indices_of_batch): 
        raw_coords = assemblies.get(idx)
        current_frame_features = _get_features_dict(
            raw_coords, 
            features_batch[idx_in_batch].permute(1, 2, 0).detach().cpu().numpy().astype(np.float16), 
            stride)
        fname = "frame" + str(idx).zfill(strwidth)
        feature_dict[fname] = current_frame_features


def extract_backbone_features_for_transformer_reID_training(
    config, video_path, batch_size=8, shuffle=1, trainingsetindex=0, 
    destfolder=None, modelprefix='', snapshot_index=None, 
    detector_snapshot_index=None, device=None, transform=None
):
    # Loading model configuration
    # Create the output folder
    from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import _validate_destfolder
    _validate_destfolder(destfolder)

    # Load the project configuration
    cfg = auxiliaryfunctions.read_config(config)
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

    snapshot_index, detector_snapshot_index = parse_snapshot_index_for_analysis(
        cfg, model_cfg, snapshot_index, detector_snapshot_index,
    )

    # Get general project parameters
    bodyparts = model_cfg["metadata"]["bodyparts"]
    unique_bodyparts = model_cfg["metadata"]["unique_bodyparts"]
    individuals = model_cfg["metadata"]["individuals"]
    with_identity = model_cfg["metadata"]["with_identity"]
    max_num_animals = len(individuals)

    if device is not None:
        model_cfg["device"] = device

    if batch_size is None:
        batch_size = cfg.get("batch_size", 1)

    snapshot = get_model_snapshots(snapshot_index, train_folder, pose_task)[0]
    print(f"Analyzing videos with {snapshot.path}")
    detector_path, detector_snapshot = None, None

    assert pose_task == Task.BOTTOM_UP, "This function is only implemented for TASK.BOTTOM_UP"

    dlc_scorer = get_scorer_name(
        cfg,
        shuffle,
        train_fraction,
        snapshot_uid=get_scorer_uid(snapshot, detector_snapshot),
        modelprefix=modelprefix,
    )

    model_config = model_cfg
    pose_task = Task(model_config["method"])
    if device is None:
        device = resolve_device(model_config)

    if transform is None:
        transform = build_transforms(model_config["data"]["inference"])

    detector_runner = None
    if pose_task == Task.BOTTOM_UP:
        pose_preprocessor = build_bottom_up_preprocessor(
            color_mode=model_config["data"]["colormode"],
            transform=transform,
        )
    else:
        pass
   
    # Build the model
    model = PoseModelReturningBackboneFeatures.build(model_config["model"])
    optimizer = None

    loaded_snapshot = torch.load(snapshot.path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in loaded_snapshot['model'].items()}  # to prevent an error
    model.load_state_dict(new_state_dict)
    if optimizer is not None and 'optimizer' in snapshot:
        optimizer.load_state_dict(snapshot["optimizer"])


    # Load the video
    video = VideoIterator(str(video_path))
    n_frames = video.get_n_frames()
    vid_w, vid_h = video.dimensions
    print(f"Forward-passing this video to extract features at the backbone: {video_path}")
    print(
        f"Video metadata: \n"
        f"  Overall # of frames:    {n_frames}\n"
        f"  Duration of video [s]:  {n_frames / max(1, video.fps):.2f}\n"
        f"  fps:                    {video.fps}\n"
        f"  resolution:             w={vid_w}, h={vid_h}\n"
    )

    model.to(device)
    model.eval()

    # Load the assemblies
    import os
    # Remove the ".mp4" extension from the video file name
    base_name = os.path.splitext(video_path)[0]
    # Add the DLC scorer and "_assemblies.pickle" to form the desired path
    assembly_path = Path(f"{base_name}{dlc_scorer}_assemblies.pickle")

    if not assembly_path.exists():
        raise FileNotFoundError(
            f"Could not find the assembles file {ass_filename}."
            f" Assemblies must be created by the model (please refer to deeplabcut.analyze_videos()) before "
            "running this function!"
        )

    assemblies = auxiliaryfunctions.read_pickle(assembly_path)

    bpt_feature_pickle_path = assembly_path.with_name(assembly_path.stem.replace("_assemblies", "_bpt_features") + ".pickle")

    if os.path.exists(bpt_feature_pickle_path.with_suffix(".pickle.dat")):
        print(
            f"A _bpt_features.pickle already exists at this path, so the extraction function will not generate a new features file:\n {bpt_feature_pickle_path}\n"
            "If you want to generate a new _bpt_features.pickle file, please delete the old _bpt_features.pickle.dat, .bak and .dir files"
        )
        return

    import shelve
    feature_dict = shelve.open(
        str(bpt_feature_pickle_path), 
        writeback=True, 
        protocol=pickle.DEFAULT_PROTOCOL, 
        flag='n') 

    frame_batch = []
    frame_indices_of_batch = []
    contexts = []
    strwidth = int(np.ceil(np.log10(len(video))))  # width for strings
    # feature_dict = {}
    stride = model_cfg['model']['heads']['bodypart']['heatmap_config']['strides'][0]

    for frame_index, image in enumerate(tqdm(video)):
        
        frame_batch.append(image)
        frame_indices_of_batch.append(frame_index)

        if len(frame_batch) == batch_size:
            features_batch, _ = _forward_pass_batch(frame_batch, model, pose_preprocessor)

            # Extract feature
            _extract_features_for_batch(frame_indices_of_batch, features_batch, feature_dict, assemblies, stride, strwidth)

            frame_batch = []
            frame_indices_of_batch = []

    if frame_batch:
        # process the last incomplete batch
        features_batch, _ = _forward_pass_batch(frame_batch, model, pose_preprocessor)
        _extract_features_for_batch(frame_indices_of_batch, features_batch, feature_dict, assemblies, stride, strwidth)


    feature_dict.close()
    print(f"Body-part features from the backbone of DeepLabCut model have been saved to {str(bpt_feature_pickle_path.with_suffix('.pickle.dat'))}\n")



# if __name__ == "__main__":

#     # so that tensors show their shape in debugging
#     normal_repr = torch.Tensor.__repr__ 
#     torch.Tensor.__repr__ = lambda self: f"{self.shape}_{normal_repr(self)}" 

#     extract_backbone_features_for_transformer_reID_training(
#         config="/home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/dlc_model-student-2023-07-26/config.yaml",
#         video_path='/home/hice1/tnguyen868/scratch/dlc_dataset/dlc_test1/temp/0001_vid_30secs.mp4',
#         batch_size=8,
#         shuffle=1,
#         trainingsetindex=0,
#         destfolder=None,
#         modelprefix='',
#         snapshot_index=None,
#         detector_snapshot_index=None,
#         device=None,
#         transform=None
#     )
