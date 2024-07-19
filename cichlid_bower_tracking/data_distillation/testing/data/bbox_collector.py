from typing import Dict

import PIL.Image
from torchvision.transforms.functional import resize, InterpolationMode
from torchvision.io import read_video 
# from torchvision.utils import save_image
# from torchvision.transforms.functional import to_tensor
import torch

import numpy as np
import pandas as pd
import PIL
# import cv2

import math, os

np.random.seed(0)

# from "sort_detections.py"...
IMG_W = 1296
IMG_H = 972

# hyperparams...
# RESZ_DIM = 100

class BBoxCollector:
    def __init__(self, clip_file: str, detections_file: str, bboxes_dir: str, clip_index: int, starting_frame_index: int, dim=128, sqr_bboxes=True, debug=False):
        '''
        Create and initialize an instance of the BBoxCollector class.

        Inputs:
            clip_file: a string value representing the filepath of the video clip which will be processed.
            detections_file: a path to the file containing the detection data generated by YOLO.
            bboxes_dir: a path to the directory in which the transformed bbox images should be saved.
            clip_index: an int representing a clip number, solely used for naming saved BBox images.
            starting_frame_index: an int representing the index from the larger video at which the first frame of the clip is located.
            dim: an integer value indicating the dimension to be used as each resized bbox's width and height; defaults to 128.
            sqr_bboxes: a Boolean indicating if bboxes should be resized to a square shape; defaults to True.
            debug: a Boolean indicating if the BBoxCollector should be run in debug mode.
        '''
        
        self.__version__ = '0.6.0'
        self.clip_file = clip_file
        self.detections_file = detections_file
        self.bboxes_dir = bboxes_dir
        self.clip_index = clip_index
        self.starting_frame_idx = starting_frame_index
        self.dim = dim
        self.sqr_bboxes = sqr_bboxes
        self.debug = debug

        # list to store each collected bbox, organized by frame
        self.bboxes = dict()

    def _get_sqr_bbox(self, frame: torch.Tensor, x_center: int, y_center: int, width: int, height: int) -> torch.Tensor:
        '''
        Carves a PyTorch Tensor representing the bounding box centered at (x_center, y_center) with dimensions (width, height) out of the passed
        frame Tensor. The carved out bbox is slightly larger than necessary to ensure the entire fish body remains in the image after rotation.
        Based on Bree's cropping method, which she sent to me via email.

        Inputs:
            frame: a PyTorch Tensor representing a video frame which contains the bbox to be carved out.
            x_center: the x-coordinate of the bbox's center, as calculated during YOLO.
            y_center: the y-coordinate of the bbox's upper-left corner, as calculated during YOLO.
            width: the width of the bbox to be saved, as calculated during YOLO.
            height: the width of the bbox to be saved, as calculated during YOLO.

        Returns: a PyTorch Tensor representing the bbox carved out of the passed frame Tensor.
        '''
        
        # store the longest dimension length like in Bree's cropping method
        max_dim = (max(abs(int(width)) + 1, abs(int(height)) + 1))
        # print(f'dim: {max_dim}')

        # get bbox coordinates based on Bree's cropping method (PyTorch instead of openCV)
        x_lo = int(max(0, x_center - 0.5 * max_dim))
        y_lo = int(max(0, y_center - 0.5 * max_dim))

        x_hi = int(max(0, x_center + 0.5 * max_dim))
        y_hi = int(max(0, y_center + 0.5 * max_dim))

        # print(f'(x_lo, y_lo): ({x_lo}, {y_lo})')
        # print(f'(x_hi, y_hi): ({x_hi}, {y_hi})')
        
        # get and return bbox by slicing passed frame
        bbox = frame[:, x_lo:x_hi + 1, y_lo:y_hi + 1]# .detach().numpy()
        # print(f'bbox shape: {bbox.shape}')

        # img = PIL.Image.fromarray(bbox.transpose((2, 1, 0)))
        # img.save(os.path.join(self.bboxes_dir, 'test.png'))

        # bbox = torch.tensor(bbox, dtype=torch.uint8)
        
        return bbox

    def _get_bbox(self, frame: torch.Tensor, x_center: int, y_center: int, width: int, height: int) -> torch.Tensor:
        '''
        Carves out a bounding box in its original (non-square) shape and size.

        Inputs:
            frame: a PyTorch Tensor representing the frame containing the bbox.
            x_center: the x-coordinate of the bbox's center.
            y_center: the y-coordinate of the bbox's center.
            width: the width of the bbox.
            height: the height of the bbox.

        Returns:
            bbox: the bbox with center at (x_center, y_center) and shape (num_channels, width, height).
        '''

        x_lo = int(x_center - (width // 2))
        y_lo = int(y_center - (height // 2))

        x_hi = int(x_center + (width // 2))
        y_hi = int(y_center + (height // 2))

        bbox = frame[:, x_lo:x_hi + 1, y_lo:y_hi + 1]
        
        return bbox

    # def _rotate_bbox(self, bbox: torch.Tensor, x_dot: float, y_dot: float, mode_str='bilinear') -> torch.Tensor:
    #     '''
    #     Rotates the passed in bbox PyTorch Tensor based on the fish's x-dimensional and y-dimensional velocities, as determined by the
    #     Kalman Filter during SORT. Inherently assumes that cichlids only swim in the direction which they are facing with their heads.

    #     Inputs:
    #         bbox: a PyTorch Tensor representing a bounding box, as carved out of a video frame by the self._get_box function; has shape (C, H, W).
    #         x_dot: the x-dimensional velocity as calculated by the Kalman Filter during SORT; listed as u_dot in the tracks data file.
    #         y_dot: the y-dimensional velocoty as calculated by the Kalman Filter during SORT; listed as v_dot in the tracks data file.
    #         mode_str: the interpolation mode to be used during rotation; must be one of {'nearest', 'nearest_exact', 'bilinear'}, but defaults to 'bilinear'.

    #     Returns: A PyTorch Tensor representing the rotated bbox.
    #     '''

    #     # use trig to determine the angle of rotation (convert to degrees)
    #     theta = -math.atan2(y_dot, x_dot) * (180.0 / math.pi)

    #     # define the interpolation mode
    #     if mode_str == 'nearest':
    #         mode = InterpolationMode.NEAREST
    #     elif mode_str == 'nearest_exact':
    #         mode = InterpolationMode.NEAREST_EXACT
    #     else:
    #         mode = InterpolationMode.BILINEAR

    #     # perform rotation and return the resulting bbox
    #     rot_bbox = rotate(bbox, theta, mode)

    #     return rot_bbox
    
    def _resize_bbox(self, bbox: torch.Tensor, mode_str='nearest') -> torch.Tensor:
        '''
        Resizes the passed bbox PyTorch Tensor to have shape (dim, dim) using the value passed as an instance variable. 

        Inputs:
            bbox: a PyTorch Tensor representing a bounding box, as carved out of a video frame by the self._get_box function; has shape (C, H, W).
            mode_str: the interpolation mode to be used during rotation; must be one of {'nearest', 'nearest_exact', 'bilinear'}, but defaults to 'bilinear'.

        Returns: a PyTorch Tensor representing the resized bbox image.            
        '''

        # define the interpolation mode
        if mode_str == 'nearest':
            mode = InterpolationMode.NEAREST
        elif mode_str == 'nearest_exact':
            mode = InterpolationMode.NEAREST_EXACT
        else:
            mode = InterpolationMode.BILINEAR

        # perform the resize and return the resulting bbox
        resz_bbox = resize(bbox, (self.dim, self.dim), mode)

        return resz_bbox

    def _save_bbox(self, frame_idx: int, frame: torch.Tensor, x_center: int, y_center: int, width: int, height: int, mode_str='bilinear') -> None:
        '''
        Saves the bounding box in the passed video frame with center (x_center, y_center), and dimensions (width, height) to the bboxes 
        dictionary.

        Inputs:
            frame_idx: an integer indicating the frame number from which the bbox will be collected.
            frame: PyTorch Tensor of shape (C, H, W) containing a specific video frame.
            x_center: the x-coordinate of the bbox's center, as calculated during YOLO.
            y_center: the y-coordinate of the bbox's upper-left corner, as calculated during YOLO.
            width: the width of the bbox to be saved, as calculated during YOLO.
            height: the width of the bbox to be saved, as calculated during YOLO.
            mode_str: the interpolation mode to be used during bbox transformation; must be one of {'nearest', 'nearest_exact', 'bilinear'}, but defaults to 'bilinear'.
        '''

        if self.sqr_bboxes:
            # pass frame through bbox extraction and transformation pipeline
            bbox = self._get_sqr_bbox(frame, x_center, y_center, width, height)
            # rot_bbox = self._rotate_bbox(bbox, x_dot, y_dot, mode_str)
            bbox = self._resize_bbox(bbox, mode_str).detach().numpy()
        else:
            bbox = self._get_bbox(frame, x_center, y_center, width, height).detach().numpy()
        
        # add bbox to dictionary of collected bboxes and return True to indicate success
        if frame_idx in list(self.bboxes.keys()):
            self.bboxes[frame_idx].append(bbox)
        else:
            self.bboxes[frame_idx] = [bbox]
            
    def _iterate(self, clip: torch.Tensor) -> None:
        '''
        Iterates through the passed video Tensor by track ID, saving each bounding box to the bboxes dictionary. Assumes
        that tracking data collected by SortFish class is in sequential order for each track ID.

        Inputs:
            video: PyTorch Tensor of shape (T, C, H, W) containing a video which has already been run through the YOLO + SORT pipeline.
        '''
        nframes = clip.shape[0]
        # print(f'n_frames: {nframes}')

        # read tracks data from tracks file built by SortFish class
        detections_df = pd.read_csv(self.detections_file)
        detections_df = detections_df[(detections_df['frame'] >= self.starting_frame_idx) & (detections_df['frame'] < self.starting_frame_idx + nframes)]
        detections_df.sort_values(by='frame', ascending=True)

        # iterate by frame
        for _, row in detections_df.iterrows():
            # print(f'frame: {row["frame"]}')
            frame_idx = row['frame'] - self.starting_frame_idx

            x1, x2 = max(row['x1'], row['x2']), min(row['x1'], row['x2']) 
            y1, y2 = max(row['y1'], row['y2']), min(row['y1'], row['y2'])

            width, height = x1 - x2 + 1, y1 - y2 + 1
            x_center, y_center = x2 + (width // 2), y2 + (height // 2)

            # print(f'frame shape: {clip[frame_idx, :, :, :].shape}')

            self._save_bbox(frame_idx, clip[frame_idx, :, :, :], x_center, y_center, width, height)

    def _save_images(self, imgtype='png') -> None:
        '''
        Saves the collected bbox images to individual files in the directory located at self.bboxes_dir.

        Inputs:
            imgtype: the file format to be used in saving the bbox images; defaults to "png".
        '''

        counts = dict()

        for frame_idx, bboxes_list in self.bboxes.items():
            for bbox in bboxes_list:
                # assert bbox.shape == (3, self.dim, self.dim)

                if frame_idx in list(counts.keys()):
                    counts[frame_idx] += 1
                else:
                    counts[frame_idx] = 1

                # bbox_tensor = to_tensor(bbox)

                # w, c, h = bbox_tensor.shape
                # bbox_tensor = bbox_tensor.reshape(c, h, w)
                
                # print(f'frame_idx @ {frame_idx + self.starting_frame_idx} - {self.starting_frame_idx} + 1 == {frame_idx + 1}')

                filename = f'clip{"0" * (9 - math.floor(1 + math.log10(self.clip_index + 1)))}{self.clip_index + 1}_'
                filename += f'frame{"0" * (4 - math.floor(1 + math.log10(frame_idx + 1)))}{frame_idx + 1}_'
                filename += f'n{counts[frame_idx]}'

                # save_image(bbox_tensor, os.path.join(self.bboxes_dir, filename + f'.{imgtype}'), format=f'{imgtype}')
                img = PIL.Image.fromarray(bbox.transpose(2, 1, 0))
                img.save(os.path.join(self.bboxes_dir, filename + f'.{imgtype}'))

    def run(self) -> None:
        '''
        Runs the BBoxCollector pipeline.

        Inputs: none.

        Returns: a dictionary with track IDs as keys and lists of bbox images as values, specifically self.bboxes.
        '''
        # Returns: a Boolean indicating that the data distillation preparer was successfully run.

        # handle conversion from self.videoObj to Tensor using the PyTorch read_video function

        if self.debug:
            print(f'Running BBox collection on video clip {self.clip_file.split("/")[-1]}...')
        
        clip = read_video(self.clip_file, output_format='TCHW')[0]
        # print(f'typeof video: {type(clip)}')
        
        # iteratively save bboxes to dictionary
        if self.debug:
            print(f'\t\tVideo {self.clip_file.split("/")[-1]} shape: {clip.shape}')
            print(f'\t...Iterating through video clip {self.clip_file.split("/")[-1]}')
        
        self._iterate(clip=clip)

        # save each collected bbox to individual PNG files
        if self.debug:
            print(f'\t...Saving BBoxes collected from video clip {self.clip_file.split("/")[-1]}')
    
        self._save_images(imgtype='png')

        # save bboxes dictionary as JSON
        # self.save_as_json()

        # return True

        if self.debug:
            print(f'\tDone collecting from video clip {self.clip_file.split("/")[-1]}')
        
        # return self.bboxes
