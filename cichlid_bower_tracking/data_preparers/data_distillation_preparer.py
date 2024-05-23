from helper_modules.file_manager import FileManager
from torchvision.io import read_video
import numpy as np
import pandas as pd
import torch
import json

np.random.seed(0)

IMG_W = 1296
IMG_H = 972

class DataDistillationPreparer:
    def __init__(self, FileManager: FileManager, videoIndex: int, tracks_file: str, bboxes_dir: str, filename: str):
        self.__version__ = '0.1.0'
        self.fm = FileManager
        self.videoObj = self.fm.returnVideoObject(videoIndex)
        self.videoIndex = videoIndex
        self.tracks_file = tracks_file
        self.bboxes_dir = bboxes_dir
        self.filename = filename

        self.bboxes = dict()

    def save_bbox(self, frame: torch.Tensor, x_center: int, y_center: int, width: int, height: int, track_id: int) -> bool:
        '''
        Inputs:
            frame: PyTorch Tensor of shape (C, H, W) containing a specific video frame.
            x_center: the x-coordinate of the bbox's center, as calculated during YOLO.
            y_center: the y-coordinate of the bbox's upper-left corner, as calculated during YOLO.
            width: the width of the bbox to be saved, as calculated during YOLO.
            height: the width of the bbox to be saved, as calculated during YOLO.
            track_id: the track ID associated with the bbox to be saved, as stored in the tracks dataset.

        Returns: A Boolean indicating if the bbox was successfully stored in the bboxes dictionary.
        '''

        x_center_scl, y_center_scl = x_center * IMG_W, y_center * IMG_H
        width_scl, height_scl = width * IMG_W, height * IMG_H

        x_upperleft, y_upperleft = x_center_scl - width_scl / 2, y_center_scl - height_scl / 2
        x_lowerright, y_lowerright = x_center_scl + width_scl / 2, y_center_scl + height_scl / 2

        bbox = frame[:, y_upperleft:y_lowerright, x_upperleft:x_lowerright].detach().numpy()
        
        if track_id in list(self.bboxes.keys()):
            self.bboxes[track_id].append(bbox)
        else:
            self.bboxes[track_id] = [bbox]

        return True
    
    def iterate(self, video: torch.Tensor) -> bool:
        '''
        Inputs:
            video: PyTorch Tensor of shape (T, C, H, W) containing a video which has already been run through the YOLO + SORT pipeline.
        
        Returns: a Boolean indicating if the complete iteration through the video Tensor was successful.
        '''
        tracks_df = pd.read_csv(self.tracks_file)
        track_ids = tracks_df['track_id'].unique().tolist()

        for track_id in track_ids:
            tmp_df = tracks_df[tracks_df['track_id'] == track_id][['frame', 'xc', 'yc', 'w', 'h']]

            for _, row in tmp_df.iterrows():
                frame_idx = row['frame']

                x_center, y_center = row['xc'], row['yc']
                width, height = row['w'], row['h']

                self.save_bbox(video[frame_idx, :, :, :], x_center, y_center, width, height, track_id)

    def save_as_json(self) -> bool:
        '''
        Inputs: none.
        Returns: a Boolean indicating if the bboxes dictionary was successfully saved as a JSON file.
        '''

        with open(f'{self.bboxes_dir}/{self.filename}.json', 'w') as file:
            json.dumps(self.bboxes, file)

        return True 


