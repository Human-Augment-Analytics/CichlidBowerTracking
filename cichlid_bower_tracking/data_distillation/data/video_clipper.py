from typing import List
import math, os

from torchvision.io import VideoReader, write_video
import torch
import numpy as np

class VideoClipper:
    def __init__(self, video_index: int, video_file: str, clips_dir: str, fps=30, fpc=1800):
        '''
        Initializes an instance of the VideoClipper class.

        Inputs:
            video_index: an int indicating the index of the video in a given project.
            video_file: the string local filepath to the video to be split up.
            clips_dir: the string local path to the directory where the clips will be stored.
            fps: an int indicating the framerate of the passed video (and therefore the clips); defaults to 30.
            fpc: an int indicating the number of "frames per clip"; defaults to 1800, or 1 minute (assuming fps=30).
        '''
        
        self.video_index = video_index
        self.video_file = video_file
        
        self.clips_dir = clips_dir
        self.fpc = fpc
        self.fps = fps

        self.clip_count = 0
        self.reader = VideoReader(self.video_file)

    def _save_clip(self, stack: List[np.ndarray]) -> None:
        '''
        Saves the passed stack of frames as a clip to the local file system.

        Inputs:
            stack: a list containing NumPy ndarray frames from the passed video.
        '''
        
        clip = torch.tensor(np.array(stack), dtype=torch.uint64)
        
        filename = f'{"0" * (9 - math.floor(1 + math.log10(self.clip_count)))}clip'
        filename = os.path.join(self.clips_dir, filename)

        video_codec = 'h264'

        write_video(filename, clip, self.fps, video_codec=video_codec)

    def run(self) -> None:
        '''
        Runs the VideoClipper on the passed video file.

        Inputs: None.
        '''
        
        stack = []

        for frame in self.reader:
            if len(stack) >= self.fpc:
                self._save_clip(stack)
                self.clip_count += 1
            else:
                frame_tensor = frame['data']
                assert isinstance(frame_tensor, torch.Tensor)
                
                stack.append(frame_tensor.detach().numpy())
        
        if len(stack) > 0:
            self._save_clip(stack)
            self.clip_count += 1