from typing import List
import math, os

from torchvision.io import read_video, write_video#, VideoReader
import torch
import numpy as np

class VideoClipper:
    def __init__(self, video_index: int, video_file: str, clips_dir: str, fps=30, fpc=300, debug=False):
        '''
        Initializes an instance of the VideoClipper class.

        Inputs:
            video_index: an int indicating the index of the video in a given project.
            video_file: the string local filepath to the video to be split up.
            clips_dir: the string local path to the directory where the clips will be stored.
            fps: an int indicating the framerate of the passed video (and therefore the clips); defaults to 30.
            fpc: an int indicating the number of "frames per clip"; defaults to 1800, or 1 minute (assuming fps=30).
        '''

        self.__version__ = '0.1.0'
        
        self.video_index = video_index
        self.video_file = video_file
        
        self.clips_dir = clips_dir
        self.fpc = fpc
        self.fps = fps

        self.debug = debug

        self.clip_count = 0
        # self.reader = VideoReader(src=self.video_file)

    def _save_clip(self, clip: torch.Tensor) -> None: # stack: List[np.ndarray]) -> None:
        '''
        Saves the passed stack of frames as a clip to the local file system.

        Inputs:
            stack: a list containing NumPy ndarray frames from the passed video.
        '''
        
        # if self.debug:
        #     print(f'\t...converting to Tensor')
        
        # clip = torch.tensor(np.array(stack), dtype=torch.uint8)

        # t, c, h, w = clip.shape
        # clip = clip.reshape((t, h, w, c))

        # print(f'\t...clip #{self.clip_count + 1} shape: {clip.shape}')
        
        filename = f'{"0" * (9 - math.floor(1 + math.log10(self.clip_count + 1)))}{self.clip_count + 1}clip'
        filename = os.path.join(self.clips_dir, filename)

        video_codec = 'h264'

        if self.debug:
            print(f'\t...writing clip {filename}.mp4')
        
        write_video(f'{filename}.mp4', clip, self.fps, video_codec=video_codec)

    def run(self) -> None:
        '''
        Runs the VideoClipper on the passed video file.

        Inputs: None.
        '''
        
        # stack = []
        # if self.debug:
        #     print('\t...beginning frame iteration')

        # for frame in self.reader:
        #     if len(stack) >= self.fpc:
        #         self._save_clip(stack)
        #         self.clip_count += 1

        #         if self.debug:
        #             print(f'\t...clip #{self.clip_count} done')

        #         stack = []
        #     else:
        #         frame_tensor = frame['data']
        #         assert isinstance(frame_tensor, torch.Tensor)
                
        #         stack.append(frame_tensor.detach().numpy())
                
        # if len(stack) > 0:
        #     self._save_clip(stack)
        #     self.clip_count += 1

        prev = None

        start_idx = 0.0
        spc = self.fpc / self.fps

        clip = torch.randn(2)

        while torch.numel(clip) > 0:
            clip = read_video(filename=self.video_file, start_pts=start_idx, end_pts=start_idx + spc - (1/30), pts_unit='sec', output_format='THWC')[0]

            if clip.shape[0] == 1:
                if prev is None:
                    prev = clip
                elif prev is not None and torch.eq(prev, clip):
                    break

            self._save_clip(clip=clip)
            self.clip_count += 1
            
