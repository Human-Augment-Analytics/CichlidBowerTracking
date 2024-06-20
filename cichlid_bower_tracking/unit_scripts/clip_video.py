import argparse

from data_distillation.data.video_clipper import VideoClipper
from helper_modules.file_manager import FileManager

# get necessary arguments for using VideoClipper
parser = argparse.ArgumentParser()

parser.add_argument('AnalysisID', type=str, help='The ID of the analysis the project belongs to')
parser.add_argument('ProjectID', type=str, help='The ID of the specific project to be considered')
parser.add_argument('VideoIndex', type=int, help='The index of the video to be split up in the file system')
parser.add_argument('--fpc', type=int, help='The (maximum) number of frames per clip')

args = parser.parse_args()

# create FileManager object and get video object using passed arguments
print(f'Getting video {args.VideoIndex}')
fm_obj = FileManager(analysisID=args.AnalysisID, projectID=args.ProjectID, check=True)
video_obj = fm_obj.returnVideoObject(args.VideoIndex)

# create local BBox Images storage directory for specified video
print(f'Creating video clips directory {video_obj.localVideoClipsDir.rstrip("/").split("/")[-1]}')
fm_obj.createDirectory(video_obj.localVideoClipsDir)

# create VideoClipper and generate video clips
fpc = 1800 if args.fpc is None else args.fpc

clipper = VideoClipper(video_index=args.VideoIndex, video_file=video_obj.localVideoFile, clips_dir=video_obj.localVideoClipsDir, fps=video_obj.framerate, fpc=fpc)
clipper.run()