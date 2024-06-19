import argparse

from data_distillation.data.video_clipper import VideoClipper
from data_distillation.data.bbox_collector import BBoxCollector
from helper_modules.file_manager import FileManager

# get necessary arguments for using BBoxCollector
parser = argparse.ArgumentParser()

parser.add_argument('AnalysisID', type=str, help='Indicates the ID of the analysis state the project belongs to')
parser.add_argument('ProjectID', type=str, help='Indicates which project the BBoxes will be extracted from')
parser.add_argument('VideoIndex', type=int, help='The index of the video to be analyzed')
parser.add_argument('--dim', type=int, help='The dimension to be used in resizing the BBox images')
parser.add_argument('--fpc', type=int, help='The number of frames per video clip to be used by the VideoClipper')

args = parser.parse_args()

# create FileManager object and get video object using passed arguments

print(f'Getting video {args.VideoIndex}')
fm_obj = FileManager(analysisID=args.AnalysisID, projectID=args.ProjectID, check=True)
video_obj  = fm_obj.returnVideoObject(args.VideoIndex)

# create local storage directories for specified video
print(f'Creating video clips directory {video_obj.localVideoClipsDir.rstrip("/").split("/")[-1]}')
fm_obj.createDirectory(video_obj.localVideoClipsDir)

print(f'Creating BBox images directory {video_obj.localVideoBBoxImagesDir.rstrip("/").split("/")[-1]}')
fm_obj.createDirectory(video_obj.localVideoBBoxImagesDir)

# create VideoClipper and generate video clips
fpc = 1800 if args.fpc is None else args.fpc

print(f'Generating smaller clips for video {video_obj.baseName}')
clipper = VideoClipper(video_index=args.VideoIndex, video_file=video_obj.localVideoFile, clips_dir=video_obj.localVideoClipsDir, fps=video_obj.framerate, fpc=1800)

clipper.run()

# create BBoxCollector and run collection function
dim = 128 if args.dim is None else args.dim

print(f'Creating BBoxCollector for video {video_obj.baseName}')
bboxc_obj = BBoxCollector(video_file=video_obj.localVideoFile, detections_file=video_obj.localFishDetectionsFile, bboxes_dir=video_obj.localVideoBBoxImagesDir, dim=args.dim)

_ = bboxc_obj.run()