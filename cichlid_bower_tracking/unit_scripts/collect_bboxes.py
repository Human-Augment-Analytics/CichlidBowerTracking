import argparse, os, shutil

from data_distillation.data.bbox_collector import BBoxCollector
from helper_modules.file_manager import FileManager

# get necessary arguments for using BBoxCollector
parser = argparse.ArgumentParser()

parser.add_argument('AnalysisID', type=str, help='Indicates the ID of the analysis state the project belongs to')
parser.add_argument('ProjectID', type=str, help='Indicates which project the BBoxes will be extracted from')
parser.add_argument('ClipFile', type=str, help='The filepath of the video clip to be collected from')
parser.add_argument('VideoIndex', type=int, help='The index of the video in the file system')
parser.add_argument('ClipIndex', type=int, help='Essentially the clip number')
parser.add_argument('StartingFrameIndex', type=int, help='The index from the larger video at which the first frame of the clip is located (1-indexed)')
parser.add_argument('--dim', type=int, help='The dimension to be used in resizing the BBox images')
parser.add_argument('--debug', type=bool, help='Boolean flag to put the BBoxCollector in debug mode')

args = parser.parse_args()

# create FileManager object using passed arguments
print(f'Using video {args.VideoIndex} clip {args.ClipIndex}')

fm_obj = FileManager(analysisID=args.AnalysisID, projectID=args.ProjectID, check=True)
video_obj  = fm_obj.returnVideoObject(args.VideoIndex)

# create local BBox Images storage directory for specified video
if os.path.exists(video_obj.localVideoBBoxImagesDir):
    print(f'Cleaning out bbox images directory {video_obj.localVideoBBoxImagesDir.rstrip("/").split("/")[-1]}')

    shutil.rmtree(video_obj.localVideoBBoxImagesDir)
else:
    print(f'Creating bbox images directory {video_obj.localVideoBBoxImagesDir.rstrip("/").split("/")[-1]}')

fm_obj.createDirectory(video_obj.localVideoBBoxImagesDir)

# create BBoxCollector and run collection function
dim = 256 if args.dim is None else args.dim
debug = False if args.debug is None else args.debug

print(f'Creating BBoxCollector for video {video_obj.baseName} clip {args.ClipFile.rstrip("/").split("/")[-1]}')
bboxc_obj = BBoxCollector(clip_file=args.ClipFile, detections_file=video_obj.localFishDetectionsFile, bboxes_dir=video_obj.localVideoBBoxImagesDir, clip_index=args.ClipIndex, starting_frame_index=args.StartingFrameIndex, dim=dim, debug=debug)

print(f'Running collection process...')
bboxc_obj.run()