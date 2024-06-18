import argparse

from data_distillation.data.bbox_collector import BBoxCollector
from helper_modules.file_manager import FileManager

# get necessary arguments for using BBoxCollector
parser = argparse.ArgumentParser()

parser.add_argument('AnalysisID', type=str, help='Indicates the ID of the analysis state the project belongs to')
parser.add_argument('ProjectID', type=str, help='Indicates which project the BBoxes will be extracted from')
parser.add_argument('VideoIndex', type=int, help='The index of the video to be analyzed')
parser.add_argument('--dim', type=int, help='The dimension to be used in resizing the BBox images')

args = parser.parse_args()

# create FileManager object and get video object using passed arguments
fm_obj = FileManager(analysisID=args.AnalysisId, projectID=args.ProjectID, check=True)
video_obj  = fm_obj.returnVideoObject(args.VideoIndex)

# create local BBox images directory for specified video
fm_obj.createDirectory(video_obj.localVideoBBoxImagesDir)

# create BBoxCollector and run collection function
bboxc_obj = BBoxCollector(video_file=video_obj.localVideoFile, detections_file=video_obj.localFishDetectionsFile, bboxes_dir=video_obj.localVideoBBoxImagesDir, dim=args.dim)
_ = bboxc_obj.run()