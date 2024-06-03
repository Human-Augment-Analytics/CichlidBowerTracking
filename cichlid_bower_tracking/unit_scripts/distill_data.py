# ==================================================================================================================
# This script is called as a subprocess of the runAnalysis.py script.
#
# It performs the data distillation (just image averaging for now) procedure on the passed video.
# ==================================================================================================================

from helper_modules.data_distiller import DataDistiller
import argparse, datetime
from torch import uint8, uint16, uint32, uint64

# init argument parser and parse arguments
parser = argparse.ArgumentParser(usage='This script runs a data distiller on the specified video file.')

parser.add_argument('VideoFile', type=str, help='This is the filepath of the video to be used in image-averaging data distillation.')
parser.add_argument('TracksFile', type=str, help='This is the filepath of the tracks CSV file to be used in cropping bboxes from the passed video.')
parser.add_argument('AvgImgDirectory', type=str, help='This is the path to the directory in which the average images will be saved.')
parser.add_argument('--videoidx', type=int, help='This is the index associated with the VideoFile, useful in naming the average images file.')
parser.add_argument('--channels', type=int, help='This is the number of channels in the bbox images, should be 3 for RGB.')
parser.add_argument('--dim', type=int, help='This is the dimension size to be used as the length and width.')
parser.add_argument('--precision', type=int, choices=[8, 16, 32, 64], help='This is the precision to be used during the averaging process.')

args = parser.parse_args()

# get DataDistiller parameters from passed arguments
video_file = args.VideoFile
tracks_file = args.TracksFile
avg_imgs_dir = args.AvgImgDirectory

video_idx = args.videoidx
channels = args.channels
dim = args.dim

npz_filename = f'videoidx-{video_idx}-avg-imgs'

if args.precision == 8:
    dtype = uint8
elif args.precision == 16:
    dtype = uint16
elif args.precision == 32:
    dtype = uint32
else:
    dtype = uint64

# initialize data distiller and run distillation (image averaging)
distiller = DataDistiller(video_file=video_file, tracks_file=tracks_file, channels=channels, dim=dim, avg_imgs_dir=avg_imgs_dir, npz_filename=npz_filename, dtype=dtype)

print(f'Running data distillation (image averaging) on {video_file} @ {datetime.datetime.now()}', flush=True)
if distiller.run_distillation():
    print(f'\tData distillation on {video_file} completed @ {datetime.datetime.now()}', flush=True)