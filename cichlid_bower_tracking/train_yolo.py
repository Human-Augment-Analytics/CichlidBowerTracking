# This script was created to allow for easier local training of YOLOv5, especially when managing the file system manually.

import argparse, os, GPUtil, subprocess, yaml, pdb

from helper_modules.file_manager import FileManager

parser = argparse.ArgumentParser()

parser.add_argument('AnalysisID', type=str, default='MC_multi', help='The ID of the projects used; defaults to \'MC_multi\'.')
parser.add_argument('Hyp', type=str, help='The filepath to the hyperparameters yaml file.')
parser.add_argument('Data', type=str, help='The filepath to the data yaml file.')
parser.add_argument('--workers', type=int, default=0, help='The number of workers to use in training.')
parser.add_argument('--epochs', type=int, default=50, help='The number of epochs to use during training.')

args = parser.parse_args()

fm_obj = FileManager(analysisID=args.AnalysisID)

gpu = GPUtil.getAvailable()[0]

command = ['python3', 'train.py', \
           '--device', str(gpu), \
           '--workers', str(args.workers), \
           '--epochs', str(args.epochs), \
           '--batch-size', '-1', \
           '--optimizer', 'AdamW', \
           '--hyp', args.Hyp, \
           '--data', args.Data, \
           '--project', 'FishDetection', \
           '--name', args.AnalysisID]

command = 'source ~/miniconda3/etc/profile.d/conda.sh; conda activate yolov5; ' + ' '.join(command) # assumes use of miniconda3 (change if necessary)
command = 'bash -c \"' + command + '\"'

os.chdir(os.getenv('HOME') + '/yolov5')
print(command)

output = subprocess.run(command, shell=True)
pdb.set_trace()

subprocess.run(['cp', '-r', '~/yolov5/FishDetection/' + args.AnalysisID, fm_obj.localYolov5InfoDir])
subprocess.run(['cp', '~/yolov5/FishDetection/' + args.AnalysisID + '/weights/best.pt', fm_obj.localYolov5WeightsFile])

fm_obj.uploadData(fm_obj.localYolov5InfoDir)
fm_obj.uploadData(fm_obj.localYolov5WeightsFile)