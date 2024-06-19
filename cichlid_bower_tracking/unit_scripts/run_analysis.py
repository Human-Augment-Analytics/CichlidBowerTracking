# Things to add

import argparse, sys, pdb, subprocess, os
from helper_modules.file_manager import FileManager as FM
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('AnalysisType', type = str, choices=['Prep','Depth','Cluster','ClusterClassification', 'TrackFish','AddFishSex','CollectBBoxes','Summary','All'], help = 'What type of analysis to perform')
parser.add_argument('AnalysisID', type = str, help = 'The ID of the analysis state this project belongs to')
parser.add_argument('ProjectID', type = str, help = 'Identify the projects you want to analyze.')
parser.add_argument('--Workers', type = int, help = 'Number of workers to use to analyze data')
parser.add_argument('--VideoIndex', nargs = '+', help = 'Restrict which videos to run the analysis on')
parser.add_argument('--Dim', type=int, help='Specific to the "CollectBBoxes" option, this indicates what dimension should be used in resizing bbox images collected')

args = parser.parse_args()

fm_obj = FM(args.AnalysisID, projectID = args.ProjectID, check = True)

# Run appropriate analysis script
if args.AnalysisType == 'Prep':
	from data_preparers.prep_preparer import PrepPreparer as PrP
	prp_obj = PrP(fm_obj)
	prp_obj.validateInputData()
	prp_obj.prepData()

elif args.AnalysisType == 'Depth':
	from data_preparers.depth_preparer import DepthPreparer as DP
	dp_obj = DP(fm_obj)
	dp_obj.validateInputData()
	dp_obj.createSmoothedArray()
	dp_obj.createDepthFigures()
	dp_obj.createRGBVideo()

elif args.AnalysisType == 'Cluster':
	from data_preparers.cluster_preparer import ClusterPreparer as CP

	if args.VideoIndex is None:
		videos = list(range(len(fm_obj.lp.movies)))
	else:
		videos = args.VideoIndex

	for videoIndex in videos:
		cp_obj = CP(fm_obj, videoIndex, args.Workers)
		cp_obj.validateInputData()
		cp_obj.runClusterAnalysis()

elif args.AnalysisType == 'ClusterClassification':
	from data_preparers.threeD_classifier_preparer import ThreeDClassifierPreparer as TDCP

	tdcp_obj = TDCP(fm_obj)
	tdcp_obj.validateInputData()
	tdcp_obj.predictLabels()
	tdcp_obj.createSummaryFile()

elif args.AnalysisType == 'TrackFish':
	import GPUtil
	from data_preparers.fish_tracking_preparer import FishTrackingPreparer as FTP
	
	# Identify videos to process
	if args.VideoIndex is None:
		videos = list(range(len(fm_obj.lp.movies)))
	else:
		videos = args.VideoIndex
	
	#Loop through videos and track fish
	ftp_objs = []
	for videoIndex in videos:
		ftp_objs.append(FTP(fm_obj, videoIndex))
		ftp_objs[-1].validateInputData()

	available_cards = GPUtil.getAvailable(order = 'first', maxMemory = 0.2, limit = 8)

	current_idx = 0
	while current_idx < len(videos):
		processes = []
		for i in range(len(available_cards)):
			for gpu in available_cards:
				if current_idx < len(videos):
					processes.append(ftp_objs[current_idx].runObjectDetectionAnalysis(gpu))
					current_idx += 1

		for p1 in processes:
			p1.communicate()
			if p1.returncode != 0:
				# print(f'\nreturn code : {p1.returncode}\n')

				raise Exception('YOLO Error')
	
	processes = []
	for idx in range(len(videos)):
		processes.append(ftp_objs[idx].runSORT())

	for p1 in processes:
		p1.communicate()
		if p1.returncode != 0:
			raise Exception('SORT Error')

	# Combine predictions
	for videoIndex in videos:
		videoObj = fm_obj.returnVideoObject(videoIndex)
		new_dt_t = pd.read_csv(videoObj.localFishTracksFile)
		new_dt_d = pd.read_csv(videoObj.localFishDetectionsFile)
		
		try:
			c_dt_t = c_dt_t.append(new_dt_t)
			c_dt_d = c_dt_d.append(new_dt_d)
		except NameError:
			c_dt_t = new_dt_t
			c_dt_d = new_dt_d
			
	c_dt_t.to_csv(fm_obj.localAllFishTracksFile)
	c_dt_d.to_csv(fm_obj.localAllFishDetectionsFile)

elif args.AnalysisType == 'CollectBBoxes':	
	if args.VideoIndex is None:
		videos = list(range(len(fm_obj.lp.movies)))
	else:
		videos = args.VideoIndex

	# dynamically obtain anaconda distro directory in HOME
	home_subdirs = os.listdir(os.getenv('HOME'))

	if 'anaconda3' in home_subdirs:
		conda_dir = 'anaconda3'
	elif 'miniconda3' in home_subdirs:
		conda_dir = 'miniconda3'
	else:
		raise Exception(f'Conda Error: Missing anaconda distribution from {os.getenv("HOME")}')

	# construct and store commands
	base_command = 'source ' + os.getenv('HOME') + f'/{conda_dir}/etc/profile.d/conda.sh; conda activate CichlidDistillation; '

	commands = []
	for videoIndex in videos:
		print(f'videoIndex: {videoIndex}')

		py_command = ['python3', '-m', 'unit_scripts.collect_bboxes', args.AnalysisID, args.ProjectID, f'{videoIndex}']
		if args.Dim is not None:
			py_command += ['--dim', f'{args.Dim}']

		py_command = ' '.join(py_command)
		full_command = base_command + py_command

		commands.append('bash -c \"' + full_command + '\"')
	
	# execute stored collection commands for each video
	processes = [subprocess.Popen(command, shell=True) for command in commands]

	command_idx = 0
	for p1 in processes:
		p1.communicate()

		if p1.returncode != 0:
			raise Exception(f'BBox Collection Error: "{commands[command_idx]}" subprocess returned non-zero code')
		
	# return to CichlidBowerTracking conda env
	# command = "source " + os.getenv('HOME') + f"/{conda_dir}/etc/profile.d/conda.sh; conda activate CichlidBowerTracking"
	# p2 = subprocess.Popen(command)

	# p2.communicate()
	# if p2.returncode != 0:
	# 	raise Exception(f'Conda Error: "{command}" subprocess returned non-zero code')

elif args.AnalysisType == 'AssociateClustersWithTracks':
	from data_preparers.cluster_track_association_preparer_new import ClusterTrackAssociationPreparer as CTAP
	
	ctap_obj = CTAP(fm_obj)
	#ctap_obj.summarizeTracks()
	#ctap_obj.associateClustersWithTracks()
	ctap_obj.createMaleFemaleAnnotationVideos()

elif args.AnalysisType == 'AddFishSex':
	p1 = subprocess.run(
		['python3', '-m', 'cichlid_bower_tracking.unit_scripts.add_fish_sex', args.projectID, args.AnalysisID])
	
# elif args.AnalysisType == 'AvgImgDistillation':	
# 	CHANNELS = 3 # channels hyperparameter (should be 1 if Greyscale or 3 if RGB)
# 	DIM = 128 # dimension hyperparameter, used in resizing cropped bbox images to shape (CHANNELS, DIM, DIM)
# 	PREC = 64 # precision hyperparameter, used in defining the max pixel-sum values (can be 8, 16, 32, 64)

# 	if args.VideoIndex is None:
# 		videos = list(range(len(fm_obj.lp.movies)))
# 	else:
# 		videos = args.VideoIndex

# 	commands = []
# 	for video_idx in videos:
# 		videoObj = fm_obj.returnVideoObject(video_idx)

# 		video_file = videoObj.mp4_file
# 		tracks_file = videoObj.localFishTracksFile
# 		avg_imgs_file = videoObj.localAvgImgsFile

# 		command = ['python3', '-m', 'unit_scripts.distill_data.py', video_file, tracks_file, avg_imgs_file, '--channels', CHANNELS, '--dim', DIM, '--precision', PREC]
# 		commands.append(command)

# 	processes = [subprocess.Popen(command) for command in commands]
# 	for p1 in processes:
# 		p1.communicate()

# 		if p1.returncode != 0:
# 			raise Exception('Data Distillation (Image Averaging) Error')
			
elif args.AnalysisType == 'Summary':
	p1 = subprocess.Popen(
			['python3', '-m', 'cichlid_bower_tracking.unit_scripts.summarize', args.projectID, '--SummaryFile', args.AnalysisID])
