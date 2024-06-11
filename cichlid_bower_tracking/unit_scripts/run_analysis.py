# Things to add

import argparse, sys, pdb, subprocess
from helper_modules.file_manager import FileManager as FM

parser = argparse.ArgumentParser()
parser.add_argument('AnalysisType', type = str, choices=['Prep','Depth','Cluster','ClusterClassification', 'TrackFish','AddFishSex','AvgImgDistillation','Summary','All'], help = 'What type of analysis to perform')
parser.add_argument('AnalysisID', type = str, help = 'The ID of the analysis state this project belongs to')
parser.add_argument('ProjectID', type = str, help = 'Identify the projects you want to analyze.')
parser.add_argument('--Workers', type = int, help = 'Number of workers to use to analyze data')
parser.add_argument('--VideoIndex', nargs = '+', help = 'Restrict which videos to run the analysis on')

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

elif args.AnalysisType == 'AssociateClustersWithTracks':
	from data_preparers.cluster_track_association_preparer_new import ClusterTrackAssociationPreparer as CTAP
	
	ctap_obj = CTAP(fm_obj)
	#ctap_obj.summarizeTracks()
	#ctap_obj.associateClustersWithTracks()
	ctap_obj.createMaleFemaleAnnotationVideos()

elif args.AnalysisType == 'AddFishSex':
	p1 = subprocess.run(
		['python3', '-m', 'cichlid_bower_tracking.unit_scripts.add_fish_sex', args.projectID, args.AnalysisID])
	
elif args.AnalysisType == 'AvgImgDistillation':	
	CHANNELS = 3 # channels hyperparameter (should be 1 if Greyscale or 3 if RGB)
	DIM = 128 # dimension hyperparameter, used in resizing cropped bbox images to shape (CHANNELS, DIM, DIM)
	PREC = 64 # precision hyperparameter, used in defining the max pixel-sum values (can be 8, 16, 32, 64)

	if args.VideoIndex is None:
		videos = list(range(len(fm_obj.lp.movies)))
	else:
		videos = args.VideoIndex

	commands = []
	for video_idx in videos:
		videoObj = fm_obj.returnVideoObject(video_idx)

		video_file = videoObj.mp4_file
		tracks_file = videoObj.localFishTracksFile
		avg_imgs_file = videoObj.localAvgImgsFile

		command = ['python3', '-m', 'unit_scripts.distill_data.py', video_file, tracks_file, avg_imgs_file, '--channels', CHANNELS, '--dim', DIM, '--precision', PREC]
		commands.append(command)

	processes = [subprocess.Popen(command) for command in commands]
	for p1 in processes:
		p1.communicate()

		if p1.returncode != 0:
			raise Exception('Data Distillation (Image Averaging) Error')
			
elif args.AnalysisType == 'Summary':
	p1 = subprocess.Popen(
			['python3', '-m', 'cichlid_bower_tracking.unit_scripts.summarize', args.projectID, '--SummaryFile', args.AnalysisID])
