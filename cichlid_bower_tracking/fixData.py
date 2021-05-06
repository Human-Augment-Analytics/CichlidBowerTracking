import argparse, subprocess, pdb, shutil, os
import pandas as pd
from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM


parser = argparse.ArgumentParser(description='This script is used to manually prepared projects for downstream analysis')
parser.add_argument('--SummaryFile', type = str, help = 'Restrict analysis to projectIDs specified in csv file, which will be rewritten. ProjectIDs must be found in a column called projectID')
args = parser.parse_args()

fm_obj = FM() 

if args.SummaryFile is not None:
	summary_file = fm_obj.localAnalysisStatesDir + args.SummaryFile
	fm_obj.downloadData(summary_file)
	dt = pd.read_csv(summary_file, index_col = 0)
	projectIDs = list(dt.projectID)
else:
	projectIDs = fm_obj.getAllProjectIDs()

for projectID in projectIDs:
	fm_obj = FM(projectID = projectID)
	print(projectID)
	lp = fm_obj.lp

	main_directory_data = subprocess.run(['rclone', 'lsf', 'cichlidVideo:McGrath/Apps/CichlidPiData/' + projectID + '/'], capture_output = True, encoding = 'utf-8').stdout.split('\n')

	# Directories to delete
	for bad_data in ['PBS/', 'Backgrounds/', 'Figures/', 'DepthAnalysis/', 'VideoAnalysis/', 'Troubleshooting/','AllClips/','MLClips/', 'MLFrames/']:
		if bad_data in main_directory_data:
			print('  Deleting: ' + bad_data)
			subprocess.run(['rclone','purge', 'cichlidVideo:McGrath/Apps/CichlidPiData/' + projectID + '/' + bad_data])

	# Files to delete
	for bad_data in ['AllClips.tar', 'MLClips.tar', 'MLFrames.tar', 'Backgrounds.tar']:
		if bad_data in main_directory_data:
			print('  Deleting: ' + bad_data)
			subprocess.run(['rclone','delete', 'cichlidVideo:McGrath/Apps/CichlidPiData/' + projectID + '/' + bad_data])

	if 'Frames/' in main_directory_data:
		if 'Frames.tar' in main_directory_data:
			print('  Deleting: Frames/')
			subprocess.run(['rclone','purge', 'cichlidVideo:McGrath/Apps/CichlidPiData/' + projectID + '/' + 'Frames/'])
		else:
			print('  Need to convert Frames to tar')

	video_directory_data = subprocess.run(['rclone', 'lsf', 'cichlidVideo:McGrath/Apps/CichlidPiData/' + projectID + '/Videos/'], capture_output = True, encoding = 'utf-8').stdout.split('\n')
	for index,vid_obj in enumerate(lp.movies):
		vid_obj = fm_obj.returnVideoObject(index)
		if os.path.basename(vid_obj.localVideoFile) not in video_directory_data:
			if os.path.basename(vid_obj.localh264File) not in video_directory_data:
				print('  Missing videofiles: ' + str(index))
			else:
				print('  Need to convert h264 file: ' + str(index))
				fm_obj.downloadData(vid_obj.localh264File)
				output = subprocess.run(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.process_video', vid_obj.localh264File, str(vid_obj.framerate), projectID], capture_output = True)
				if output.returncode != 0:
					pdb.set_trace()
				subprocess.run(['rm', '-rf', vid_obj.localh264File])
				subprocess.run(['rm', '-rf', vid_obj.localVideoFile])
		
		if os.path.basename(vid_obj.localh264File) in video_directory_data:	
			print('  Deleting h264 file: ' +  str(index))	
			subprocess.run(['rclone','delete', 'cichlidVideo:McGrath/Apps/CichlidPiData/' + projectID + '/Videos/' + os.path.basename(vid_obj.localh264File)])
	
	analysis_directory_data = subprocess.run(['rclone', 'lsf', 'cichlidVideo:McGrath/Apps/CichlidPiData/' + projectID + '/MasterAnalysisFiles/'], capture_output = True, encoding = 'utf-8').stdout.split('\n')
	for bad_data in ['smoothedDepthData.npy', 'DepthRGBVideo.mp4']:
		if bad_data in analysis_directory_data:
			print('  Deleting: ' + bad_data)
			subprocess.run(['rclone','delete', 'cichlidVideo:McGrath/Apps/CichlidPiData/' + projectID + '/MasterAnalysisFiles/' + bad_data])

	for main_data in main_directory_data:
		if '.npy' in main_data or '.pdf' in main_data:
			print(' Deleting: ' + main_data)
			subprocess.run(['rclone','delete', 'cichlidVideo:McGrath/Apps/CichlidPiData/' + projectID + '/' + main_data])

	analysis_directory_data = subprocess.run(['rclone', 'lsf', 'cichlidVideo:McGrath/Apps/CichlidPiData/' + projectID + '/MasterAnalysisFiles/'], capture_output = True, encoding = 'utf-8').stdout.split('\n')
	if 'AllLabeledClusters.csv' in analysis_directory_data:
		print('  Deleting: AllLabeledClusters.csv')
		subprocess.run(['rclone','delete', 'cichlidVideo:McGrath/Apps/CichlidPiData/' + projectID + '/MasterAnalysisFiles/AllLabeledClusters.csv'])

	subprocess.run(['rm', '-rf', fm_obj.localProjectDir])


