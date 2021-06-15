# To do
# 1. Make summary file on Dropbox
# 2. Handle Sigint to make sure uploads complete

import argparse, subprocess, pdb, datetime, os
import pandas as pd
from cichlid_bower_tracking.helper_modules.file_manager import FileManager as FM

parser = argparse.ArgumentParser(description='This script is used to manually prepared projects for downstream analysis')
parser.add_argument('AnalysisType', type = str, choices=['Prep','Depth','Cluster','ClusterClassification','Summary'], help = 'Type of analysis to run')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--ProjectIDs', type = str, nargs = '+', help = 'Name of projectIDs to run analysis on')
group.add_argument('--SummaryFile', type = str, help = 'Name of csv file that specifies projects to analyze')
parser.add_argument('--Workers', type = int, help = 'Number of workers')
parser.add_argument('--ModelID', type = str, help = 'ModelID to use to classify clusters with')
parser.add_argument('--Force', type = bool, default=False, help='if True, run the analysis even if the summary file indicates is has already been run. Default False')

args = parser.parse_args()

# Identify projects to run analysis on
if args.ProjectIDs is not None:
	projectIDs = args.ProjectIDs # Specified at the command line
else:
	fm_obj = FM(summaryFile=args.SummaryFile)

	fm_obj.downloadData(fm_obj.localSummaryFile)
	dt = pd.read_csv(fm_obj.localSummaryFile)

	if args.Force:
		projectIDs = list(dt.projectID)
	else:
		projectIDs = list(dt[dt[args.AnalysisType] == False].projectID) # Only run analysis on projects that need it

if args.Workers is None:
	workers = os.cpu_count()
else:
	workers = args.Workers
if args.ModelID is None:
	args.ModelID = 'Model18_All'

# To run analysis efficiently, we download and upload data in the background while the main script runs
uploadProcesses = [] # Keep track of all of the processes still uploading so we don't quit before they finish
print('Downloading: ' + projectIDs[0] + ' ' + str(datetime.datetime.now()))
subprocess.run(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.download_data',args.AnalysisType, '--ProjectID', projectIDs[0], '--ModelID', args.ModelID])
for i, projectID in enumerate(projectIDs):
	print('Running: ' + projectID + ' ' + str(datetime.datetime.now()))

	# Run appropriate analysis script
	if args.AnalysisType == 'Prep':
		p1 = subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.prep_data', projectID])
	elif args.AnalysisType == 'Depth':
		p1 = subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.analyze_depth', projectID])
	elif args.AnalysisType == 'Cluster':
		p1 = subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.analyze_clusters', projectID,'--Workers', str(workers)])
	elif args.AnalysisType == 'ClusterClassification':
		p1 = subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.classify_clusters', projectID, args.ModelID])
	elif args.AnalysisType == 'Summary':
		if args.SummaryFile is None:
			p1 = subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.summarize', projectID])
		else:
			p1 = subprocess.Popen(
				['python3', '-m', 'cichlid_bower_tracking.unit_scripts.summarize', projectID, '--SummaryFile', args.SummaryFile])

	# In the meantime, download data for next project in the background
	if i+1 < len(projectIDs):
		print('Downloading: ' + projectID + ' ' + str(datetime.datetime.now()))
		p2 = subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.download_data', args.AnalysisType, '--ProjectID', projectIDs[i+1]])
	
	# Pause script until current analysis is complete and data for next project is downloaded
	p1.communicate()
	try:
		p2.communicate() # Need to catch an exception if only one project is analyzed
	except NameError:
		pass
	#Modify summary file if necessary
	if args.SummaryFile:
		dt.loc[dt.projectID == projectID,args.AnalysisType] = True
		dt.to_csv(fm_obj.localSummaryFile, index=False)
		fm_obj.uploadData(fm_obj.localSummaryFile)


	#Upload data and keep track of it
	print('Uploading: ' + projectID + ' ' + str(datetime.datetime.now()))

	uploadProcesses.append(subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.upload_data', args.AnalysisType, '--Delete', '--ProjectID', projectID]))
	#uploadProcesses.append(subprocess.Popen(['python3', '-m', 'cichlid_bower_tracking.unit_scripts.upload_data', args.AnalysisType, projectID]))

for i,p in enumerate(uploadProcesses):
	print('Finishing uploading process ' + str(i) + ': ' + str(datetime.datetime.now()))
	p.communicate()
print('Finished analysis: ' + str(datetime.datetime.now()))
