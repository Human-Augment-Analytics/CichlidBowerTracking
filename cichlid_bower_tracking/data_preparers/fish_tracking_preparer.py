from misc.pace_vars import USING_PACE

import subprocess, os, pdb, datetime, shutil
from shapely.geometry import Point, Polygon

class FishTrackingPreparer():
	# This class takes in a yolov5 model + video information and:
	# 1. Detects fish objects and classifies into normal or reflection using a yolov5 object
	# 2. 
	# 3. Automatically identifies bower location
	# 4. Analyze building, shape, and other pertinent info of the bower

	def __init__(self, fileManager, videoIndex):

		self.__version__ = '1.0.1'
		self.fileManager = fileManager
		self.videoObj = self.fileManager.returnVideoObject(videoIndex)
		self.videoIndex = videoIndex
		self.fileManager.downloadData(self.fileManager.localYolov5WeightsFile)

	def validateInputData(self):
		# print(f'detections file: {self.videoObj.localFishDetectionsFile}')
		# print(f'tracks file: {self.videoObj.localFishTracksFile}')


		assert os.path.exists(self.videoObj.localVideoFile)

		assert os.path.exists(self.fileManager.localTroubleshootingDir)
		assert os.path.exists(self.fileManager.localAnalysisDir)
		assert os.path.exists(self.fileManager.localTempDir)
		assert os.path.exists(self.fileManager.localLogfileDir)
		assert os.path.exists(self.fileManager.localYolov5WeightsFile)

	def runObjectDetectionAnalysis(self, gpu = 0):

		print('Running Object detection on ' + self.videoObj.baseName + ' ' + str(datetime.datetime.now()), flush = True)
		self.annotations_dir = self.fileManager.localTempDir + self.videoObj.localVideoFile.split('/')[-1].replace('.mp4','')

		if os.path.exists(self.annotations_dir + '/exp/labels/'):
			shutil.rmtree(self.annotations_dir + '/exp/labels/')

		command = ['python3', 'detect.py']
		command.extend(['--weights', self.fileManager.localYolov5WeightsFile])
		command.extend(['--source', self.videoObj.localVideoFile])
		command.extend(['--device', str(gpu)])
		command.extend(['--project', self.annotations_dir])
		command.extend(['--save-txt', '--nosave', '--save-conf','--agnostic-nms'])

		# dynamically obtain anaconda distro directory in HOME
		home_subdirs = os.listdir(os.getenv('HOME'))
		# print(f'\nhome_subdirs:\n{home_subdirs}\n')

		if 'anaconda3' in home_subdirs:
			conda_dir = 'anaconda3'
		elif 'miniconda3' in home_subdirs:
			conda_dir = 'miniconda3'
		else:
			raise Exception(f'FishTrackingPreparer Error: Missing anaconda distribution from {os.getenv("HOME")}')

		command = "source " + os.getenv('HOME') + f"/{conda_dir}/etc/profile.d/conda.sh; conda activate yolov5; " + ' '.join(command)

		os.chdir(os.getenv('HOME') + '/yolov5')
		print('bash -c \"' + command + '\"')
		output = subprocess.Popen('bash -c \"' + command + '\"', shell = True, stderr = open(os.getenv('HOME') + '/' + self.videoObj.baseName + '_detectionerrors.txt', 'w'), stdout=open(os.getenv('HOME') + '/' + self.videoObj.baseName + '_detectionstdout.txt', 'w'))
		#os.chdir(os.getenv('HOME') + '/CichlidBowerTracking/cichlid_bower_tracking')
		return output

	def runSORT(self):
		self.annotations_dir = self.fileManager.localTempDir + self.videoObj.localVideoFile.split('/')[-1].replace('.mp4','')

		# dynamically obtain anaconda distro directory in HOME
		home_subdirs = os.listdir(os.getenv('HOME'))

		if 'anaconda3' in home_subdirs:
			conda_dir = 'anaconda3'
		elif 'miniconda3' in home_subdirs:
			conda_dir = 'miniconda3'
		else:
			raise Exception(f'FishTrackingPreparer Error: Missing anaconda distribution from {os.getenv("HOME")}')
		
		# change directory
		new_dir = os.getenv('HOME') + '/'
		if USING_PACE:
			new_dir += 'ondemand' + '/'
		new_dir += '/CichlidBowerTracking/cichlid_bower_tracking'

		os.chdir(new_dir)
		print('Running Sort detection on ' + self.videoObj.baseName + ' ' + str(datetime.datetime.now()), flush = True)

		command = ['python3', 'unit_scripts/sort_detections.py', self.annotations_dir + '/exp/labels/', self.videoObj.localFishDetectionsFile, self.videoObj.localFishTracksFile, self.videoObj.baseName]

		command = "source " + os.getenv('HOME') + f"/{conda_dir}/etc/profile.d/conda.sh; conda activate CichlidSort; " + ' '.join(command)
		#subprocess.run('bash -c \"' + command + '\"', shell = True)

		print('bash -c \"' + command + '\"')
		output = subprocess.Popen('bash -c \"' + command + '\"', shell = True, stderr = open(os.getenv('HOME') + '/' + self.videoObj.baseName + '_trackingerrors.txt', 'w'), stdout=open(os.getenv('HOME') + '/' + self.videoObj.baseName + '_trackingstdout.txt', 'w'))
		return output


