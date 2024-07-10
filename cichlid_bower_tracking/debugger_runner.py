# This script was created to make debugging runAnalysis.py easier.
# 
# NOTE: you NEED to generate a launch.json file before trying to debug (assuming you use VS Code).
# if you want to see the code run through all code (not just yours), add the following key-value pair to the inner dictionary
# inside the "configurations" array in launch.json:
# 
# "justMyCode": false
# ===============================================================================================================================

# (mandatory) debugging hyperparameters...
analysis_type = 'TrackFish'
analysis_id = 'MC_multi' 

# (optional) debugging hyperparameters...
project_ids = ['MC_s15_tr2_BowerBuilding']
workers = -1
video_index = -1
delete = False

# -------------------------------------------------------------------------------------------------------------------------------
# DO NOT CHANGE BELOW THIS LINE UNLESS YOU SUFFICIENTLY UNDERSTAND THE CODE BASE

import os, subprocess

# create base command with mandatory hyperparams
command = ['python3', './runAnalysis.py']
command.extend([analysis_type, analysis_id])

# extend base command with optional hyperparams
if project_ids is not None:
    command.extend(['--ProjectIDs'] + [project_id for project_id in project_ids])
if workers >= 0:
    command.extend(['--Workers', workers])
if video_index >= 0:
    command.extend(['--VideoIndex', video_index])
if delete:
    command.extend(['--Delete'])

# dynamically obtain anaconda distro directory in HOME
home_subdirs = os.listdir(os.getenv('HOME'))

if 'anaconda3' in home_subdirs:
    conda_dir = 'anaconda3'
elif 'miniconda3' in home_subdirs:
    conda_dir = 'miniconda3'
else:
    raise Exception(f'FishTrackingPreparer Error: Missing anaconda distribution from {os.getenv("HOME")}')

# create full command
command = ' '.join(command)
command = 'source ' + os.getenv('HOME') + f'/{conda_dir}/etc/profile.d/conda.sh; conda activate CichlidBowerTracking; ' + command
command = 'bash -c \"' + command + '\"'

# run command as subprocess
p1 = subprocess.Popen(command, shell=True)
p1.communicate()

if p1.returncode != 0:
    raise Exception(f'ERROR : Return Code {p1.returncode}')