from helper_modules.file_manager import FileManager as FM
import argparse, GPUtil, os, sys, subprocess, yaml, pdb

# This code ensures that modules can be found in their relative directories
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

# Create arguments for the script
parser = argparse.ArgumentParser(description='This script is used to manually prepared projects for downstream analysis')

# mandatory arguments
parser.add_argument('AnalysisType', type=str, choices=['FishDetection','SexDetermination','SandManipulationClassification', 'PyraTCAiT:cls', 'PyraTCAiT:reid'], help='Type of analysis to run')
parser.add_argument('AnalysisID', type = str, default='MC_multi', help = 'ID of analysis state name')

# fish detection args
parser.add_argument('--file', type=str, help='The relative filepath (from YOLOV5_Annotations/) of the specific dataset to be used in training')

# PyraT-CAiT args
# setup args
parser.add_argument('--checkpoints-dir', type=str, help='The path to the directory where the checkpoint files will be stored.')
parser.add_argument('--scheduler', type=str, choices=['reduce-on-plateau', 'warmup-cosine'], default='warmup-cosine', help='The type of learning rate scheduler to use.')
parser.add_argument('--trainset', type=str, help='The filepath to the dataset to be used in training.')
parser.add_argument('--validset', type=str, help='The filepath to the dataset to be used in validation.')
parser.add_argument('--batch-size', '-B', type=int, default=16, help='The size of each batch to be used by both datasets.')
parser.add_argument('--num-classes', '-n', type=int, default=10450, help='The number of classes to be used in the classifier\'s head MLP; defaults to 10450.')
parser.add_argument('--channels', '-c', type=int, choices=[1, 3], default=3, help='The number of channels in the images (1 for greyscale, 3 for RGB); defaults to 3.')
parser.add_argument('--train-dim', type=int, default=224, help='The dimension of the images; defaults to 224.')
parser.add_argument('--valid-dim', type=int, default=256, help='The dimension of the images; defaults to 224.')
parser.add_argument('--use-ddp', '-U', default=False, action='store_true', help='Indicates whether training should be distributed across multiple GPUs.')
parser.add_argument('--num-epochs', '-E', type=int, default=1, help='The number of epochs to use in the time trial; defaults to 1.')
parser.add_argument('--start-epoch', '-e', type=int, default=0, help='The epoch to start training from; only useful if resuming from previous training; defaults to 0.')
parser.add_argument('--device-type', '-i', type=str, choices=['gpu', 'cpu'], default='gpu', help='The device type to use during training/validation; defaults to \'gpu\'.')                    
parser.add_argument('--num-workers', '-W', type=int, default=0, help='The number of workers to use in the dataloaders.')

# model hyperparameters
parser.add_argument('--embed-dims', '-q', type=int, nargs='+', default=[64, 128, 320, 512], help='The embedding dimensions to use in the stages of a PyraT-CAiT model.')
parser.add_argument('--head-counts', '-H', type=int, nargs='+', default=[1, 2, 5, 8], help='The number of attention heads to use in the stages of a PyraT-CAiT model.')
parser.add_argument('--mlp-ratios', '-m', type=int, nargs='+', default=[8, 8, 4, 4], help='The MLP expansion ratios to be used in the stages of a PyraT-CAiT model.')
parser.add_argument('--sr-ratios', '-M', type=int, nargs='+', default=[8, 4, 2, 1], help='The spatial reduction ratios to be used in the stages of a PyraT-CAiT model.')
parser.add_argument('--depths', '-a', type=int, nargs='+', default=[3, 8, 27, 3], help='The depth of each stage of a PyraT-CAiT model.')
parser.add_argument('--num-stages', '-g', type=int, default=4, help='The number of stages to use in a PyraT-CAiT model.')
parser.add_argument('--dropout', '-w', type=float, default=0.1, help='The dropout probability to be used in the stages of a PyraT-CAiT model.')
parser.add_argument('--use-improved', '-I', default=False, action='store_true', help='Indicates whether the PyraT-CAiT model should use the improved stage implementation.')
parser.add_argument('--patch-size', '-p', type=int, default=4, help='The patch size to be used in patch embedding (meaningless if using the \"--use-minipatch\"/\"-m\" option and model is T-CAiT-based); defaults to 16.')

# optimization arguments
parser.add_argument('--cls-loss', type=str, choices=['standard-ce', 'label-smoothing-ce'], default='standard-ce', help='The type of cross entropy classification loss to use; only meaningful if AnalysisType has prefix "PyraTCAiT".')
parser.add_argument('--learning-rate', type=float, default=1e-4, help='The initial learning rate to be used by the AdamW optimizer.')
parser.add_argument('--betas', type=float, nargs='+', default=[0.9, 0.999], help='The beta values to be used by the AdamW optimizer.')
parser.add_argument('--weight-decay', type=float, default=2.5e-4, help='The weight decay to be used by the AdamW optimizer (default value inspired by Loshchilov and Hutter\'s "Fixing Weight Decay Regularization in Adam", Figure 2).')
parser.add_argument('--patience', type=int, default=10, help='The number of epochs without improvement before the scheduler reduces the learning rate; only useful for ReduceLROnPlateau scheduler.')
parser.add_argument('--warmup-epochs', type=int, default=5, help='The number of warmup epochs to be used by the scheduler; only useful for WarmupCosineScheduler.')
parser.add_argument('--eta-min', type=float, default=0.0, help='The minimum learning rate after cosine annealing; only useful for WarmupCosineScheduler.')

# augmentation arguments
parser.add_argument('--jitter', type=float, nargs='+', default=[0.4, 0.4, 0.4, 0.1], help='The color jitter augmentation hyperparameters, in the following order: brightness, contrast, saturation, hue.')
parser.add_argument('--norm-means', type=float, nargs='+', default=[0.485, 0.456, 0.406], help='The RGB color channel means to use in normalizing the data; default values specific to ImageNet.')
parser.add_argument('--norm-stds', type=float, nargs='+', default=[0.229, 0.224, 0.225], help='The TGB color channel standard deviations to use in normalizing the data; default values specific to ImageNet.')

# misc arguments
parser.add_argument('--debug', default=False, action='store_true', help='Puts the script in debug mode so it outputs logging messages.')
parser.add_argument('--disable-progress-bar', default=False, action='store_true', help='Indicates that no progress bar should be printed out during the training/validation processes.')

# parse args
args = parser.parse_args()

# Identify projects to run analysis on
fm_obj = FM(args.AnalysisID)
fm_obj.downloadData(fm_obj.localSummaryFile)
if not fm_obj.checkFileExists(fm_obj.localSummaryFile):
	print('Cant find ' + fm_obj.localSummaryFile)
	sys.exit()

if args.AnalysisType == 'FishDetection':
	
	fm_obj.downloadData(fm_obj.localYolov5AnnotationsDir, tarred = True)

	dataset_file = fm_obj.localYolov5AnnotationsDir + ('dataset.yaml' if args.file is None else args.file)
	with open(dataset_file, 'r') as file:
		dataset = yaml.safe_load(file)
	dataset['path'] = fm_obj.localYolov5AnnotationsDir
	with open(dataset_file, 'w') as file:
		yaml.dump(dataset, file)

	fm_obj.downloadData(fm_obj.localObjectDetectionDir + 'hyp.yaml')

	gpu = GPUtil.getAvailable(order = 'first', maxMemory = 0.2, limit = 8)[0]

	command = ['python3', 'train.py']
	command.extend(['--device', str(gpu)])
	command.extend(['--epochs', '2000'])
	command.extend(['--batch-size','-1'])
	command.extend(['--optimizer','AdamW'])
	command.extend(['--hyp',fm_obj.localObjectDetectionDir + 'hyp.yaml'])
	command.extend(['--data',dataset_file])
	command.extend(['--project', args.AnalysisType])
	command.extend(['--name', args.AnalysisID])

	command = "source " + os.getenv('HOME') + "/anaconda3/etc/profile.d/conda.sh; conda activate yolov5; " + ' '.join(command)

	os.chdir(os.getenv('HOME') + '/yolov5')
	print('bash -c \"' + command + '\"')
	output = subprocess.run('bash -c \"' + command + '\"', shell = True, stderr = open(os.getenv('HOME') + '/' + 'training_errors.txt', 'w'), stdout=subprocess.DEVNULL)
	pdb.set_trace()
	subprocess.run(['cp', '-r', os.getenv('HOME') + '/yolov5/' + args.AnalysisType + '/' + args.AnalysisID, fm_obj.localYolov5InfoDir])
	subprocess.run(['cp', os.getenv('HOME') + '/yolov5/' + args.AnalysisType + '/' + args.AnalysisID + '/weights/best.pt', fm_obj.localYolov5WeightsFile])
	fm_obj.uploadData(fm_obj.localYolov5InfoDir)
	fm_obj.uploadData(fm_obj.localYolov5WeightsFile)
elif args.analysisType.split(':')[0] == 'PyraTCAiT':
	# make sure checkpoints-dir, trainset, and validset arguments are passed
	assert args.checkpoints_dir is not None, f'Error: Must pass checkpoints directory when training PyraT-CAiT!'
	assert args.trainset is not None, f'Error: Must pass trainset when training PyraT-CAiT!'
	assert args.validset is not None, f'Error: Must pass validset when training PyraT-CAiT!'

	# build subprocess command
	command = ['python3', 'train_pyra_tcait.py']
	
	# append mandatory arguments
	command.extend([args.checkpoints_dir, args.analysisType.split(':')[-1], args.scheduler, args.trainset, args.validset])

	# append setup options
	command.extend(['--batch-size', args.batch_size])
	command.extend(['--num-classes', args.num_classes])
	command.extend(['--channels', args.channels])
	command.extend(['--train-dim', args.train_dim])
	command.extend(['--valid-dim', args.valid_dim])
	if args.use_ddp:
		command.extend(['--use-ddp'])
	command.extend(['--num-epochs', args.num_epochs])
	command.extend(['--start-epoch', args.start_epoch])
	command.extend(['--device-type', args.device_type])
	command.extend(['--num-workers', args.num_workers])

	# append optional hyperparameters
	command.extend(['--embed-dims', ' '.join(args.embed_dims)])
	command.extend(['--head-counts', ' '.join(args.head_counts)])
	command.extend(['--mlp-ratios', ' '.join(args.mlp_ratios)])
	command.extend(['--sr-ratios', ' '.join(args.sr_ratios)])
	command.extend(['--depths', ' '.join(args.depths)])
	command.extend(['--num-stages', args.num_stages])
	command.extend(['--dropout', args.dropout])
	if args.use_improved:
		command.extend(['--use-improved'])
	command.extend(['--patch-size', args.patch_size])

	# append optimization arguments
	command.extend(['--cls-loss', args.cls_loss])
	command.extend(['--learning-rate', args.learning_rate])
	command.extend(['--betas', ' '.join(args.betas)])
	command.extend(['--weight-decay', args.weight_decay])
	command.extend(['--patience', args.patience])
	command.extend(['--warmup-epochs', args.warmup_epochs])
	command.extend(['--eta-min', args.eta_min])

	# append augmentation arguments
	command.extend(['--jitter', ' '.join(args.jitter)])
	command.extend(['--norm-means', ' '.join(args.norm_means)])
	command.extend(['--norm-stds', ' '.join(args.norm_stds)])

	# append misc arguments
	if args.debug:
		command.extend(['--debug'])
	if args.disable_progress_bar:
		command.extend(['--disable-progress-bar'])

	# finalize command string and print it
	command = ' '.join(command)
	command = 'source ' + os.getenv('HOME') + '/miniconda3/etc/profile.d/conda.sh; conda activate CichlidDistillation; '
	command = 'bash -c \"' + command + '\"'

	print(command)

	# execute command as a subprocess
	output = subprocess.run(command, shell = True, stderr = open(os.getenv('HOME') + '/' + 'training_errors.txt', 'w'))
