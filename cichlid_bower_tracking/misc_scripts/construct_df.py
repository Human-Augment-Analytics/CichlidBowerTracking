import argparse, os, datetime, random
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('basedir', type=str, help='The path to the base directory containing the data files.')
parser.add_argument('tgtfile', type=str, help='The csv file where the dataset will be stored.')
parser.add_argument('--batchsize', '-b', type=int, default=500, help='The batch size to be used in constructing the dataset.')
parser.add_argument('--minperclass', type=int, default=2, help='The minimum number of samples to use per class; used in cleaning out classes with too few images.')
parser.add_argument('--maxperclass', type=int, help='The maximum number of samples to use per class.')
parser.add_argument('--seed', '-s', type=int, default=42, help='The seed to use in the random module.')
parser.add_argument('--debug', '-d', default=False, action='set_true', help='Runs the dataset cleaner with inter-process print statements.')

args = parser.parse_args()

assert args.minperclass >= 2, f'Invalid Argument to \"--minperclass\" option: must be at least 2 ({args.minperclass} passed).'

# remove classes with too few files
print(f'Cleaning up the data inside {args.basedir} for a triplet-structured dataset [{datetime.datetime.now()}]...')

class_dirs = os.listdir(parser.basedir)
num_classes = len(class_dirs)

for class_dir in class_dirs:
    full_class_dir = args.basedir.rstrip('/ ') + '/' + class_dir
    class_files = os.listdir(full_class_dir)

    if len(class_files) < args.minperclass: 
        class_dirs.remove(class_dir)

        if args.debug:
            print(f'\t...Class \"{class_dir}\" removed [{datetime.datetime.now()}].')

new_num_classes = len(os.listdir(parser.basedir))
num_removed = num_classes - new_num_classes

print(f'Cleaning complete [{datetime.datetime.now()}]: dataset now contains {new_num_classes} ({num_removed} removed).\n')

# construct the dataset in a csv file
print(f'Constructing a triplet dataset using data found inside {args.basedir} [{datetime.datetime.now()}]...')

ANCHOR_IDX, POS_IDX, NEG_IDX = 0, 1, 2
random.seed(args.seed)

df = pd.DataFrame(columns=['anchor_path', 'positive_path', 'negative_path', 'class'])
df.to_csv(args.tgtfile, mode='w', index=False)

total_num_samples = 0

states_dict = {class_dir: dict() for class_dir in class_dirs}
negative_class_dirs = [class_dir for class_dir in class_dirs]

class_num, curr_batch_size = 0, 0
for class_dir in class_dirs:
    full_class_dir = parser.basedir.rstrip('/ ') + '/' + class_dir
    states_dict[class_dir] = {class_file: [False, False, False] for class_file in anchor_files}

    class_files = os.listdir(full_class_dir)

    num_samples = 0
    while num_samples <= (args.maxperclass if args.maxperclass is not None else len(class_files)) and sum([states_dict[class_dir][class_file][ANCHOR_IDX] for class_file in class_files]) < len(class_files):
        # get anchor image file
        anchor_files = [anchor_file for anchor_file in class_files if not states_dict[class_dir][anchor_file][ANCHOR_IDX]]

        anchor_idx = int(random.random() * len(class_files))
        anchor_file = anchor_files[anchor_idx]
        full_anchor_file = full_class_dir.rstrip('/ ') + '/' + anchor_files[anchor_idx]
        
        states_dict[class_dir][anchor_file][ANCHOR_IDX] = True

        # get positive image file
        positive_files = [positive_file for positive_file in class_files if (positive_file != anchor_file and not states_dict[class_dir][positive_file][POS_IDX])]
        
        positive_idx = anchor_idx
        positive_file = positive_files[positive_idx]
        full_positive_file = full_class_dir.rstrip('/ ') + '/' + positive_files[positive_idx]

        while positive_file == anchor_file:
            positive_idx = anchor_idx
            positive_file = positive_files[positive_idx]
            full_positive_file = full_class_dir.rstrip('/ ') + '/' + positive_files[positive_idx]

        states_dict[class_dir][positive_file][POS_IDX] = True

        # get negative image file
        negative_class_idx = int(random.random() * len(negative_class_dirs))
        negative_class_dir = negative_class_dirs[negative_class_idx]

        full_negative_class_dir = args.basedir.rstrip('/ ') + '/' + negative_class_dir
        negative_files = [negative_file for negative_file in os.listdir(full_negative_class_dir) if not states_dict[negative_class_dir][negative_file][NEG_IDX]]

        negative_idx = int(random.random() * len(negative_files))
        negative_file = negative_files[negative_idx]
        full_negative_file = full_negative_class_dir.rstrip('/ ') + '/' + negative_files[negative_idx]
        
        states_dict[negative_class_dir][negative_file][NEG_IDX] = True

        if sum([states_dict[negative_class_dir][negative_file][NEG_IDX] for negative_file in negative_files]) == len(negative_files):
            negative_class_dirs.remove(negative_class_dir)

        # append to DataFrame, increment num_samples and curr_batch_size, and write to csv (if necessary)
        df.append({'anchor_path': anchor_file, 'positive_path': positive_file, 'negative_path': negative_file, 'class': class_num}, ignore_index=True)
        
        num_samples += 1
        curr_batch_size += 1

        if curr_batch_size == args.batchsize:
            df.to_csv(args.tgtfile, mode='a', index=False, header=False)
            df = pd.DataFrame(columns=['anchor_path', 'positive_path', 'negative_path', 'class'])

            curr_batch_size = 0

            if args.debug:
                print(f'\t...Batch written to {args.tgtfile} [{datetime.datetime.now()}].')

    total_num_samples += num_samples
    class_num += 1

# if any data remaining in the DataFrame, write it to the csv file
df.to_csv(args.tgtfile, mode='w', index=False)
total_num_samples += len(df['anchor_path'])

print(f'Dataset construction complete [{datetime.datetime.now()}]: csv with {total_num_samples} samples stored at {args.tgtfile}.')