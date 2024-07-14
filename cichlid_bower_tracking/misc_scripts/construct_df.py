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

df = pd.DataFrame(columns=['anchor_path', 'positive_path', 'negative_path'])
df.to_csv(args.tgtfile, mode='w', index=False)

total_num_samples = 0

states_dict = {class_dir: dict() for class_dir in class_dirs}
curr_batch_size = 0
for class_dir in class_dirs:
    full_class_dir = parser.basedir.rstrip('/ ') + '/' + class_dir

    anchor_files = os.listdir(full_class_dir)
    positive_files = [class_file for class_file in anchor_files]

    states_dict[class_dir] = {class_file: [False, False, False] for class_file in anchor_files}

    num_samples = 0
    while num_samples <= (args.maxperclass if args.maxperclass is not None else len(anchor_files)) and len(anchor_files) > 0:
        # get anchor image file
        anchor_idx = int(random.random() * len(class_files))

        anchor_file = full_class_dir.rstrip('/ ') + '/' + anchor_files[anchor_idx]
        states_dict[class_dir][anchor_file][ANCHOR_IDX] = True

        anchor_files.remove(anchor_file)

        # get positive image file
        positive_idx = anchor_idx
        positive_file = full_class_dir.rstrip('/ ') + '/' + positive_files[positive_idx]

        while positive_file == anchor_file:
            positive_idx = int(random.random() * len(class_files))
            positive_file = full_class_dir.rstrip('/ ') + '/' + positive_files[positive_idx]

        states_dict[class_dir][positive_file][POS_IDX] = True

        # get negative image file
        states_outer_keys = list(states_dict.keys())

        negative_class_idx = int(random.random() * len(states_outer_keys))
        negative_class_dir = states_outer_keys[negative_class_idx]

        full_negative_class_dir = args.basedir.rstrip('/ ') + '/' + negative_class_dir
        negative_class_files = os.listdir(full_negative_class_dir)

        negative_idx = int(random.random() * len(negative_class_files))

        negative_file = full_negative_class_dir.rstrip('/ ') + '/' + negative_class_files[negative_idx]
        states_dict[negative_class_dir][negative_file][NEG_IDX] = True

        # append to DataFrame, increment num_samples and curr_batch_size, and write to csv (if necessary)
        df.append({'anchor_path': anchor_file, 'positive_path': positive_file, 'negative_path': negative_file}, ignore_index=True)
        num_samples += 1
        curr_batch_size += 1

        if curr_batch_size == args.batchsize:
            df.to_csv(args.tgtfile, mode='a', index=False, header=False)
            df = pd.DataFrame(columns=['anchor_path', 'positive_path', 'negative_path'])

            curr_batch_size = 0

            if args.debug:
                print(f'\t...Batch written to {args.tgtfile} [{datetime.datetime.now()}].')

    total_num_samples += num_samples

# if any data remaining in the DataFrame, write it to the csv file
df.to_csv(args.tgtfile, mode='w', index=False)
total_num_samples += len(df['anchor_path'])

print(f'Dataset construction complete [{datetime.datetime.now()}]: csv with {total_num_samples} samples stored at {args.tgtfile}.')