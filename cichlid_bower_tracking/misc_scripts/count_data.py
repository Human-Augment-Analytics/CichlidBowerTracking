from typing import List, Dict, Union 
import argparse, os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Accept arguments that refine the counting process.')

parser.add_argument('base_dir', type=str, help='The base directory of the dataset.')
parser.add_argument('--count_type', '-t', type=str, choices=['videos', 'frames'],  help='Specifies what specifically should be counted (unique videos, individual frames, etc.).')
parser.add_argument('--split', '-s', type=str, choices=['train', 'test', 'valid'],  help='Specifies which split the count should be performed on.')
parser.add_argument('--plot', '-p', action='store_true', help='Flag specifying that the results of the count should be plotted and displayed; only useful if no split is specified.')

args = parser.parse_args()

DATASET_PATH = args.base_dir
if not os.path.exists(DATASET_PATH):
    print(f'Invalid Base Directory: {DATASET_PATH} does not exist.')

def unique_videos(split: str=None) -> Union[Dict[str, List[str]], List[str]]:
    if split == None:
        uniques_dict = dict()

        subdirs_list = os.listdir(DATASET_PATH)
        for subdir in subdirs_list:
            if subdir not in list(uniques_dict.keys()):
                uniques_dict[subdir] = []

            subdir_path = DATASET_PATH.rstrip(' /') + '/' + subdir

            frame_files_list = os.listdir(subdir_path)
            for frame_file in frame_files_list:
                video_file = '_'.join(frame_file.split('_')[:4])

                if video_file not in list(uniques_dict[subdir]):
                    uniques_dict[subdir].append(f'{video_file}')

        return uniques_dict

    else:
        uniques_list = []

        frame_files_list = os.listdir(DATASET_PATH.rstrip(' /') + '/' + split)
        for frame_file in frame_files_list:
            video_file = '_'.join(frame_file.split('_')[:4])

            if video_file not in uniques_list:
                uniques_list.append(f'{video_file}')

        return uniques_list

def count_uniques(uniques: Union[Dict[str, List[str]], List[str]]) -> Union[Dict[str, int], int]:
    if isinstance(uniques, dict):
        counts_dict = dict()

        for subdir in list(uniques.keys()):
            n_uniques = len(uniques[subdir])

            counts_dict[subdir] = n_uniques

        return counts_dict
    
    else:
        n_uniques = len(uniques)

        return n_uniques

def count_frames(split: str=None) -> Union[Dict[str, int], int]:
    if split == None:
        counts_dict = dict()

        subdirs_list = os.listdir(DATASET_PATH)
        for subdir in subdirs_list:
            subdir_path = DATASET_PATH.rstrip(' /') + '/' + subdir

            frame_files_list = os.listdir(subdir_path)
            n_frames = len(frame_files_list)

            counts_dict[subdir] = n_frames
        return counts_dict

    else:
        frames_list = os.listdir(DATASET_PATH.rstrip(' /') + '/' + split)
        n_frames = len(frames_list)

        return n_frames

def count_all(split: str=None) -> Union[Dict[str, Dict[str, int]], Dict[str, int]]:
    if split == None:
        outer_dict = dict()

        subdirs_list = os.listdir(DATASET_PATH)
        for subdir in subdirs_list:
            if subdir not in list(outer_dict.keys()):
                outer_dict[subdir] = dict()

            subdir_path = DATASET_PATH.rstrip(' /') + '/' + subdir
            frame_files_list = os.listdir(subdir_path)

            for frame_file in frame_files_list:
                video_file = '_'.join(frame_file.split('_')[:4])

                if video_file not in list(outer_dict[subdir].keys()):
                    outer_dict[subdir][video_file] = 1
                else:
                    outer_dict[subdir][video_file] += 1

        return outer_dict
    else:
        inner_dict = dict()

        frame_files_list = os.listdir(DATASET_PATH.rstrip(' /') + '/' + split)
        for frame_file in frame_files_list:
            video_file = '_'.join(frame_file.split('_')[:4])

            if video_file not in list(inner_dict.keys()):
                inner_dict[video_file] = 1
            else:
                inner_dict[video_file] += 1

        return inner_dict

if args.count_type is None:
    print('\ncounting...')
    all_dict = count_all(split=args.split)

    if not args.split:
        for subdir, inner_dict in all_dict.items():
            print(f'\t{subdir}')

            for video_file, n_frames in inner_dict.items():
                msg = f'\t\t{video_file:20s} -> {n_frames:3d}'

                print(msg)

    else:
        for video_file, n_frames in all_dict.items():
            msg = f'\t{args.split:10s}: {video_file:20s} -> {n_frames:3d}'

            print(msg)

elif args.count_type == 'videos':
    print('\ncounting unique videos...')
    uniques = unique_videos(split=args.split)

    if not args.split:
        counts_dict = count_uniques(uniques=uniques)

        for subdir in list(uniques.keys()):
            msg = f'\t{subdir:10s}: {[f"{video_file:20s}" for video_file in uniques[subdir]]} -> {counts_dict[subdir]:3d}'

            print(msg)

    else:
        n_uniques = count_uniques(uniques=uniques)
        msg = f'\t{args.split:10s}: {[f"{video_file:20s}" for video_file in uniques]} -> {n_uniques:3d}'

        print(msg)

else:
    print('\ncounting frames...')
    frames = count_frames(split=args.split)

    if not args.split:
        for subdir, n_frames in frames.items():
            msg = f'\t{subdir:10s} -> {n_frames:3d}'

            print(msg)

    else:
        msg = f'\t{args.split:10s} -> {frames:3d}'

        print(msg)

