# code generated using Claude 3.5 Sonnet

import argparse
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def validate_triplets(csv_file):
    anchor_counts = defaultdict(int)
    positive_counts = defaultdict(int)
    negative_counts = defaultdict(int)
    negative_class_counts = defaultdict(int)
    all_images = set()
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            anchor, positive, negative, _ = row
            anchor_counts[anchor] += 1
            positive_counts[positive] += 1
            negative_counts[negative] += 1
            negative_class = negative.split('/')[-2]  # Assumes path format /path/to/class/image.jpg
            negative_class_counts[negative_class] += 1
            all_images.update([anchor, positive, negative])
    
    # Condition 1: Each image used exactly once as an anchor
    anchor_usage = list(anchor_counts.values())
    avg_anchor_usage = np.mean(anchor_usage)
    print(f"Condition 1: Average anchor usage: {avg_anchor_usage:.2f}")
    print(f"             Min: {min(anchor_usage)}, Max: {max(anchor_usage)}")
    
    # Condition 2: Each image used exactly once as a positive
    positive_usage = list(positive_counts.values())
    avg_positive_usage = np.mean(positive_usage)
    print(f"Condition 2: Average positive usage: {avg_positive_usage:.2f}")
    print(f"             Min: {min(positive_usage)}, Max: {max(positive_usage)}")
    
    # Condition 3: Each class used at least once as a negative
    min_class_usage = min(negative_class_counts.values())
    print(f"Condition 3: Minimum class usage as negative: {min_class_usage}")
    
    # Condition 4: Balanced selection of images as negatives
    negative_usage = list(negative_counts.values())
    avg_negative_usage = np.mean(negative_usage)
    print(f"Condition 4: Average negative usage: {avg_negative_usage:.2f}")
    print(f"             Min: {min(negative_usage)}, Max: {max(negative_usage)}")
    
    # Count images never used as negatives
    unused_as_negative = len(all_images) - len(negative_counts)
    print(f"             Images never used as negatives: {unused_as_negative}")
    
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    
    # Anchor usage plot
    axs[0, 0].hist(anchor_usage, bins=20)
    axs[0, 0].set_title('Anchor Usage Distribution')
    axs[0, 0].set_xlabel('Times Used as Anchor')
    axs[0, 0].set_ylabel('Number of Images')
    
    # Positive usage plot
    axs[0, 1].hist(positive_usage, bins=20)
    axs[0, 1].set_title('Positive Usage Distribution')
    axs[0, 1].set_xlabel('Times Used as Positive')
    axs[0, 1].set_ylabel('Number of Images')
    
    # Negative class usage plot
    class_usage = list(negative_class_counts.values())
    axs[1, 0].bar(range(len(class_usage)), sorted(class_usage))
    axs[1, 0].set_title('Negative Class Usage')
    axs[1, 0].set_xlabel('Class Index (sorted)')
    axs[1, 0].set_ylabel('Times Used as Negative')
    
    # Negative image usage plot (including unused count)
    bins = np.arange(max(negative_usage) + 2) - 0.5
    counts, _, _ = axs[1, 1].hist(negative_usage, bins=bins, label='Used as Negative')
    axs[1, 1].bar(-0.5, unused_as_negative, width=1, label='Never Used as Negative', color='red', alpha=0.7)
    axs[1, 1].set_title('Negative Usage Distribution')
    axs[1, 1].set_xlabel('Times Used as Negative')
    axs[1, 1].set_ylabel('Number of Images')
    axs[1, 1].legend()
    
    # Adjust x-axis for better visibility of the 'Never Used' bar
    axs[1, 1].set_xlim(-1, max(bins))
    
    plt.tight_layout()
    plt.savefig('triplet_validation_plots.png')
    print("Plots saved as 'triplet_validation_plots.png'")

def main():
    parser = argparse.ArgumentParser(description='Validate triplet dataset')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing triplets')
    args = parser.parse_args()

    validate_triplets(args.csv_file)

if __name__ == '__main__':
    main()