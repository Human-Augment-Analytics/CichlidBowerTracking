# code generated using Claude 3.5 Sonnet

import os
import argparse
import csv
import random
from collections import defaultdict
import json
import time

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def save_checkpoint(checkpoint_file, state):
    with open(checkpoint_file, 'w') as f:
        json.dump(state, f)

def generate_triplets(root_dir, output_file, checkpoint_file, batch_size=10000):
    checkpoint = load_checkpoint(checkpoint_file)
    
    if checkpoint:
        class_to_images = {k: set(v) for k, v in checkpoint['class_to_images'].items()}
        classes = checkpoint['classes']
        class_to_label = checkpoint['class_to_label']
        negative_image_count = defaultdict(int, checkpoint['negative_image_count'])
        processed_images = set(checkpoint['processed_images'])
        unused_negatives = set(checkpoint['unused_negatives'])
    else:
        # Collect all image paths and organize them by class
        class_to_images = defaultdict(set)
        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for img in os.listdir(class_path):
                    if img.lower().endswith(('.jpeg', '.jpg', '.png')):
                        class_to_images[class_dir].add(os.path.join(class_path, img))

        # Convert class names to integer labels
        classes = sorted(class_to_images.keys())
        class_to_label = {cls: idx for idx, cls in enumerate(classes)}
        negative_image_count = defaultdict(int)
        processed_images = set()
        unused_negatives = set(img for images in class_to_images.values() for img in images)

    all_images = list(unused_negatives.union(processed_images))
    random.shuffle(all_images)

    # Open CSV file in append mode
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not checkpoint:
            writer.writerow(['anchor', 'positive', 'negative', 'label'])

        batch_start_time = time.time()
        for i, anchor_img in enumerate(all_images):
            if anchor_img in processed_images:
                continue

            anchor_class = os.path.basename(os.path.dirname(anchor_img))
            
            # Find a positive image (different from anchor)
            positive_candidates = [img for img in class_to_images[anchor_class] if img != anchor_img]
            if not positive_candidates:
                continue
            positive_img = random.choice(positive_candidates)

            # Find a negative image (from a different class)
            negative_class_candidates = [c for c in classes if c != anchor_class]
            random.shuffle(negative_class_candidates)
            
            negative_img = None
            for neg_class in negative_class_candidates:
                unused_neg_images = [img for img in class_to_images[neg_class] if img in unused_negatives]
                if unused_neg_images:
                    negative_img = unused_neg_images[0]
                    break
            
            if negative_img is None:
                # If all images have been used as negatives, select the least used one
                all_neg_candidates = [img for c in negative_class_candidates for img in class_to_images[c]]
                negative_img = min(all_neg_candidates, key=lambda x: negative_image_count[x])

            # Update negative image usage count
            negative_image_count[negative_img] += 1
            if negative_img in unused_negatives:
                unused_negatives.remove(negative_img)

            # Add triplet to CSV
            writer.writerow([anchor_img, positive_img, negative_img, class_to_label[anchor_class]])

            processed_images.add(anchor_img)

            # Save checkpoint after each batch
            if (i + 1) % batch_size == 0:
                state = {
                    'class_to_images': {k: list(v) for k, v in class_to_images.items()},
                    'classes': classes,
                    'class_to_label': class_to_label,
                    'negative_image_count': dict(negative_image_count),
                    'processed_images': list(processed_images),
                    'unused_negatives': list(unused_negatives)
                }
                save_checkpoint(checkpoint_file, state)
                
                batch_end_time = time.time()
                print(f"Processed {i + 1} images. Batch processing time: {batch_end_time - batch_start_time:.2f} seconds")
                batch_start_time = time.time()

    # Final checkpoint
    state = {
        'class_to_images': {k: list(v) for k, v in class_to_images.items()},
        'classes': classes,
        'class_to_label': class_to_label,
        'negative_image_count': dict(negative_image_count),
        'processed_images': list(processed_images),
        'unused_negatives': list(unused_negatives)
    }
    save_checkpoint(checkpoint_file, state)

    print(f"Generated triplets for {len(processed_images)} images and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate triplet dataset from ImageNet-1K')
    parser.add_argument('root_dir', type=str, help='Root directory of ImageNet-1K dataset')
    parser.add_argument('output_file', type=str, help='Output CSV file path')
    parser.add_argument('checkpoint_file', type=str, help='Checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size for checkpointing')
    args = parser.parse_args()

    generate_triplets(args.root_dir, args.output_file, args.checkpoint_file, args.batch_size)

if __name__ == '__main__':
    main()