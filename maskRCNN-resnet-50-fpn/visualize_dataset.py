#!/usr/bin/env python3
"""
Visualization tool for the custom segmentation dataset.

This script allows you to:
1. View random samples from the dataset
2. Check specific images by ID
3. Visualize both polygon outlines and mask images

Usage:
    python visualize_dataset.py --dataset_dir path/to/dataset [--sample_id ID]
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import cv2
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize dataset annotations')
    parser.add_argument('--dataset_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--sample_id', type=int, default=None, help='Specific sample ID to visualize (default: random)')
    return parser.parse_args()

def load_annotation(annotations_dir, annotation_file):
    """Load a single annotation file."""
    with open(os.path.join(annotations_dir, annotation_file), 'r') as f:
        return json.load(f)

def visualize_sample(dataset_dir, annotation_file):
    """Visualize a single sample from the dataset."""
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    
    # Load annotation
    annotation = load_annotation(annotations_dir, annotation_file)
    
    # Load image
    image_path = os.path.join(dataset_dir, annotation['image_path'])
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    image = np.array(Image.open(image_path).convert('RGB'))
    height, width = image.shape[:2]
    
    # Create figure with 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Left: Show image with polygon outlines
    axes[0].imshow(image)
    axes[0].set_title("Image with Polygon Annotations", fontsize=12)
    
    # Create a color map for classes
    classes = set(mask['label'] for mask in annotation['masks'])
    class_colors = {cls: plt.cm.tab10(i % 10) for i, cls in enumerate(sorted(classes))}
    
    # Draw polygons
    for mask_info in annotation['masks']:
        label = mask_info['label']
        points = mask_info['points']
        color = class_colors[label]
        
        # Draw polygon outline
        if points and len(points) >= 3:
            poly = Polygon(points, fill=False, edgecolor=color, linewidth=2)
            axes[0].add_patch(poly)
            
            # Add label text
            axes[0].text(
                points[0][0], points[0][1], 
                label, 
                fontsize=9,
                color='white',
                bbox=dict(facecolor=color, alpha=0.7, boxstyle='round,pad=0.2')
            )
    
    axes[0].set_xlim(0, width)
    axes[0].set_ylim(height, 0)  # Reverse Y axis to match image coordinates
    axes[0].axis('off')
    
    # Right: Show all instance masks as a combined color image
    combined_mask = np.zeros((height, width, 4), dtype=np.float32)
    
    # Process each mask
    for i, mask_info in enumerate(annotation['masks']):
        label = mask_info['label']
        mask_path = os.path.join(dataset_dir, mask_info['mask_path'])
        
        # Load or create mask
        if os.path.exists(mask_path):
            # Load mask from file
            mask = np.array(Image.open(mask_path).convert('L'))
            mask = (mask > 0).astype(np.float32)
        else:
            # Create mask from polygon
            points = mask_info['points']
            mask = np.zeros((height, width), dtype=np.float32)
            
            if points and len(points) >= 3:
                pts = np.array(points, dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 1)
        
        # Add this mask to the combined visualization
        color = class_colors[label]
        for c in range(3):
            combined_mask[:, :, c] += mask * color[c] * 0.7  # Slightly transparent
        
        # Add opacity based on mask
        combined_mask[:, :, 3] += mask * 0.3
    
    # Normalize and ensure alpha doesn't exceed 1.0
    if combined_mask[:, :, :3].max() > 0:
        # Normalize RGB channels
        max_value = combined_mask[:, :, :3].max()
        combined_mask[:, :, :3] /= max_value
    
    combined_mask[:, :, 3] = np.minimum(combined_mask[:, :, 3], 1.0)
    
    # Show combined mask over a gray background
    gray_bg = np.ones((height, width, 3)) * 0.9  # Light gray
    mask_vis = combined_mask[:, :, :3] * combined_mask[:, :, 3:] + \
              gray_bg * (1 - combined_mask[:, :, 3:])
    
    axes[1].imshow(mask_vis)
    axes[1].set_title("Instance Masks", fontsize=12)
    axes[1].axis('off')
    
    # Show information about the sample
    plt.suptitle(f"Sample: {os.path.splitext(annotation_file)[0]}", fontsize=14)
    
    # Add a small legend for classes
    legend_elements = [plt.Line2D([0], [0], color=class_colors[cls], lw=4, label=cls) 
                      for cls in sorted(classes)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=min(5, len(classes)), 
              bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for the legend
    plt.show()

def main():
    args = parse_args()
    dataset_dir = args.dataset_dir
    
    # Validate dataset directory
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    if not os.path.exists(annotations_dir):
        print(f"Error: Annotations directory not found: {annotations_dir}")
        return
    
    # Get all annotation files
    annotation_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.json')])
    
    if not annotation_files:
        print("Error: No annotation files found in the dataset")
        return
    
    print(f"Found {len(annotation_files)} annotation files in the dataset")
    
    # Select annotation file to visualize
    if args.sample_id is not None:
        # Show specific sample
        if args.sample_id < 0 or args.sample_id >= len(annotation_files):
            print(f"Error: Sample ID {args.sample_id} is out of range (0-{len(annotation_files)-1})")
            return
        annotation_file = annotation_files[args.sample_id]
    else:
        # Show random sample
        annotation_file = random.choice(annotation_files)
    
    print(f"Visualizing sample: {annotation_file}")
    visualize_sample(dataset_dir, annotation_file)

if __name__ == "__main__":
    main()