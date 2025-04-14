#!/usr/bin/env python3
"""
Analyze the segmentation dataset and generate statistics.

This script provides insights about:
1. Number of images and annotations
2. Class distribution
3. Image sizes
4. Number of instances per image
5. Average polygon sizes

Usage:
    python dataset_stats.py --dataset_dir path/to/dataset [--visualize]
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze segmentation dataset statistics')
    parser.add_argument('--dataset_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    return parser.parse_args()

def analyze_dataset(dataset_dir, visualize=False):
    """Analyze the dataset and generate statistics."""
    # Initialize data structures for statistics
    class_counts = Counter()
    image_sizes = []
    instances_per_image = []
    polygon_points_count = defaultdict(list)
    polygon_areas = defaultdict(list)
    
    # Get annotation files
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
    
    print(f"Found {len(annotation_files)} annotation files")
    
    # Process each annotation file
    for ann_file in annotation_files:
        # Load annotation
        with open(os.path.join(annotations_dir, ann_file), 'r') as f:
            try:
                annotation = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {ann_file}, skipping")
                continue
        
        # Extract image size
        width = annotation.get('width')
        height = annotation.get('height')
        
        if width and height:
            image_sizes.append((width, height))
        
        # Count masks (instances) in this image
        masks = annotation.get('masks', [])
        instances_per_image.append(len(masks))
        
        # Process each mask
        for mask_info in masks:
            # Get class
            label = mask_info.get('label')
            if label:
                class_counts[label] += 1
            
            # Get polygon data
            points = mask_info.get('points', [])
            if points:
                # Count number of points
                if label:
                    polygon_points_count[label].append(len(points))
                
                # Calculate polygon area using shoelace formula
                points_array = np.array(points)
                if len(points_array) >= 3:  # Need at least 3 points for area
                    x = points_array[:, 0]
                    y = points_array[:, 1]
                    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    if label:
                        polygon_areas[label].append(area)
    
    # Print statistics
    print("\n===== Dataset Statistics =====")
    print(f"Total number of annotation files: {len(annotation_files)}")
    
    # Class distribution
    print("\nClass distribution:")
    for label, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count} instances")
    
    # Image sizes
    if image_sizes:
        widths, heights = zip(*image_sizes)
        avg_width = sum(widths) / len(widths)
        avg_height = sum(heights) / len(heights)
        print(f"\nAverage image size: {avg_width:.1f} × {avg_height:.1f} pixels")
        print(f"Size range: Width [{min(widths)}-{max(widths)}], Height [{min(heights)}-{max(heights)}]")
    
    # Instances per image
    if instances_per_image:
        avg_instances = sum(instances_per_image) / len(instances_per_image)
        print(f"\nAverage instances per image: {avg_instances:.2f}")
        print(f"Max instances in an image: {max(instances_per_image)}")
    
    # Average polygon complexity (number of points)
    print("\nAverage polygon points per class:")
    for label, points in sorted(polygon_points_count.items()):
        avg_points = sum(points) / len(points)
        print(f"  {label}: {avg_points:.1f} points")
    
    # Average polygon size (area)
    print("\nAverage polygon area per class:")
    for label, areas in sorted(polygon_areas.items()):
        avg_area = sum(areas) / len(areas)
        print(f"  {label}: {avg_area:.1f} sq pixels")
    
    # Visualize statistics if requested
    if visualize:
        visualize_statistics(class_counts, image_sizes, instances_per_image, polygon_areas)

def visualize_statistics(class_counts, image_sizes, instances_per_image, polygon_areas):
    """Generate visualizations for dataset statistics."""
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Class distribution
    ax1 = fig.add_subplot(2, 2, 1)
    classes = [label for label in class_counts.keys()]
    counts = [count for count in class_counts.values()]
    
    ax1.bar(classes, counts, color='skyblue')
    ax1.set_title('Class Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Instances')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Image size distribution
    if image_sizes:
        ax2 = fig.add_subplot(2, 2, 2)
        widths, heights = zip(*image_sizes)
        
        ax2.scatter(widths, heights, alpha=0.5, s=20)
        ax2.set_title('Image Size Distribution')
        ax2.set_xlabel('Width (pixels)')
        ax2.set_ylabel('Height (pixels)')
        ax2.grid(True, alpha=0.3)
    
    # 3. Instances per image histogram
    if instances_per_image:
        ax3 = fig.add_subplot(2, 2, 3)
        
        ax3.hist(instances_per_image, bins=max(10, max(instances_per_image)), color='lightgreen', alpha=0.7)
        ax3.set_title('Instances per Image')
        ax3.set_xlabel('Number of Instances')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
    
    # 4. Polygon area by class (boxplot)
    if polygon_areas:
        ax4 = fig.add_subplot(2, 2, 4)
        
        # Prepare data for boxplot
        areas_data = []
        labels = []
        
        for label, areas in sorted(polygon_areas.items()):
            areas_data.append(areas)
            labels.append(label)
        
        ax4.boxplot(areas_data, vert=True, patch_artist=True)
        ax4.set_xticklabels(labels, rotation=45)
        ax4.set_title('Polygon Area by Class')
        ax4.set_ylabel('Area (sq pixels)')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    analyze_dataset(args.dataset_dir, args.visualize)

if __name__ == "__main__":
    main()