#!/usr/bin/env python3
"""
Generate mask images from polygon points in annotation files.

This script:
1. Reads annotation JSON files
2. Creates binary mask images from polygon points
3. Saves them to the masks directory

Usage:
    python generate_masks.py --dataset_dir path/to/dataset [--force]
"""

import os
import json
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Generate mask images from polygon points')
    parser.add_argument('--dataset_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--force', action='store_true', help='Force regeneration of existing masks')
    return parser.parse_args()

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def generate_mask_from_polygon(points, height, width):
    """Generate a binary mask from polygon points."""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if not points or len(points) < 3:
        print("Warning: Polygon has fewer than 3 points, skipping")
        return mask
    
    # Convert points to the format expected by fillPoly
    pts = np.array(points, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # Fill polygon with white (255)
    cv2.fillPoly(mask, [pts], 255)
    
    return mask

def process_annotations(dataset_dir, force=False):
    """Process all annotations and generate mask images."""
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    masks_dir = os.path.join(dataset_dir, 'masks')
    
    # Ensure masks directory exists
    ensure_dir(masks_dir)
    
    # Get list of annotation files
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
    
    if not annotation_files:
        print("No annotation files found")
        return
    
    print(f"Processing {len(annotation_files)} annotation files...")
    
    masks_generated = 0
    masks_skipped = 0
    
    # Process each annotation file
    for ann_file in tqdm(annotation_files):
        with open(os.path.join(annotations_dir, ann_file), 'r') as f:
            try:
                annotation = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Could not parse {ann_file} as JSON, skipping")
                continue
        
        # Get image dimensions
        height = annotation.get('height')
        width = annotation.get('width')
        
        if not height or not width:
            # If dimensions aren't in the annotation, try to get them from the image
            image_path = os.path.join(dataset_dir, annotation.get('image_path', ''))
            if os.path.exists(image_path):
                img = Image.open(image_path)
                width, height = img.size
            else:
                print(f"Error: Cannot determine dimensions for {ann_file}, skipping")
                continue
        
        # Process each mask in the annotation
        for i, mask_info in enumerate(annotation.get('masks', [])):
            # Get mask path from annotation
            mask_rel_path = mask_info.get('mask_path')
            if not mask_rel_path:
                print(f"Warning: Mask in {ann_file} has no path specified, skipping")
                continue
            
            mask_full_path = os.path.join(dataset_dir, mask_rel_path)
            
            # Check if mask already exists and force flag is not set
            if os.path.exists(mask_full_path) and not force:
                masks_skipped += 1
                continue
            
            # Get polygon points
            points = mask_info.get('points', [])
            
            # Generate mask from polygon points
            mask = generate_mask_from_polygon(points, height, width)
            
            # Save mask
            os.makedirs(os.path.dirname(mask_full_path), exist_ok=True)
            Image.fromarray(mask).save(mask_full_path)
            masks_generated += 1
    
    print(f"Done! Generated {masks_generated} masks, skipped {masks_skipped} existing masks.")

def main():
    args = parse_args()
    process_annotations(args.dataset_dir, args.force)

if __name__ == "__main__":
    main()