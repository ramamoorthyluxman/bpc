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
    
    # Points validation is now handled in the process_annotations function
    # Convert points to the format expected by fillPoly
    pts = np.array(points, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # Fill polygon with white (255)
    cv2.fillPoly(mask, [pts], 255)
    
    return mask

def get_base_filename(image_path):
    """Extract the base filename from the image path without extension."""
    return os.path.splitext(os.path.basename(image_path))[0]

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
    masks_removed = 0
    
    # Process each annotation file
    for ann_file in tqdm(annotation_files):
        ann_file_path = os.path.join(annotations_dir, ann_file)
        with open(ann_file_path, 'r') as f:
            try:
                annotation = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Could not parse {ann_file} as JSON, skipping")
                continue
        
        # Get image dimensions
        height = annotation.get('height')
        width = annotation.get('width')
        
        # Get image path
        image_path = annotation.get('image_path', '')
        if not image_path:
            print(f"Error: No image path specified in {ann_file}, skipping")
            continue
            
        # Get base filename for generating mask paths
        base_filename = get_base_filename(image_path)
        
        if not height or not width:
            # If dimensions aren't in the annotation, try to get them from the image
            full_image_path = os.path.join(dataset_dir, image_path)
            if os.path.exists(full_image_path):
                img = Image.open(full_image_path)
                width, height = img.size
            else:
                print(f"Error: Cannot determine dimensions for {ann_file}, skipping")
                continue
        
        # Track which masks to remove
        masks_to_remove = []
        
        # Process each mask in the annotation
        for i, mask_info in enumerate(annotation.get('masks', [])):
            # Get polygon points
            points = mask_info.get('points', [])
            
            # Check if the polygon has enough points
            if not points or len(points) < 3:
                print(f"Warning: Polygon has fewer than 3 points in {ann_file}, mask index {i}, removing this mask")
                masks_to_remove.append(i)
                
                # If there's an existing mask file, remove it
                mask_rel_path = mask_info.get('mask_path')
                if mask_rel_path:
                    mask_full_path = os.path.join(dataset_dir, mask_rel_path)
                    if os.path.exists(mask_full_path):
                        try:
                            os.remove(mask_full_path)
                            print(f"  Removed mask file: {mask_full_path}")
                        except Exception as e:
                            print(f"  Error removing mask file {mask_full_path}: {e}")
                
                masks_removed += 1
                continue
            
            # Get mask path from annotation, or generate one if not present
            mask_rel_path = mask_info.get('mask_path')
            if not mask_rel_path:
                # Create a mask path based on the image filename
                mask_rel_path = f"masks/{base_filename}_{i}.png"
                mask_info['mask_path'] = mask_rel_path  # Update the annotation with the new path
            
            mask_full_path = os.path.join(dataset_dir, mask_rel_path)
            
            # Check if mask already exists and force flag is not set
            if os.path.exists(mask_full_path) and not force:
                masks_skipped += 1
                continue
            
            # Generate mask from polygon points
            mask = generate_mask_from_polygon(points, height, width)
            
            # Save mask
            os.makedirs(os.path.dirname(mask_full_path), exist_ok=True)
            Image.fromarray(mask).save(mask_full_path)
            masks_generated += 1
        
        # Remove masks with insufficient points
        if masks_to_remove:
            # We need to remove from the end to avoid index shifting
            for idx in sorted(masks_to_remove, reverse=True):
                annotation['masks'].pop(idx)
            print(f"  Removed {len(masks_to_remove)} masks from {ann_file}")
        
        # Update the annotation file
        if force or masks_generated > 0 or masks_removed > 0:
            with open(ann_file_path, 'w') as f:
                json.dump(annotation, f, indent=2)
            print(f"  Updated annotation file: {ann_file}")
    
    print(f"Done! Generated {masks_generated} masks, skipped {masks_skipped} existing masks, removed {masks_removed} invalid masks.")

def main():
    args = parse_args()
    process_annotations(args.dataset_dir, args.force)

if __name__ == "__main__":
    main()