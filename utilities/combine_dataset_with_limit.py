#!/usr/bin/env python3
import os
import json
import shutil
from pathlib import Path
import argparse

def combine_datasets(input_dirs, output_dir, limit_per_dataset=None):
    """
    Combine multiple datasets into a master dataset.
    
    Args:
        input_dirs (list): List of paths to dataset directories
        output_dir (str): Path to output master dataset directory
        limit_per_dataset (int, optional): Maximum number of image/annotation pairs to include per dataset
    """
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    
    # Create a master dataset info file
    master_dataset_info = {
        "name": "master_dataset",
        "description": "Combined dataset from multiple sources",
        "child_datasets": [],
        "file_mappings": {}
    }
    
    # Process each input directory as a dataset
    for child_dataset_path in input_dirs:
        # Extract dataset name from path
        child_dataset = os.path.basename(child_dataset_path.rstrip('/'))
        
        # Skip if not a directory
        if not os.path.isdir(child_dataset_path):
            print(f"Warning: {child_dataset_path} is not a directory, skipping")
            continue
            
        print(f"Processing dataset: {child_dataset}")
        
        # Add to child datasets list
        master_dataset_info["child_datasets"].append(child_dataset)
        
        # Process dataset_info.json if it exists
        dataset_info_path = os.path.join(child_dataset_path, "dataset_info.json")
        if os.path.exists(dataset_info_path):
            try:
                with open(dataset_info_path, 'r') as f:
                    child_info = json.load(f)
                print(f"Loaded dataset info for {child_dataset}")
            except Exception as e:
                print(f"Warning: Could not load dataset info for {child_dataset}: {e}")
                child_info = {"name": child_dataset}
        else:
            child_info = {"name": child_dataset}
        
        # Get list of annotation files and apply limit if needed
        annotation_files = []
        annotations_dir = os.path.join(child_dataset_path, "annotations")
        if os.path.exists(annotations_dir):
            annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
            
            # Apply limit if specified
            if limit_per_dataset is not None and len(annotation_files) > limit_per_dataset:
                annotation_files = annotation_files[:limit_per_dataset]
                print(f"Limiting dataset {child_dataset} to {limit_per_dataset} annotations")
        
        # Keep track of included images for limiting
        included_images = set()
        
        # Process annotations
        for filename in annotation_files:
            # Create new filename with dataset prefix
            new_filename = f"{child_dataset}_{filename}"
            
            # Copy and rename file
            src_path = os.path.join(annotations_dir, filename)
            dst_path = os.path.join(output_dir, "annotations", new_filename)
            
            # Copy file
            shutil.copy2(src_path, dst_path)
            
            # Update annotation content to reflect new image filename
            try:
                with open(dst_path, 'r') as f:
                    annotation_data = json.load(f)
                
                # Check if annotation has image_path field and update it
                if isinstance(annotation_data, dict):
                    # Record original values for mapping
                    orig_image_path = None
                    orig_mask_paths = []
                    
                    # Handle image_path field
                    if "image_path" in annotation_data:
                        orig_image_path = annotation_data["image_path"]
                        # Extract just the filename from the path
                        image_filename = os.path.basename(orig_image_path)
                        # Add to included images
                        included_images.add(image_filename)
                        # Create new image filename with dataset prefix
                        new_image_filename = f"{child_dataset}_{image_filename}"
                        # Update with new path but same directory structure
                        annotation_data["image_path"] = os.path.join(os.path.dirname(orig_image_path), new_image_filename)
                    
                    # Handle masks if present
                    if "masks" in annotation_data and isinstance(annotation_data["masks"], list):
                        for i, mask in enumerate(annotation_data["masks"]):
                            if "mask_path" in mask:
                                orig_mask_path = mask["mask_path"]
                                orig_mask_paths.append(orig_mask_path)
                                # Extract just the filename from the path
                                mask_filename = os.path.basename(orig_mask_path)
                                # Create new mask filename with dataset prefix
                                new_mask_filename = f"{child_dataset}_{mask_filename}"
                                # Update with new path but same directory structure
                                annotation_data["masks"][i]["mask_path"] = os.path.join(os.path.dirname(orig_mask_path), new_mask_filename)
                    
                    # Handle any other generic image field
                    if "image" in annotation_data:
                        orig_image = annotation_data["image"]
                        # Add to included images
                        included_images.add(orig_image)
                        # Create new image filename with dataset prefix
                        new_image = f"{child_dataset}_{orig_image}"
                        # Update annotation
                        annotation_data["image"] = new_image
                    
                    # Add to file mappings
                    master_dataset_info["file_mappings"][new_filename] = {
                        "original_dataset": child_dataset,
                        "original_filename": filename,
                        "original_image_path": orig_image_path,
                        "original_mask_paths": orig_mask_paths
                    }
                
                # Write updated annotation back to file
                with open(dst_path, 'w') as f:
                    json.dump(annotation_data, f, indent=2)
                    
            except Exception as e:
                print(f"Warning: Could not update annotation {new_filename}: {e}")
        
        # Process images (only those referenced in annotations if limit is applied)
        images_dir = os.path.join(child_dataset_path, "images")
        if os.path.exists(images_dir):
            for filename in os.listdir(images_dir):
                # Skip non-image files (simple check)
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    continue
                
                # If limit is applied, only include images referenced in annotations
                if limit_per_dataset is not None and filename not in included_images:
                    continue
                    
                # Create new filename with dataset prefix
                new_filename = f"{child_dataset}_{filename}"
                
                # Copy and rename file
                src_path = os.path.join(images_dir, filename)
                dst_path = os.path.join(output_dir, "images", new_filename)
                
                # Copy file
                shutil.copy2(src_path, dst_path)
        
        # Process masks if they exist (only those referenced in annotations if limit is applied)
        masks_dir = os.path.join(child_dataset_path, "masks")
        if os.path.exists(masks_dir):
            mask_files_to_include = set()
            # Get mask filenames referenced in annotations
            for mapping in master_dataset_info["file_mappings"].values():
                if mapping["original_dataset"] == child_dataset:
                    for mask_path in mapping["original_mask_paths"]:
                        if mask_path:
                            mask_files_to_include.add(os.path.basename(mask_path))
            
            # Copy only the needed masks
            for filename in os.listdir(masks_dir):
                # Skip non-image files (simple check)
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    continue
                    
                # If limit is applied, only include masks referenced in annotations
                if limit_per_dataset is not None and filename not in mask_files_to_include:
                    continue
                    
                # Create new filename with dataset prefix
                new_filename = f"{child_dataset}_{filename}"
                
                # Copy and rename file
                src_path = os.path.join(masks_dir, filename)
                dst_path = os.path.join(output_dir, "masks", new_filename)
                
                # Copy file
                shutil.copy2(src_path, dst_path)
    
    # Write master dataset info
    with open(os.path.join(output_dir, "dataset_info.json"), 'w') as f:
        json.dump(master_dataset_info, f, indent=2)
    
    print(f"Master dataset created at: {output_dir}")
    print(f"Combined {len(master_dataset_info['child_datasets'])} datasets")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine multiple datasets into a master dataset.")
    parser.add_argument("--input-dirs", nargs='+', required=True, help="Paths to dataset directories")
    parser.add_argument("--output-dir", required=True, help="Path to output master dataset directory")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of image/annotation pairs per dataset (default: no limit)")
    
    args = parser.parse_args()
    
    combine_datasets(args.input_dirs, args.output_dir, args.limit)