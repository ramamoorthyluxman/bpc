"""
Data preparation script for SAM fine-tuning.
This script converts polygon annotations to masks and organizes the dataset.
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for SAM fine-tuning")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to dataset directory containing annotations and images folders")
    parser.add_argument("--output_dir", type=str, default="./sam_dataset",
                        help="Output directory for processed data")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize masks during conversion")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to process (None for all)")
    return parser.parse_args()

def polygon_to_mask(polygon, img_height, img_width):
    """Convert polygon points to binary mask."""
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    # Convert points to integer and reshape
    polygon = np.array(polygon, dtype=np.int32)
    # Fill the polygon
    cv2.fillPoly(mask, [polygon], 1)
    return mask

def visualize_mask(image, mask, title="Mask Visualization"):
    """Visualize image with overlaid mask."""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.title("Overlaid")
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def process_annotations(data_dir, output_dir, visualize=False, num_samples=None):
    """Process annotations and convert to masks."""
    annotations_dir = os.path.join(data_dir, "annotations")
    images_dir = os.path.join(data_dir, "images")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    
    # List annotation files
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]
    
    if num_samples is not None:
        annotation_files = annotation_files[:num_samples]
    
    print(f"Processing {len(annotation_files)} annotation files...")
    
    # Dictionary to store dataset info
    dataset_info = {
        "images": [],
        "categories": set(),
        "num_masks": 0
    }
    
    for annotation_file in tqdm(annotation_files):
        try:
            # Load annotation
            with open(os.path.join(annotations_dir, annotation_file), 'r') as f:
                annotation = json.load(f)
            
            # Get image path and file name
            # Check which key exists in the annotation and use that one
            if "imagePath" in annotation:
                image_filename = os.path.basename(annotation["imagePath"])                
                image_path = os.path.join(images_dir, os.path.basename(annotation["imagePath"]))
            elif "image_path" in annotation:
                image_filename = os.path.basename(annotation["image_path"])
                image_path = os.path.join(images_dir, os.path.basename(annotation["image_path"]))
            else:
                print("Error: No image path key found in annotation (tried both 'imagePath' and 'image_path')")
                image_path = None
                image_filename = None
            
            
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, skipping...")
                continue
            
            # Load image
            image = np.array(Image.open(image_path).convert("RGB"))
            h, w = annotation["height"], annotation["width"]
            
            # Process each shape (polygon annotation)
            masks_info = []
            
            for i, shape in enumerate(annotation["masks"]):
                
                # Extract points and label
                points = shape["points"]
                label = shape["label"]
                
                # Add category to dataset info
                dataset_info["categories"].add(label)
                
                # Convert polygon to mask
                mask = polygon_to_mask(points, h, w)
                
                # Visualize if requested
                if visualize:
                    visualize_mask(image, mask, f"Mask for {label}")
                
                # Save mask
                mask_filename = image_filename
                cv2.imwrite(os.path.join(output_dir, "masks", mask_filename), mask * 255)
                
                # Store mask info
                masks_info.append({
                    "mask_path": f"masks/{mask_filename}",
                    "label": label,
                    "points": points
                })
                
                dataset_info["num_masks"] += 1
            
            # Copy image to output directory
            shutil.copy(image_path, os.path.join(output_dir, "images", image_filename))
            
            # Save processed annotation
            processed_annotation = {
                "image_path": f"images/{image_filename}",
                "height": h,
                "width": w,
                "masks": masks_info
            }
            
            with open(os.path.join(output_dir, "annotations", annotation_file), 'w') as f:
                json.dump(processed_annotation, f, indent=2)
            
            # Add to dataset info
            dataset_info["images"].append(image_filename)
            
        except Exception as e:
            print(f"Error processing {annotation_file}: {e}")
    
    # Convert categories set to list and save dataset info
    dataset_info["categories"] = list(dataset_info["categories"])
    dataset_info["num_images"] = len(dataset_info["images"])
    
    with open(os.path.join(output_dir, "dataset_info.json"), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print("Data preparation completed successfully!")
    print(f"Total images: {dataset_info['num_images']}")
    print(f"Total masks: {dataset_info['num_masks']}")
    print(f"Categories: {dataset_info['categories']}")

def main():
    args = parse_args()
    process_annotations(args.data_dir, args.output_dir, args.visualize, args.num_samples)

if __name__ == "__main__":
    main()