import cv2
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import ast
import re
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'gms')))

from gms_transformation import gms_transformation


def extract_camera_type_from_path(image_path: str) -> str:
    """
    Extract camera type from image path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Camera type (e.g., 'cam1', 'photoneo', etc.)
    """
    # Convert to Path object for easier manipulation
    path_obj = Path(image_path)
    filename = path_obj.stem  # Get filename without extension
    
    # Common camera type patterns
    camera_patterns = [
        r'_cam(\d+)',           # _cam1, _cam2, etc.
        r'_photoneo',           # _photoneo
        r'_kinect',             # _kinect
        r'_realsense',          # _realsense
        r'_zed',                # _zed
        r'_([a-zA-Z]+\d*)',     # any pattern like _camera1, _sensor2, etc.
    ]
    
    for pattern in camera_patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            if 'cam' in pattern:
                return f"cam{match.group(1)}"  # Return cam1, cam2, etc.
            elif 'photoneo' in pattern:
                return "photoneo"
            elif 'kinect' in pattern:
                return "kinect"
            elif 'realsense' in pattern:
                return "realsense"
            elif 'zed' in pattern:
                return "zed"
            else:
                return match.group(1)  # Return the captured group
    
    # Fallback: try to extract from directory structure
    # Look for camera info in parent directories
    parts = path_obj.parts
    for part in reversed(parts):
        if 'cam' in part.lower():
            cam_match = re.search(r'cam(\d+)', part, re.IGNORECASE)
            if cam_match:
                return f"cam{cam_match.group(1)}"
        elif 'photoneo' in part.lower():
            return "photoneo"
        elif 'kinect' in part.lower():
            return "kinect"
        elif 'realsense' in part.lower():
            return "realsense"
        elif 'zed' in part.lower():
            return "zed"
    
    # Default fallback
    print(f"Warning: Could not extract camera type from path: {image_path}")
    return "unknown"


def object_masking_pipeline(json_file_path: str, csv_file_path: str, output_dir: str = "output") -> List[Dict]:
    """
    Extract and save masked test and reference images for each detected object.
    
    Args:
        json_file_path (str): Path to JSON file containing object detection results
        csv_file_path (str): Path to CSV file containing dataset with ground truth
        output_dir (str): Directory to save masked images (default: "output")
    
    Returns:
        List[Dict]: Results containing object info and file paths
        
    Generated Outputs:
        - Masked test images (from detection)
        - Masked reference images (from dataset)
        - Simple metadata about saved images
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created: {output_path.absolute()}")
    
    # Create subdirectories
    test_images_dir = output_path / "test_images"
    reference_images_dir = output_path / "reference_images"
    test_images_dir.mkdir(exist_ok=True)
    reference_images_dir.mkdir(exist_ok=True)
    
    # Load JSON data
    with open(json_file_path, 'r') as f:
        detection_data = json.load(f)
    
    # Load CSV data
    dataset_df = pd.read_csv(csv_file_path)
    
    # Extract image info from JSON
    image_path = detection_data['image_path']
    image_height = detection_data['height']
    image_width = detection_data['width']
    detected_objects = detection_data['masks']
    
    # Extract camera type from the main image path
    main_camera_type = extract_camera_type_from_path(image_path)
    print(f"Detected camera type: {main_camera_type}")
    
    # Load the main image
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    main_image = cv2.imread(image_path)
    
    results = []
    
    print(f"Processing {len(detected_objects)} detected objects...")
    
    for obj_idx, detected_obj in enumerate(detected_objects):
        print(f"Processing object {obj_idx + 1}/{len(detected_objects)}: {detected_obj['label']}")
        
        # Extract object ID from label (assuming format like "obj_000018")
        object_id = int(detected_obj['label'].split('_')[-1])
        
        # Create mask for detected object using POLYGON
        polygon_points = detected_obj['points']
        print(f"  Using polygon with {len(polygon_points)} points")
        
        detected_mask = create_mask_from_polygon(polygon_points, image_height, image_width)
        detected_masked_image = apply_mask_to_image(main_image, detected_mask)
        
        
        # Save test image (detected object)
        test_filename = f"test_{obj_idx + 1:02d}_{detected_obj['label']}_polygon.png"
        test_filepath = test_images_dir / test_filename
        
        cv2.imwrite(str(test_filepath), detected_masked_image)
        
        # Find matching objects in dataset - filter by camera type
        matching_entries = dataset_df[
            (dataset_df['object_id'] == object_id) & 
            (dataset_df['camera_type'] == main_camera_type)
        ]
        
        if len(matching_entries) == 0:
            print(f"  No matching entries found for object {object_id} with camera type '{main_camera_type}'")
            
            # Try without camera type filter as fallback
            print(f"  Trying without camera type filter...")
            matching_entries_fallback = dataset_df[dataset_df['object_id'] == object_id]
            
            if len(matching_entries_fallback) == 0:
                print(f"  No matching entries found for object {object_id} at all")
                # Still save the test image info
                result = {
                    'object_index': obj_idx,
                    'detected_object': detected_obj,
                    'detected_polygon': polygon_points,
                    'camera_type': main_camera_type,
                    'test_image_path': str(test_filepath),
                    'reference_image_paths': [],
                    'reference_count': 0
                }
                results.append(result)
                continue
            else:
                print(f"  Found {len(matching_entries_fallback)} entries without camera type filter")
                matching_entries = matching_entries_fallback
        
        print(f"  Found {len(matching_entries)} potential reference images")
        
        reference_image_paths = []
        
        # Process each matching dataset entry
        for ref_idx, (_, dataset_entry) in enumerate(matching_entries.iterrows()):
            try:
                print(f"    Processing reference {ref_idx + 1}/{len(matching_entries)} (Camera: {dataset_entry['camera_type']})")
                
                # Load dataset image
                dataset_image_path = dataset_entry['image_path']
                if not Path(dataset_image_path).exists():
                    print(f"    Dataset image not found: {dataset_image_path}")
                    continue
                
                dataset_image = cv2.imread(dataset_image_path)
                
                # Get dataset polygon
                dataset_polygon = get_dataset_polygon(dataset_entry)
                
                if not dataset_polygon:
                    print(f"    Could not get polygon for dataset entry")
                    continue
                
                # Create mask for dataset object using POLYGON
                dataset_mask = create_mask_from_polygon(dataset_polygon, 
                                                      int(dataset_entry['image_height']), 
                                                      int(dataset_entry['image_width']))
                
                dataset_masked_image = apply_mask_to_image(dataset_image, dataset_mask)
                
                # Save reference image
                ref_filename = f"ref_{obj_idx + 1:02d}_{detected_obj['label']}_{ref_idx + 1:02d}_{dataset_entry['camera_type']}_polygon.png"
                ref_filepath = reference_images_dir / ref_filename
                save_image(dataset_masked_image, ref_filepath)
                # cv2.imwrite(str(ref_filepath), dataset_masked_image)
                reference_image_paths.append(str(ref_filepath))

                print(f"    Saved reference image: {ref_filename}")
                
            except Exception as e:
                print(f"    Error processing dataset entry: {e}")
                continue
        
        

        for ref_img_path in reference_image_paths:
            img1 =  cv2.imread(str(test_filepath))
            img2 = cv2.imread(ref_img_path)
            matches_image, matches_gms, H, transformed_src2, inliers = gms_transformation(img1, img2) 
            if len(inliers) > 6:
                print("nb of matches: ", len(matches_gms))        

                # Visualize side by side: src1, src2, transformed src2
                h = max(img1.shape[0], img2.shape[0], transformed_src2.shape[0])
                w_total = img1.shape[1] + img2.shape[1] + transformed_src2.shape[1]
                comparison = np.zeros((h, w_total, 3), dtype=np.uint8)
                
                comparison[0:img1.shape[0], 0:img1.shape[1]] = img1
                comparison[0:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
                comparison[0:transformed_src2.shape[0], img1.shape[1]+img2.shape[1]:] = transformed_src2
                
                cv2.imwrite(os.path.join(output_path, "transformation_" + str(obj_idx) + "_" + str(ref_idx) + ".png"), comparison)
                cv2.imwrite(os.path.join(output_path, "matches_" + str(obj_idx) + "_" + str(ref_idx) + ".png"), matches_image)

                break                

                

def get_dataset_polygon(dataset_entry) -> List[List[int]]:
    """
    Get polygon for dataset entry from polygon_mask column.
    
    Returns polygon points as list of [x, y] coordinates
    """
    # Get polygon from polygon_mask column
    if 'polygon_mask' in dataset_entry:
        polygon_str = dataset_entry['polygon_mask']
        polygon = parse_polygon_string(polygon_str)
        if polygon:
            return polygon
    
    return None

def parse_polygon_string(polygon_str: str) -> List[List[int]]:
    """Parse polygon string from CSV file."""
    try:
        # Handle different formats
        polygon_str = polygon_str.strip()
        
        # Try direct evaluation first
        if polygon_str.startswith('[[') and polygon_str.endswith(']]'):
            return ast.literal_eval(polygon_str)
        
        # Handle space-separated format
        if '], [' in polygon_str:
            return ast.literal_eval(polygon_str)
        
        # Handle other formats
        # Remove extra spaces and fix format
        polygon_str = re.sub(r'\s+', ' ', polygon_str)
        polygon_str = polygon_str.replace(' ]', ']').replace('[ ', '[')
        
        return ast.literal_eval(polygon_str)
        
    except Exception as e:
        print(f"Error parsing polygon string: {polygon_str}, Error: {e}")
        return []

def create_mask_from_polygon(polygon_points: List[List[int]], height: int, width: int) -> np.ndarray:
    """
    Create a binary mask from polygon points.
    
    Args:
        polygon_points: List of [x, y] points defining the polygon
        height: Image height
        width: Image width
        
    Returns:
        Binary mask with 1s inside the polygon
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if not polygon_points:
        return mask
    
    # Convert polygon points to numpy array
    polygon_np = np.array(polygon_points, dtype=np.int32)
    
    # Fill the polygon region
    cv2.fillPoly(mask, [polygon_np], 255)
    
    return mask

def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to image, zeroing out pixels outside the mask."""
    masked_image = image.copy()
    mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
    masked_image = (masked_image * mask_3d).astype(np.uint8)
    return masked_image

def save_image(image: np.ndarray, filepath: Path):
    """Save image to file."""
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white', pad_inches=0)
    plt.close()

def save_summary_info(results: List[Dict], output_dir: Path):
    """Save summary information about all processed objects."""
    
    summary_filepath = output_dir / "processing_summary.txt"
    
    with open(summary_filepath, 'w') as f:
        f.write("OBJECT MASKING PIPELINE SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Total objects processed: {len(results)}\n")
        f.write(f"Processing method: Polygon masking with preprocessing\n")
        f.write(f"Preprocessing: CLAHE contrast enhancement + bilateral noise reduction\n")
        f.write("="*50 + "\n\n")
        
        total_test_images = len(results)
        total_reference_images = sum(result['reference_count'] for result in results)
        
        f.write(f"Images saved:\n")
        f.write(f"  Test images: {total_test_images}\n")
        f.write(f"  Reference images: {total_reference_images}\n")
        f.write(f"  Total images: {total_test_images + total_reference_images}\n\n")
        
        camera_types = set(result['camera_type'] for result in results)
        f.write(f"Camera types: {', '.join(camera_types)}\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-"*30 + "\n")
        
        for result in results:
            f.write(f"Object: {result['detected_object']['label']}\n")
            f.write(f"  Camera Type: {result['camera_type']}\n")
            f.write(f"  Polygon Points: {len(result['detected_polygon'])}\n")
            f.write(f"  BBox: {result['detected_bbox']}\n")
            f.write(f"  Test Image: {Path(result['test_image_path']).name}\n")
            f.write(f"  Reference Images: {result['reference_count']}\n")
            if result['reference_count'] > 0:
                for ref_path in result['reference_image_paths']:
                    f.write(f"    - {Path(ref_path).name}\n")
            f.write("-"*30 + "\n")
    
    print(f"Saved processing summary: {summary_filepath}")
    
    # Also save results as JSON for programmatic access
    json_summary_filepath = output_dir / "processing_results.json"
    
    # Convert results to JSON-serializable format
    json_results = []
    for result in results:
        json_result = {
            'object_index': result['object_index'],
            'object_label': result['detected_object']['label'],
            'camera_type': result['camera_type'],
            'polygon_points_count': len(result['detected_polygon']),
            'bbox': result['detected_bbox'],
            'test_image_path': result['test_image_path'],
            'reference_image_paths': result['reference_image_paths'],
            'reference_count': result['reference_count']
        }
        json_results.append(json_result)
    
    with open(json_summary_filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Saved JSON results: {json_summary_filepath}")


def register():
    
    results = object_masking_pipeline("/home/rama/bpc_ws/bpc/maskRCNN/results/000000_annotation.json", "/home/rama/bpc_ws/bpc/utilities/filtered_dataset.csv")
    
    

if __name__ == "__main__":
    register()