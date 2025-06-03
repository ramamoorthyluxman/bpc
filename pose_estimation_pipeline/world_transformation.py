import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

def parse_rotation(s):
    """Parse space-separated string into 3x3 rotation matrix"""
    values = list(map(float, s.split()))
    if len(values) != 9:
        raise ValueError(f"Rotation matrix requires 9 values, got {len(values)}")
    return np.array(values).reshape(3, 3)

def parse_translation(s):
    """Parse space-separated string into 3x1 translation vector"""
    values = list(map(float, s.split()))
    if len(values) != 3:
        raise ValueError(f"Translation vector requires 3 values, got {len(values)}")
    return np.array(values).reshape(3, 1)

def load_camera_metadata(scene_dir):
    """Load camera metadata from scene_camera_photoneo.json"""
    json_path = os.path.join(scene_dir, 'scene_camera_photoneo.json')
    with open(json_path, 'r') as f:
        return json.load(f)

def convert_pose_to_world_frame(row, scene_dir):
    """Convert object pose from camera frame to world frame"""
    # Load camera metadata for this scene
    camera_metadata = load_camera_metadata(scene_dir)
    
    # Get camera pose for this image (im_id)
    im_id = str(row['im_id'])  # JSON keys are strings
    if im_id not in camera_metadata:
        raise ValueError(f"No camera metadata found for image ID {im_id} in scene {row['scene_id']}")
    
    cam_data = camera_metadata[im_id]
    
    # Extract camera rotation (R_w2c) and translation (t_w2c)
    R_w2c = np.array(cam_data['cam_R_w2c']).reshape(3, 3)
    t_w2c = np.array(cam_data['cam_t_w2c']).reshape(3, 1)
    
    # Parse object's rotation and translation from strings
    R_obj_c = parse_rotation(row['R'])
    t_obj_c = parse_translation(row['t'])
    
    # Compute camera-to-world transform
    R_c2w = R_w2c.T  # Inverse of rotation
    t_c2w = -R_c2w @ t_w2c
    
    # Transform object pose to world frame
    R_obj_w = R_c2w @ R_obj_c
    t_obj_w = R_c2w @ t_obj_c + t_c2w
    
    # Convert back to space-separated strings
    R_str = ' '.join(map(str, R_obj_w.flatten()))
    t_str = ' '.join(map(str, t_obj_w.flatten()))
    
    return R_str, t_str

def process_csv(input_csv, dataset_root, output_csv):
    """Process the CSV file to convert poses to world frame"""
    # Read input CSV
    df = pd.read_csv(input_csv)
    
    # Convert numeric columns to strings if needed
    df['scene_id'] = df['scene_id'].astype(str).str.zfill(6)
    df['im_id'] = df['im_id'].astype(str)
    
    # Convert each row
    new_rows = []
    for _, row in df.iterrows():
        scene_dir = os.path.join(dataset_root, row['scene_id'])
        
        try:
            R_w, t_w = convert_pose_to_world_frame(row, scene_dir)
            
            # Create new row with world frame pose
            new_row = row.to_dict()
            new_row['R'] = R_w
            new_row['t'] = t_w
            new_rows.append(new_row)
        except Exception as e:
            print(f"Error processing scene {row['scene_id']} image {row['im_id']}: {str(e)}")
            continue
    
    # Create new DataFrame and save
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(output_csv, index=False)
    print(f"Successfully saved converted poses to {output_csv}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert object poses from camera to world frame')
    parser.add_argument('input_csv', help='Path to input CSV file')
    parser.add_argument('dataset_root', help='Root directory of the dataset')
    parser.add_argument('output_csv', help='Path to save the output CSV file')
    
    args = parser.parse_args()
    
    process_csv(args.input_csv, args.dataset_root, args.output_csv)