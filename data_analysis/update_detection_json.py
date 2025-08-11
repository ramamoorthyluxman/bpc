import json
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple

from pxl_to_point import pixel_to_3d_point


def find_valid_neighbor_pixel(center_u: int, center_v: int, 
                            depth_path: str, 
                            k_matrix,
                            depth_scale,
                            search_radius: int = 5) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
    """Find a valid neighboring pixel if the center pixel has invalid depth."""
    
    # Try center pixel first
    point_3d = pixel_to_3d_point(center_u, center_v, depth_path, k_matrix, depth_scale)
    
    if not np.any(np.isnan(point_3d)):
        return point_3d, (center_u, center_v)
    
    # Load depth image to check bounds
    depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    if depth_img is None:
        return None, None
    
    height, width = depth_img.shape
    
    # Search in expanding circles around the center
    for radius in range(1, search_radius + 1):
        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                if abs(du) != radius and abs(dv) != radius:
                    continue
                
                u = center_u + du
                v = center_v + dv
                
                if u < 0 or u >= width or v < 0 or v >= height:
                    continue
                
                point_3d = pixel_to_3d_point(u, v, depth_path, k_matrix, depth_scale)
                if not np.any(np.isnan(point_3d)):
                    return point_3d, (u, v)
    
    return None, None

def transform_3d_point(point, csv_row):
    """
    Transform a 3D point from camera coordinates to world coordinates.
    
    Args:
        point: List or array of 3D coordinates [x, y, z] in camera space
        csv_row: Pandas Series or dict containing transformation data with keys:
                cam_R_w2c_0 to cam_R_w2c_8 (rotation matrix elements)
                cam_t_w2c_0 to cam_t_w2c_2 (translation vector elements)
    
    Returns:
        numpy.ndarray: Transformed 3D point [x', y', z'] in world space
    """
    # Convert point to numpy array
    point = np.array(point, dtype=float)
    
    # Extract rotation matrix elements and reshape to 3x3
    rotation_elements = [
        csv_row['r_w2c_11'], csv_row['r_w2c_12'], csv_row['r_w2c_13'],
        csv_row['r_w2c_21'], csv_row['r_w2c_22'], csv_row['r_w2c_23'],
        csv_row['r_w2c_31'], csv_row['r_w2c_32'], csv_row['r_w2c_33']
    ]
    R_w2c = np.array(rotation_elements, dtype=float).reshape(3, 3)
    
    # Extract translation vector
    t_w2c = np.array([
        csv_row['t_w2c_x'],
        csv_row['t_w2c_y'],
        csv_row['t_w2c_z']
    ], dtype=float)
    
    # To transform from camera to world, we need to invert the w2c transformation
    # For rotation matrices, the inverse is the transpose
    R_c2w = R_w2c.T
    
    # Apply inverse transformation: world_point = R_c2w @ (camera_point - t_w2c)
    # This is equivalent to: world_point = R_w2c.T @ (camera_point - t_w2c)
    transformed_point = R_c2w @ (point - t_w2c)
    
    return transformed_point



def transform_pose_to_world(rotation_matrix, position, csv_row):
    """
    Transform a 6D pose from camera coordinates to world coordinates.
    
    Args:
        position: List or array of 3D position [x, y, z] in camera space
        rotation_matrix: 3x3 rotation matrix in camera space
        csv_row: Pandas Series or dict containing transformation data with keys:
                cam_R_w2c_0 to cam_R_w2c_8 (rotation matrix elements)
                cam_t_w2c_0 to cam_t_w2c_2 (translation vector elements)
    
    Returns:
        tuple: (transformed_position, transformed_rotation_matrix)
    """
    # Convert inputs to numpy arrays
    position = np.array(position, dtype=float)
    rotation_matrix = np.array(rotation_matrix, dtype=float)
    
    # Extract rotation matrix elements and reshape to 3x3
    rotation_elements = [
        csv_row['r_w2c_11'], csv_row['r_w2c_12'], csv_row['r_w2c_13'],
        csv_row['r_w2c_21'], csv_row['r_w2c_22'], csv_row['r_w2c_23'],
        csv_row['r_w2c_31'], csv_row['r_w2c_32'], csv_row['r_w2c_33']
    ]
    R_w2c = np.array(rotation_elements, dtype=float).reshape(3, 3)
    
    # Extract translation vector
    t_w2c = np.array([
        csv_row['t_w2c_x'],
        csv_row['t_w2c_y'],
        csv_row['t_w2c_z']
    ], dtype=float)
    
    # Camera-to-world transformation
    R_c2w = R_w2c.T
    
    # Transform position
    transformed_position = R_c2w @ (position - t_w2c)
    
    # Transform rotation
    transformed_rotation = R_c2w @ rotation_matrix
    
    return transformed_rotation, transformed_position


def transform_pose_to_world_backup(R_camera, T_camera, csv_row):
    """
    Transform a pose (rotation matrix + translation) from camera coordinates to world coordinates.
    
    Args:
        R_camera: 3x3 numpy array representing rotation matrix in camera space
        T_camera: 3D numpy array representing translation vector in camera space
        csv_row: Pandas Series or dict containing transformation data with keys:
                cam_R_w2c_0 to cam_R_w2c_8 (rotation matrix elements)
                cam_t_w2c_0 to cam_t_w2c_2 (translation vector elements)
    
    Returns:
        tuple: (R_world, T_world)
               - R_world: 3x3 numpy array representing rotation matrix in world space
               - T_world: 3D numpy array representing translation vector in world space
    """
    # Convert inputs to numpy arrays
    R_camera = np.array(R_camera, dtype=float)
    T_camera = np.array(T_camera, dtype=float)
    
    # Extract rotation matrix elements and reshape to 3x3
    rotation_elements = [
        csv_row['cam_R_w2c_0'], csv_row['cam_R_w2c_1'], csv_row['cam_R_w2c_2'],
        csv_row['cam_R_w2c_3'], csv_row['cam_R_w2c_4'], csv_row['cam_R_w2c_5'],
        csv_row['cam_R_w2c_6'], csv_row['cam_R_w2c_7'], csv_row['cam_R_w2c_8']
    ]
    R_w2c = np.array(rotation_elements, dtype=float).reshape(3, 3)
    
    # Extract translation vector
    t_w2c = np.array([
        csv_row['cam_t_w2c_0'],
        csv_row['cam_t_w2c_1'],
        csv_row['cam_t_w2c_2']
    ], dtype=float)
    
    # Camera-to-world transformation
    R_c2w = R_w2c.T  # Inverse of rotation matrix is its transpose
    # R_c2w = R_w2c

    # Transform translation: world_translation = R_c2w @ (camera_translation - t_w2c)
    # Following the same pattern as the original 3D point transformation
    # T_world = (R_c2w @ T_camera) - t_w2c
    T_world = (R_c2w @ T_camera) + t_w2c
    
    # Transform rotation: R_world = R_c2w @ R_camera
    R_world = R_c2w @ R_camera
    
    return R_world, T_world


def add_3d_centers_to_json(cam_row: None,
                          json_data: Dict[str, Any], 
                          depth_path: str, 
                          search_radius: int = 5) -> Dict[str, Any]:
    """
    Add 3D center locations to objects in JSON data.
    
    Args:
        json_data: Dictionary containing the JSON data with object detections
        depth_path: Path to corresponding depth image
        camera_params_path: Path to camera parameters JSON file
        search_radius: Search radius for finding valid neighboring pixels if center is invalid
        
    Returns:
        Updated JSON data with 'object_center_3d_loc' field added to each object
    """
    
    
    
    # Handle nested JSON structure
    if 'results' in json_data:
        masks_data = json_data['results'][0]['masks']
    elif 'masks' in json_data:
        masks_data = json_data['masks']
    else:
        raise ValueError("Could not find masks data in JSON. Expected 'results' or 'masks' key.")
    
    
    # Process each object
    for i, obj in enumerate(masks_data):
        label = obj['label']
        geometric_center = obj['geometric_center']
        u, v = geometric_center[0], geometric_center[1]  # u=column, v=row
        

        # Extract K matrix values and reshape to 3x3
        k_matrix = np.array([float(cam_row[f'k{i//3+1}{i%3+1}']) for i in range(9)]).reshape(3, 3)

        # Extract depth scale
        depth_scale = float(cam_row['depth_scale'])

        # Try to find valid 3D point
        point_3d, used_pixel = find_valid_neighbor_pixel(
            int(u), int(v), depth_path, k_matrix, depth_scale, search_radius
        )
        
        
        if point_3d is not None:
            # Convert to regular Python list for JSON serialization
            center_3d = [float(point_3d[0]), float(point_3d[1]), float(point_3d[2])]
            obj['object_center_3d_local'] = center_3d
            obj['object_center_3d_world'] = transform_3d_point(center_3d, cam_row)
            
            
        else:
            obj['object_center_3d_local'] = None
            obj['object_center_3d_world'] = None
            print(f"  ❌ No valid depth found")
    
    # Print summary
    valid_count = sum(1 for obj in masks_data if obj.get('object_center_3d_loc') is not None)
    
    return json_data

# Example usage:
# updated_json = add_3d_centers_to_json(your_json_data, "depth.png", "camera.json")