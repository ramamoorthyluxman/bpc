import json
import numpy as np
import cv2
import open3d as o3d
import argparse
from pathlib import Path

def load_camera_params(json_path):
    """Load camera intrinsic parameters from JSON file."""
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params

def create_point_cloud(rgb_path, depth_path, camera_params):
    """Create point cloud from RGB and depth images using camera parameters."""
    # Load images
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)  # Load as-is for 16-bit depth maps
    
    # Make sure images are loaded properly
    if rgb is None:
        raise ValueError(f"Could not read RGB image from {rgb_path}")
    if depth is None:
        raise ValueError(f"Could not read depth image from {depth_path}")
    
    # Extract camera parameters
    fx = camera_params["fx"]
    fy = camera_params["fy"]
    cx = camera_params["cx"]
    cy = camera_params["cy"]
    depth_scale = camera_params["depth_scale"]
    
    # Create point cloud
    height, width = depth.shape
    points = []
    colors = []
    
    for v in range(height):
        for u in range(width):
            # Get depth value (and apply scale)
            z = depth[v, u] * depth_scale
            
            # Skip invalid depth values
            if z <= 0:
                continue
            
            # Calculate 3D coordinates
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            
            # Add point and its color
            points.append([x, y, z])
            colors.append(rgb[v, u] / 255.0)  # Normalize color values to [0, 1]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    return pcd

def main():
    parser = argparse.ArgumentParser(description="Generate point cloud from RGB and depth images")
    parser.add_argument("--rgb", required=True, help="Path to RGB image")
    parser.add_argument("--depth", required=True, help="Path to depth image")
    parser.add_argument("--camera", required=True, help="Path to camera parameters JSON file")
    parser.add_argument("--output", required=True, help="Output path for the point cloud file")
    args = parser.parse_args()
    
    # Load camera parameters
    camera_params = load_camera_params(args.camera)
    
    # Create point cloud
    print("Creating point cloud...")
    pcd = create_point_cloud(args.rgb, args.depth, camera_params)
    
    # Save point cloud
    output_path = args.output
    if not output_path.endswith(('.ply', '.pcd')):
        output_path += '.ply'
    
    print(f"Saving point cloud to {output_path}...")
    o3d.io.write_point_cloud(output_path, pcd)
    print("Done!")

if __name__ == "__main__":
    main()