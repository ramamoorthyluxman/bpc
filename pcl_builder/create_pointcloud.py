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

def create_organized_point_cloud(rgb_path, depth_path, camera_params):
    """Create organized point cloud from RGB and depth images using camera parameters."""
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
    points = np.zeros((height, width, 3), dtype=np.float32)  # Organized structure
    colors = np.zeros((height, width, 3), dtype=np.float32)  # Organized colors
    
    # Vectorized implementation for speed
    v, u = np.indices((height, width))
    z = depth * depth_scale
    
    # Calculate coordinates (vectorized)
    valid_mask = z > 0
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    
    # Only compute valid points (to avoid division by zero)
    x[valid_mask] = (u[valid_mask] - cx) * z[valid_mask] / fx
    y[valid_mask] = (v[valid_mask] - cy) * z[valid_mask] / fy
    
    # Set all points
    points[:, :, 0] = x
    points[:, :, 1] = y
    points[:, :, 2] = z
    colors[:, :] = rgb / 255.0  # Normalize color values to [0, 1]
    
    # Create Open3D point cloud with organized data
    pcd = o3d.geometry.PointCloud()
    
    # Reshape to a flat list while preserving order
    pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
    
    # Store width and height in metadata
    metadata = {
        "width": width,
        "height": height,
        "is_organized": True
    }
    
    return pcd, metadata, width, height

def build_pcl(rgb, depth, camera, output=None):
    """
    Build point cloud and return it directly (no file I/O)
    
    Args:
        rgb: Path to RGB image
        depth: Path to depth image  
        camera: Path to camera parameters JSON
        output: Ignored (kept for compatibility)
        
    Returns:
        pcd: Open3D point cloud object
        metadata: Dictionary with width, height, is_organized
        width: Image width
        height: Image height
    """
    # Load camera parameters
    camera_params = load_camera_params(camera)
    
    # Create point cloud (no file operations)
    pcd, metadata, width, height = create_organized_point_cloud(rgb, depth, camera_params)
    
    # Return the point cloud directly - no saving!
    return pcd, metadata, width, height

def build_pcl_with_save(rgb, depth, camera, output):
    """
    Build point cloud and save to file (for backward compatibility)
    """
    # Build the point cloud
    pcd, metadata, width, height = build_pcl(rgb, depth, camera)
    
    # Save point cloud as PCD if output path is provided
    output_path = output
    if not output_path.endswith('.pcd'):
        output_path = str(Path(output_path).with_suffix('.pcd'))
    
    print(f"Saving organized point cloud to {output_path}...")
    
    # Write the point cloud
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False, 
                             compressed=True, print_progress=True)
    
    # Manually modify PCD file to include width and height
    with open(output_path, 'rb') as f:
        pcd_content = f.read()
    
    if b'WIDTH' in pcd_content:
        modified_content = pcd_content.replace(
            b'WIDTH', f'WIDTH {width}\nHEIGHT {height}\n# '.encode())
    else:
        header_end = pcd_content.find(b'DATA binary_compressed')
        if header_end != -1:
            header_end = pcd_content.find(b'\n', header_end) + 1
            header = pcd_content[:header_end]
            binary_data = pcd_content[header_end:]
            
            width_height = f'WIDTH {width}\nHEIGHT {height}\n'.encode()
            modified_content = header.replace(b'DATA binary_compressed', 
                                            width_height + b'DATA binary_compressed')
            modified_content += binary_data
        else:
            print("Warning: Could not modify PCD file to include organized data structure.")
            modified_content = pcd_content
    
    with open(output_path, 'wb') as f:
        f.write(modified_content)
    
    print(f"Done! Created organized point cloud with {width}x{height} = {width*height} points")
    
    return pcd, metadata, width, height

def main():
    """Command line interface - still saves to file for CLI usage"""
    parser = argparse.ArgumentParser(description="Generate organized point cloud from RGB and depth images")
    parser.add_argument("--rgb", required=True, help="Path to RGB image")
    parser.add_argument("--depth", required=True, help="Path to depth image")
    parser.add_argument("--camera", required=True, help="Path to camera parameters JSON file")
    parser.add_argument("--output", required=True, help="Output path for the point cloud file")
    parser.add_argument("--no-save", action="store_true", help="Don't save to file, just process")
    args = parser.parse_args()
    
    if args.no_save:
        # Just build the point cloud without saving
        print("Creating organized point cloud (no file output)...")
        pcd, metadata, width, height = build_pcl(args.rgb, args.depth, args.camera)
        print(f"Done! Created organized point cloud with {width}x{height} = {width*height} points")
        print("Point cloud built in memory only (not saved)")
    else:
        # Build and save point cloud
        print("Creating organized point cloud...")
        pcd, metadata, width, height = build_pcl_with_save(args.rgb, args.depth, args.camera, args.output)

if __name__ == "__main__":
    main()