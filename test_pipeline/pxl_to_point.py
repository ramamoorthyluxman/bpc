import json
import numpy as np
import cv2
from typing import List, Tuple, Union, Optional

def load_camera_params(json_path: str) -> dict:
    """Load camera intrinsic parameters from JSON file."""
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params

def pixels_to_3d_points(pixel_indices: List[Tuple[int, int]], 
                       depth_path: str, 
                       k_matrix,
                       depth_scale,
                       return_colors: bool = False,
                       rgb_path: Optional[str] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Convert specific pixel indices to 3D points efficiently.
    
    Args:
        pixel_indices: List of (u, v) pixel coordinates where u=column, v=row
        depth_path: Path to depth image
        camera_params: Either path to camera JSON file or camera parameters dict
        return_colors: Whether to return RGB colors for the pixels
        rgb_path: Path to RGB image (required if return_colors=True)
    
    Returns:
        np.ndarray: Array of 3D points with shape (N, 3) where N is number of pixels
        If return_colors=True, returns tuple (points, colors) where colors has shape (N, 3)
        
    Note:
        - Invalid depth pixels (0, NaN, inf) will have NaN coordinates
        - Pixel coordinates are in OpenCV format: u=column (x), v=row (y)
        - 3D coordinates are in camera frame: X=right, Y=down, Z=forward
    """
    
    fx = float(k_matrix[0, 0])
    fy = float(k_matrix[1, 1])
    cx = float(k_matrix[0, 2])
    cy = float(k_matrix[1, 2])
    
    # Load depth image
    depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
    # print("Depth image type: ", depth_img.dtype)
    # print(depth_img.min(), depth_img.max())

    if depth_img is None:
        raise ValueError(f"Could not load depth image: {depth_path}")
    
    # Load RGB image if colors are requested
    rgb_img = None
    if return_colors:
        if rgb_path is None:
            raise ValueError("rgb_path is required when return_colors=True")
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            raise ValueError(f"Could not load RGB image: {rgb_path}")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    height, width = depth_img.shape
    
    # Convert pixel indices to numpy arrays for vectorized operations
    pixel_indices = np.array(pixel_indices)
    u_coords = pixel_indices[:, 0]  # column coordinates
    v_coords = pixel_indices[:, 1]  # row coordinates
    
    # Validate pixel coordinates
    valid_mask = (u_coords >= 0) & (u_coords < width) & (v_coords >= 0) & (v_coords < height)
    if not valid_mask.all():
        invalid_pixels = pixel_indices[~valid_mask]
        print(f"Warning: {len(invalid_pixels)} pixels are outside image bounds: {invalid_pixels.tolist()}")
    
    # Initialize output arrays
    num_pixels = len(pixel_indices)
    points_3d = np.full((num_pixels, 3), np.nan, dtype=np.float32)
    colors = None
    if return_colors:
        colors = np.full((num_pixels, 3), np.nan, dtype=np.float32)

    # Process only valid pixels
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) > 0:
        
        u_valid = u_coords[valid_indices]
        v_valid = v_coords[valid_indices]

        # Extract depth values for valid pixels
        depth_values = depth_img[v_valid, u_valid].astype(np.float32)

        # Convert depth to Z coordinates
        z_coords = depth_values * depth_scale

        # Compute X and Y coordinates using camera intrinsics
        x_coords = (u_valid - cx) * z_coords / fx
        y_coords = (v_valid - cy) * z_coords / fy
        
        # Handle invalid depth values (0, NaN, inf)
        depth_valid_mask = (depth_values > 0) & np.isfinite(depth_values)
        
        # Set coordinates to NaN for invalid depth
        x_coords = np.where(depth_valid_mask, x_coords, np.nan)
        y_coords = np.where(depth_valid_mask, y_coords, np.nan)
        z_coords = np.where(depth_valid_mask, z_coords, np.nan)
        
        # Store results for valid pixels
        points_3d[valid_indices] = np.column_stack([x_coords, y_coords, z_coords])
        
        # Extract colors if requested
        if return_colors and rgb_img is not None:
            rgb_values = rgb_img[v_valid, u_valid].astype(np.float32) / 255.0
            colors[valid_indices] = rgb_values

    
    
    # Print summary
    valid_depth_count = np.sum(np.isfinite(points_3d[:, 2]))
    # print(f"Processed {num_pixels} pixels:")
    # print(f"  - Valid coordinates: {len(valid_indices)}")
    # print(f"  - Valid depth: {valid_depth_count}")
    # print(f"  - Invalid/NaN: {num_pixels - valid_depth_count}")
    
    if return_colors:
        return points_3d, colors
    else:
        return points_3d

def pixel_to_3d_point(u: int, v: int, 
                     depth_path: str, 
                     k_matrix, 
                     depth_scale,
                     return_color: bool = False,
                     rgb_path: Optional[str] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Convert a single pixel to 3D point (convenience function).
    
    Args:
        u: Column coordinate (x)
        v: Row coordinate (y)
        depth_path: Path to depth image
        camera_params: Either path to camera JSON file or camera parameters dict
        return_color: Whether to return RGB color for the pixel
        rgb_path: Path to RGB image (required if return_color=True)
    
    Returns:
        np.ndarray: 3D point with shape (3,) [x, y, z]
        If return_color=True, returns tuple (point, color) where color has shape (3,)
    """
    
    if return_color:
        points, colors = pixels_to_3d_points([(u, v)], depth_path, k_matrix, depth_scale, 
                                           return_colors=True, rgb_path=rgb_path)
        return points[0], colors[0]
    else:
        points = pixels_to_3d_points([(u, v)], depth_path, k_matrix, depth_scale)
        return points[0]

# Example usage and test function
def test_pixel_conversion():
    """Test the pixel to 3D conversion with example data."""
    
    # Example camera parameters (your format)
    camera_params = {
        "cx": 1954.1872863769531,
        "cy": 1103.6978149414062,
        "depth_scale": 0.1,
        "fx": 3981.985991142684,
        "fy": 3981.985991142684,
        "height": 2160,
        "width": 3840
    }
    
    # Example pixel indices (u, v) where u=column, v=row
    pixel_indices = [
        (1920, 1080),  # Center pixel
        (100, 200),    # Top-left region
        (3000, 1500),  # Bottom-right region
        (1954, 1103)   # Near optical center
    ]
    
    print("Example usage:")
    print(f"Camera parameters: {camera_params}")
    print(f"Pixel indices to convert: {pixel_indices}")
    print()
    
    # This would be your actual function call:
    # points_3d = pixels_to_3d_points(pixel_indices, "depth.png", camera_params)
    # print(f"3D points: {points_3d}")
    
    # Single pixel conversion example:
    # point_3d = pixel_to_3d_point(1920, 1080, "depth.png", camera_params)
    # print(f"Single 3D point: {point_3d}")
    
    print("To use with your actual data:")
    print("points_3d = pixels_to_3d_points(pixel_indices, 'your_depth.png', 'your_camera.json')")
    print("or")
    print("point_3d = pixel_to_3d_point(u, v, 'your_depth.png', 'your_camera.json')")

if __name__ == "__main__":
    test_pixel_conversion()