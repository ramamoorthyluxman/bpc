import cv2
import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

def create_pointcloud_with_colors(depth_path: str, 
                                 rgb_path: str,
                                 k_matrix, 
                                 depth_scale, 
                                 use_gpu: bool = True,
                                 ):
    """
    Create a colored point cloud from depth and RGB images.
    
    Args:
        depth_path: Path to depth image
        rgb_path: Path to RGB image
        k_matrix: Camera intrinsic matrix (3x3)
        depth_scale: Scale factor for depth values
        use_gpu: Whether to use GPU acceleration

    Returns:
        point_cloud: Nx3 array of [x, y, z] coordinates (valid points only)
        colors: Nx3 array of [r, g, b] values (0-255) (valid points only)
    """
    depth_scale = float(depth_scale)
    
    # Load images
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    if depth_image is None or rgb_image is None:
        raise ValueError("Could not load one or both images")
    
    height, width = depth_image.shape
    
    # Decide whether to use GPU or CPU
    use_gpu = use_gpu and CUPY_AVAILABLE
    xp = cp if use_gpu else np
    
    # Convert to appropriate array types
    if use_gpu:
        depth = cp.asarray(depth_image, dtype=cp.float32) * depth_scale
        rgb = cp.asarray(rgb_image, dtype=cp.uint8)
        k_matrix = cp.asarray(k_matrix, dtype=cp.float32)
    else:
        depth = depth_image.astype(np.float32) * depth_scale
        rgb = rgb_image.astype(np.uint8)
        k_matrix = np.asarray(k_matrix, dtype=np.float32)
    
    # Extract camera parameters
    fx = k_matrix[0, 0]
    fy = k_matrix[1, 1]
    cx = k_matrix[0, 2]
    cy = k_matrix[1, 2]
    
    # Create coordinate grids
    u_coords, v_coords = xp.meshgrid(xp.arange(width), xp.arange(height))
    
    # Vectorized 3D coordinate calculation
    z = depth
    x = (u_coords - cx) * z / fx
    y = (v_coords - cy) * z / fy
    
    # Create validity mask (filter out invalid depth values)
    valid_mask = z > 0  # Only points with valid depth
    
    # Flatten arrays and apply mask
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    valid_flat = valid_mask.flatten()
    
    # Apply validity mask
    x_valid = x_flat[valid_flat]
    y_valid = y_flat[valid_flat]
    z_valid = z_flat[valid_flat]
    
    # Stack coordinates for valid points only
    point_cloud = xp.stack([x_valid, y_valid, z_valid], axis=1)
    
    # Extract colors for valid points
    # Reshape RGB to (height*width, 3) then apply same mask
    rgb_flat = rgb.reshape(-1, 3)
    colors_valid = rgb_flat[valid_flat]
    
    # Convert back to CPU if using GPU
    if use_gpu:
        point_cloud = cp.asnumpy(point_cloud)
        colors_valid = cp.asnumpy(colors_valid)
    
    return point_cloud, colors_valid

def pixel_to_3d_point(u: int, v: int, 
                      depth_path: str, 
                      k_matrix, 
                      depth_scale):
    """Convert a 2D pixel coordinate to 3D point using depth information."""
    # Load depth image
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    # Get depth value at pixel
    depth = depth_image[v, u].astype(np.float32) * depth_scale
    
    # Extract camera parameters
    fx = float(k_matrix[0, 0])
    fy = float(k_matrix[1, 1])
    cx = float(k_matrix[0, 2])
    cy = float(k_matrix[1, 2])
    
    # Convert to 3D coordinates
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    return [x, y, z]


def pixel_to_point_normal(u: int, v: int, 
                         depth_path: str, 
                         k_matrix, 
                         depth_scale,
                         radius: int = 40):
    """Return normal vector using many neighboring 3D points in a circular area."""
    # Load depth image to check bounds
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    height, width = depth_image.shape
    
    points = []
    
    # Sample points in circular neighborhood
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # Check if point is within circle and image bounds
            if dx*dx + dy*dy <= radius*radius:
                pu, pv = u + dx, v + dy
                if 0 <= pu < width and 0 <= pv < height:
                    try:
                        point = pixel_to_3d_point(pu, pv, depth_path, k_matrix, depth_scale)
                        if point[2] > 0:  # Valid depth
                            points.append(point)
                    except:
                        continue
    
    if len(points) < 10:  # Need enough points
        return None
    
    # Fit plane to points using least squares
    points = np.array(points)
    centroid = np.mean(points, axis=0)
    
    # Center the points
    centered = points - centroid
    
    # SVD to find normal (smallest singular vector)
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]  # Last row = smallest singular vector
    
    # Ensure normal points toward camera
    if normal[2] > 0:
        normal = -normal
        
    return normal.tolist()