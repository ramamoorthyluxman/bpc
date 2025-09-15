import json
import numpy as np
import cv2
import argparse
from pathlib import Path
import time

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print(f"GPU acceleration available: {cp.cuda.get_device_id()}")
except ImportError:
    print("CuPy not available. Install with: pip install cupy-cuda11x (or cupy-cuda12x)")
    GPU_AVAILABLE = False
    cp = np  # Fallback to numpy

def load_camera_params(json_path):
    """Load camera intrinsic parameters from JSON file."""
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params

def create_organized_point_cloud_gpu(rgb_path, depth_path, camera_params):
    """Create ORGANIZED point cloud using GPU acceleration - maintains pixel-to-point mapping."""
    
    # Load images
    import time
    load_start = time.time()
    
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
    if rgb is None or depth is None:
        raise ValueError("Could not load images")
    
    load_time = time.time() - load_start
    print(f"Image loading time: {load_time:.3f} seconds")
    
    # Extract camera parameters
    fx = camera_params["fx"]
    fy = camera_params["fy"]
    cx = camera_params["cx"]
    cy = camera_params["cy"]
    depth_scale = camera_params["depth_scale"]
    
    height, width = depth.shape
    total_pixels = height * width
    print(f"Processing {height}x{width} image on GPU (ORGANIZED mode)...")
    print(f"Output will have exactly {total_pixels:,} points (1:1 pixel mapping)")
    
    # Process on GPU or CPU
    gpu_start = time.time()
    
    if GPU_AVAILABLE:
        # Move data to GPU
        rgb_gpu = cp.asarray(rgb, dtype=cp.float32)
        depth_gpu = cp.asarray(depth, dtype=cp.float32)
        
        # Create coordinate grids on GPU - FULL SIZE (no filtering)
        v_gpu, u_gpu = cp.mgrid[0:height, 0:width]
        
        # Flatten to get pixel ordering (row-wise)
        u_flat = u_gpu.ravel()  # x coordinates
        v_flat = v_gpu.ravel()  # y coordinates
        depth_flat = depth_gpu.ravel()
        rgb_flat = rgb_gpu.reshape(-1, 3)  # Flatten RGB
        
        # Compute Z values (including invalid ones)
        z_flat = depth_flat * depth_scale
        
        # Compute 3D coordinates for ALL pixels
        x_flat = (u_flat - cx) * z_flat / fx
        y_flat = (v_flat - cy) * z_flat / fy
        
        # Handle invalid depth: set coordinates to NaN where depth is 0 or invalid
        invalid_mask = (depth_flat <= 0) | cp.isnan(depth_flat) | cp.isinf(depth_flat)
        x_flat = cp.where(invalid_mask, cp.nan, x_flat)
        y_flat = cp.where(invalid_mask, cp.nan, y_flat)
        z_flat = cp.where(invalid_mask, cp.nan, z_flat)
        
        # Stack coordinates - maintains exact pixel ordering
        points_gpu = cp.stack([x_flat, y_flat, z_flat], axis=1)
        colors_gpu = rgb_flat / 255.0
        
        # Transfer back to CPU
        points = cp.asnumpy(points_gpu)
        colors = cp.asnumpy(colors_gpu)
        
        # Count valid points for statistics
        valid_count = cp.sum(~invalid_mask).item()
        
        # Clear GPU memory
        del rgb_gpu, depth_gpu, v_gpu, u_gpu, u_flat, v_flat, depth_flat, rgb_flat
        del z_flat, x_flat, y_flat, invalid_mask, points_gpu, colors_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
    else:
        # Fallback to numpy with same logic
        print("GPU not available, using numpy fallback")
        
        # Create coordinate grids - FULL SIZE
        v, u = np.mgrid[0:height, 0:width]
        
        # Flatten to get pixel ordering
        u_flat = u.ravel()
        v_flat = v.ravel()
        depth_flat = depth.ravel().astype(np.float32)
        rgb_flat = rgb.reshape(-1, 3).astype(np.float32)
        
        # Compute Z values
        z_flat = depth_flat * depth_scale
        
        # Compute 3D coordinates for ALL pixels
        x_flat = (u_flat - cx) * z_flat / fx
        y_flat = (v_flat - cy) * z_flat / fy
        
        # Handle invalid depth: set coordinates to NaN where depth is 0 or invalid
        invalid_mask = (depth_flat <= 0) | np.isnan(depth_flat) | np.isinf(depth_flat)
        x_flat = np.where(invalid_mask, np.nan, x_flat)
        y_flat = np.where(invalid_mask, np.nan, y_flat)
        z_flat = np.where(invalid_mask, np.nan, z_flat)
        
        # Stack coordinates
        points = np.stack([x_flat, y_flat, z_flat], axis=1)
        colors = rgb_flat / 255.0
        
        # Count valid points
        valid_count = np.sum(~invalid_mask)
    
    gpu_time = time.time() - gpu_start
    print(f"GPU computation time: {gpu_time:.3f} seconds")
    
    # Verify organization
    assert len(points) == total_pixels, f"Point cloud size mismatch! Expected {total_pixels}, got {len(points)}"
    print(f"✅ ORGANIZED point cloud created: {len(points):,} points")
    print(f"   Valid points: {valid_count:,} ({100*valid_count/total_pixels:.1f}%)")
    print(f"   Invalid points: {total_pixels-valid_count:,} (kept as NaN)")
    print(f"   Pixel-to-point mapping: PERFECT 1:1")
    
    metadata = {
        "width": width,
        "height": height,
        "is_organized": True,  # NOW ORGANIZED!
        "total_points": len(points),
        "valid_points": valid_count,
        "invalid_points": total_pixels - valid_count,
        "total_pixels": total_pixels
    }
    
    return points, colors, metadata, width, height

def create_organized_point_cloud_batched(rgb_path, depth_path, camera_params, batch_size=512):
    """Process image in batches while maintaining organization."""
    
    # Load images
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
    if rgb is None or depth is None:
        raise ValueError("Could not load images")
    
    # Extract camera parameters
    fx = camera_params["fx"]
    fy = camera_params["fy"]
    cx = camera_params["cx"]
    cy = camera_params["cy"]
    depth_scale = camera_params["depth_scale"]
    
    height, width = depth.shape
    total_pixels = height * width
    print(f"Processing {height}x{width} image in batches of {batch_size} rows (ORGANIZED mode)...")
    
    if not GPU_AVAILABLE:
        # Fallback to regular processing
        return create_organized_point_cloud_gpu(rgb_path, depth_path, camera_params)
    
    # Pre-allocate output arrays - FULL SIZE
    all_points = np.empty((total_pixels, 3), dtype=np.float32)
    all_colors = np.empty((total_pixels, 3), dtype=np.float32)
    
    point_index = 0
    total_valid = 0
    
    # Process in batches of rows
    for y_start in range(0, height, batch_size):
        y_end = min(y_start + batch_size, height)
        batch_height = y_end - y_start
        batch_pixels = batch_height * width
        
        # Extract batch
        depth_batch = depth[y_start:y_end, :]
        rgb_batch = rgb[y_start:y_end, :]
        
        # Move batch to GPU
        depth_gpu = cp.asarray(depth_batch, dtype=cp.float32)
        rgb_gpu = cp.asarray(rgb_batch, dtype=cp.float32)
        
        # Create coordinate grids for this batch
        v_gpu, u_gpu = cp.mgrid[y_start:y_end, 0:width]
        
        # Flatten batch
        u_flat = u_gpu.ravel()
        v_flat = v_gpu.ravel()
        depth_flat = depth_gpu.ravel()
        rgb_flat = rgb_gpu.reshape(-1, 3)
        
        # Compute coordinates for this batch
        z_flat = depth_flat * depth_scale
        x_flat = (u_flat - cx) * z_flat / fx
        y_flat = (v_flat - cy) * z_flat / fy
        
        # Handle invalid depth
        invalid_mask = (depth_flat <= 0) | cp.isnan(depth_flat) | cp.isinf(depth_flat)
        x_flat = cp.where(invalid_mask, cp.nan, x_flat)
        y_flat = cp.where(invalid_mask, cp.nan, y_flat)
        z_flat = cp.where(invalid_mask, cp.nan, z_flat)
        
        # Stack batch results
        batch_points = cp.stack([x_flat, y_flat, z_flat], axis=1)
        batch_colors = rgb_flat / 255.0
        
        # Copy to pre-allocated arrays (maintains order)
        all_points[point_index:point_index+batch_pixels] = cp.asnumpy(batch_points)
        all_colors[point_index:point_index+batch_pixels] = cp.asnumpy(batch_colors)
        
        # Count valid points in this batch
        batch_valid = cp.sum(~invalid_mask).item()
        total_valid += batch_valid
        point_index += batch_pixels
        
        # Clear GPU memory for this batch
        del depth_gpu, rgb_gpu, v_gpu, u_gpu, u_flat, v_flat, depth_flat, rgb_flat
        del z_flat, x_flat, y_flat, invalid_mask, batch_points, batch_colors
        
        print(f"Processed batch {y_start//batch_size + 1}/{(height + batch_size - 1)//batch_size}, "
              f"valid points: {batch_valid:,}/{batch_pixels:,}")
    
    # Verify we processed everything
    assert point_index == total_pixels, f"Batch processing error: {point_index} != {total_pixels}"
    
    print(f"✅ ORGANIZED point cloud created: {total_pixels:,} points")
    print(f"   Valid points: {total_valid:,} ({100*total_valid/total_pixels:.1f}%)")
    print(f"   Invalid points: {total_pixels-total_valid:,} (kept as NaN)")
    
    # Clear all GPU memory
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
    
    metadata = {
        "width": width,
        "height": height,
        "is_organized": True,
        "total_points": total_pixels,
        "valid_points": total_valid,
        "invalid_points": total_pixels - total_valid,
        "total_pixels": total_pixels
    }
    
    return all_points, all_colors, metadata, width, height

def build_pcl(rgb, depth, camera, output=None, save_pcl=True, use_gpu=True, batch_size=None):
    """
    Build ORGANIZED point cloud with perfect pixel-to-point mapping.
    
    Args:
        rgb: Path to RGB image
        depth: Path to depth image  
        camera: Path to camera parameters JSON
        output: Output file path (required if save_pcl=True)
        save_pcl: Whether to save the point cloud to file (default: True)
        use_gpu: Whether to use GPU acceleration (default: True)
        batch_size: Process in batches for very large images (None = process all at once)
        
    Returns:
        tuple: (point_cloud, width, height) where point_cloud is an Open3D PointCloud object
               Point cloud will have exactly width×height points with perfect pixel mapping
    """
    # Load camera parameters
    camera_params = load_camera_params(camera)
    
    print(f"🔄 ORGANIZED POINT CLOUD MODE - Perfect pixel-to-point mapping")
    
    if use_gpu and GPU_AVAILABLE:
        if batch_size is not None:
            points, colors, metadata, width, height = create_organized_point_cloud_batched(rgb, depth, camera_params, batch_size)
        else:
            points, colors, metadata, width, height = create_organized_point_cloud_gpu(rgb, depth, camera_params)
    else:
        # CPU fallback - same organized logic
        print("Using CPU fallback (ORGANIZED)")
        rgb_img = cv2.imread(rgb)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        depth_img = cv2.imread(depth, cv2.IMREAD_ANYDEPTH)
        
        height, width = depth_img.shape
        total_pixels = height * width
        
        # Create coordinate grids - FULL SIZE
        v, u = np.mgrid[0:height, 0:width]
        u_flat = u.ravel()
        v_flat = v.ravel()
        depth_flat = depth_img.ravel().astype(np.float32)
        rgb_flat = rgb_img.reshape(-1, 3).astype(np.float32)
        
        # Compute coordinates for ALL pixels
        z_flat = depth_flat * camera_params["depth_scale"]
        x_flat = (u_flat - camera_params["cx"]) * z_flat / camera_params["fx"]
        y_flat = (v_flat - camera_params["cy"]) * z_flat / camera_params["fy"]
        
        # Handle invalid depth
        invalid_mask = (depth_flat <= 0) | np.isnan(depth_flat) | np.isinf(depth_flat)
        x_flat = np.where(invalid_mask, np.nan, x_flat)
        y_flat = np.where(invalid_mask, np.nan, y_flat)
        z_flat = np.where(invalid_mask, np.nan, z_flat)
        
        # Stack coordinates
        points = np.stack([x_flat, y_flat, z_flat], axis=1)
        colors = rgb_flat / 255.0
        
        valid_count = np.sum(~invalid_mask)
        print(f"✅ ORGANIZED point cloud created: {total_pixels:,} points")
        print(f"   Valid points: {valid_count:,} ({100*valid_count/total_pixels:.1f}%)")
        
        metadata = {
            "width": width,
            "height": height,
            "is_organized": True,
            "total_points": total_pixels,
            "valid_points": valid_count,
            "invalid_points": total_pixels - valid_count,
            "total_pixels": total_pixels
        }
    
    # Save to file if requested
    if save_pcl:
        if output is None:
            raise ValueError("Output path is required when save_pcl=True")
        save_organized_point_cloud(points, colors, width, height, metadata, output)
    
    try:
        import open3d as o3d
        
        print(f"Creating Open3D PointCloud with {len(points):,} points...")
        conversion_start = time.time()
        
        # Create Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        
        # Direct assignment - points array is already organized
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        conversion_time = time.time() - conversion_start
        print(f"Open3D conversion time: {conversion_time:.3f} seconds")
        
        # Final verification
        expected_size = width * height
        actual_size = len(pcd.points)
        print(f"🔍 VERIFICATION:")
        print(f"   Expected size: {expected_size:,} points")
        print(f"   Actual size: {actual_size:,} points")
        print(f"   Perfect mapping: {'✅ YES' if actual_size == expected_size else '❌ NO'}")
        
        return pcd, width, height
        
    except ImportError:
        print("Error: open3d required for PCD format. Install with: pip install open3d")
        print("Returning raw numpy arrays instead.")
        return points, colors, width, height

def save_organized_point_cloud(points, colors, width, height, metadata, output_path):
    """Save organized point cloud to PCD file."""
    try:
        import open3d as o3d
    except ImportError:
        print("Error: open3d required for saving PCD files. Install with: pip install open3d")
        return
    
    # Ensure PCD extension
    if not output_path.endswith('.pcd'):
        output_path = str(Path(output_path).with_suffix('.pcd'))
    
    total_points = len(points)
    valid_points = metadata["valid_points"]
    invalid_points = metadata["invalid_points"]
    
    print(f"Saving ORGANIZED point cloud to {output_path}...")
    print(f"   Total points: {total_points:,}")
    print(f"   Valid points: {valid_points:,}")
    print(f"   Invalid points: {invalid_points:,} (NaN)")
    
    # Convert to Open3D format for saving
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Write the point cloud
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False, 
                             compressed=True, print_progress=True)
    
    print(f"✅ Done! Saved organized point cloud")
    print(f"   Image size: {width}x{height} = {width*height:,} pixels")
    print(f"   Point cloud size: {total_points:,} points")
    print(f"   Perfect 1:1 mapping: {'✅ YES' if total_points == width*height else '❌ NO'}")

def main():
    """Command line interface with GPU acceleration - ORGANIZED mode."""
    parser = argparse.ArgumentParser(description="Generate ORGANIZED point cloud with perfect pixel mapping")
    parser.add_argument("--rgb", required=True, help="Path to RGB image")
    parser.add_argument("--depth", required=True, help="Path to depth image")
    parser.add_argument("--camera", required=True, help="Path to camera parameters JSON file")
    parser.add_argument("--output", help="Output path for the point cloud file (required if saving)")
    parser.add_argument("--no-save", action="store_true", help="Don't save point cloud, just return data")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--batch-size", type=int, help="Process in batches for very large images")
    args = parser.parse_args()
    
    import time
    start_time = time.time()
    
    use_gpu = not args.no_gpu and GPU_AVAILABLE
    save_pcl = not args.no_save
    
    # Validate arguments
    if save_pcl and not args.output:
        parser.error("--output is required when saving point cloud (default behavior). Use --no-save to skip saving.")
    
    print(f"🔄 ORGANIZED POINT CLOUD MODE")
    print(f"GPU: {'ON' if use_gpu else 'OFF'}, Save: {'ON' if save_pcl else 'OFF'}")
    if args.batch_size:
        print(f"Batch size: {args.batch_size} rows")
    
    # Use the unified function
    result = build_pcl(
        args.rgb, args.depth, args.camera,
        output=args.output,
        save_pcl=save_pcl,
        use_gpu=use_gpu,
        batch_size=args.batch_size
    )
    
    # Report results
    pcd, width, height = result
    if hasattr(pcd, 'points'):
        num_points = len(pcd.points)
    else:
        num_points = len(pcd)  # numpy array fallback
    
    expected_points = width * height
    
    if save_pcl:
        print("✅ Organized point cloud saved successfully")
    else:
        print(f"✅ Organized point cloud created (not saved)")
    
    print(f"📊 Final Results:")
    print(f"   Image size: {width}x{height} = {expected_points:,} pixels")
    print(f"   Point cloud size: {num_points:,} points")
    print(f"   Perfect mapping: {'✅ YES' if num_points == expected_points else '❌ NO'}")
    
    end_time = time.time()
    total_time = end_time - start_time
    pixels_per_second = expected_points / total_time
    
    print(f"⏱️  Total processing time: {total_time:.3f} seconds")
    print(f"🚀 Processing speed: {pixels_per_second/1e6:.2f} million pixels/second")

if __name__ == "__main__":
    main()