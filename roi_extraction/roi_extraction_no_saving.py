import numpy as np
import json
import open3d as o3d
from matplotlib.path import Path
import os
from collections import defaultdict

def load_polygons_from_json(json_file_path):
    """
    Load all polygon points from a JSON file with the provided structure.
    Returns a dictionary where keys are object labels and values are lists of polygons.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Dictionary to store all polygons by label
    polygons_by_label = {}
    
    # Extract masks from the JSON
    masks = data.get('masks', [])
    
    # Extract width and height for the image
    width = data.get('width')
    height = data.get('height')
    
    # Process each mask in the JSON
    for mask_index, mask in enumerate(masks):
        label = mask.get('label')
        points = mask.get('points', [])
        
        # Convert to a list of (x, y) tuples
        polygon_coords = [(point[0], point[1]) for point in points]
        
        # Add the polygon to the dictionary, creating a new list if the label doesn't exist yet
        if label not in polygons_by_label:
            polygons_by_label[label] = []
        
        # Store the polygon with its index
        polygons_by_label[label].append((mask_index, polygon_coords))
    
    return polygons_by_label, width, height

def create_combined_mask(polygons_by_label, width, height):
    """
    Create a combined mask for all polygons using vectorized operations.
    Returns a dictionary mapping (label, mask_index) to pixel indices.
    """
    # Create coordinate grids
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    pixel_coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
    
    # Dictionary to store indices for each polygon
    polygon_indices = {}
    
    print(f"Processing {sum(len(polys) for polys in polygons_by_label.values())} polygons...")
    
    for label, polygons in polygons_by_label.items():
        for mask_index, polygon_coords in polygons:
            if len(polygon_coords) < 3:  # Skip invalid polygons
                continue
                
            # Create matplotlib Path object for vectorized containment check
            path = Path(polygon_coords)
            
            # Get bounding box to limit search area
            poly_array = np.array(polygon_coords)
            min_x, min_y = np.floor(poly_array.min(axis=0)).astype(int)
            max_x, max_y = np.ceil(poly_array.max(axis=0)).astype(int)
            
            # Clamp to image bounds
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(width - 1, max_x)
            max_y = min(height - 1, max_y)
            
            # Create subset of coordinates within bounding box
            bbox_mask = ((pixel_coords[:, 0] >= min_x) & (pixel_coords[:, 0] <= max_x) & 
                        (pixel_coords[:, 1] >= min_y) & (pixel_coords[:, 1] <= max_y))
            
            if not np.any(bbox_mask):
                continue
                
            bbox_coords = pixel_coords[bbox_mask]
            
            # Vectorized containment check
            inside_mask = path.contains_points(bbox_coords)
            
            if np.any(inside_mask):
                # Get the original indices of points inside the polygon
                bbox_indices = np.where(bbox_mask)[0]
                inside_indices = bbox_indices[inside_mask]
                
                # Convert 2D indices to 1D point cloud indices
                polygon_indices[(label, mask_index)] = inside_indices
                
                print(f"  {label} polygon {mask_index}: {len(inside_indices)} pixels")
    
    return polygon_indices

def filter_valid_points(points, indices):
    """
    Filter out invalid points (NaN or infinite values) from the given indices.
    """
    valid_mask = np.logical_not(np.logical_or(
        np.isnan(points[indices]).any(axis=1),
        np.isinf(points[indices]).any(axis=1)
    ))
    return indices[valid_mask]

def get_pcd_dimensions(pcd_file_path):
    """
    Get the width and height of an organized point cloud from a PCD file.
    Handles both ASCII and binary PCD formats.
    """
    try:
        # First try to read the file as binary
        with open(pcd_file_path, 'rb') as f:
            header_lines = []
            line = f.readline().decode('ascii', errors='ignore').strip()
            
            # Read the header lines
            while line and not line.startswith('DATA'):
                header_lines.append(line)
                line = f.readline().decode('ascii', errors='ignore').strip()
            
            # Extract width and height from the header
            width, height = None, None
            for line in header_lines:
                if line.startswith('WIDTH'):
                    width = int(line.split()[-1])
                elif line.startswith('HEIGHT'):
                    height = int(line.split()[-1])
                
                # If both width and height are found, we can stop
                if width is not None and height is not None:
                    break
        
        # If we couldn't find the dimensions in the header, try alternative method
        if width is None or height is None:
            # Use Open3D to load the point cloud and get metadata
            pcd = o3d.io.read_point_cloud(pcd_file_path)
            
            # Try to get dimensions from loaded point cloud
            try:
                # Some point cloud libraries store these as attributes
                width = pcd.width
                height = pcd.height
            except:
                # If not available as attributes, we can use heuristics
                # For organized point clouds, width * height = number of points
                total_points = len(np.asarray(pcd.points))
                
                # Try to factorize using common image dimensions
                for w in [640, 1280, 1920, 3840, 512, 800, 1024, 2048]:
                    if total_points % w == 0:
                        width = w
                        height = total_points // w
                        break
    except Exception as e:
        print(f"Error reading PCD file: {e}")
        # As a fallback, use Open3D to load the point cloud
        try:
            pcd = o3d.io.read_point_cloud(pcd_file_path)
            # Try to infer dimensions from the point count
            total_points = len(np.asarray(pcd.points))
            
            # Try common resolutions
            for w in [640, 1280, 1920, 3840, 512, 800, 1024, 2048]:
                if total_points % w == 0:
                    width = w
                    height = total_points // w
                    break
                    
            # If still not found, try to find factors close to square root
            if width is None:
                sqrt_val = int(np.sqrt(total_points))
                for w in range(sqrt_val - 20, sqrt_val + 20):
                    if w > 0 and total_points % w == 0:
                        width = w
                        height = total_points // w
                        break
        except Exception as inner_e:
            print(f"Failed to load PCD with Open3D: {inner_e}")
    
    # If we couldn't find the dimensions, raise an error
    if width is None or height is None:
        raise ValueError("Could not extract width and height from PCD file")
    
    return width, height

def extract_and_save_labeled_point_clouds(pcd_file_path, json_file_path, output_dir):
    """
    Optimized version that extracts point clouds for all labeled objects simultaneously.
    
    Args:
        pcd_file_path (str): Path to the organized PCD file
        json_file_path (str): Path to the JSON file with polygon coordinates
        output_dir (str): Directory to save the extracted point clouds
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the width and height directly from the PCD file
    width, height = get_pcd_dimensions(pcd_file_path)
    print(f"Retrieved from PCD file - Width: {width}, Height: {height}")
    
    # Load the point cloud
    print(f"Loading point cloud from {pcd_file_path}...")
    point_cloud = o3d.io.read_point_cloud(pcd_file_path)
    
    # Get points as numpy array
    points = np.asarray(point_cloud.points)
    print(f"Loaded {len(points)} points.")
    
    # Get colors if available
    colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None
    if colors is not None:
        print(f"Point cloud has color information.")
    
    # Get normals if available
    normals = np.asarray(point_cloud.normals) if point_cloud.has_normals() else None
    if normals is not None:
        print(f"Point cloud has normal vectors.")
    
    # Load polygons from JSON
    print(f"Loading polygon data from {json_file_path}...")
    polygons_by_label, json_width, json_height = load_polygons_from_json(json_file_path)
    print(f"Found {len(polygons_by_label)} unique labels in the JSON file.")
    
    # Check if the JSON width/height match the PCD width/height
    if json_width is not None and json_height is not None:
        if json_width != width or json_height != height:
            print(f"Warning: JSON dimensions ({json_width}x{json_height}) don't match PCD dimensions ({width}x{height})")
            print(f"Using PCD dimensions: {width}x{height}")
    
    # Create combined mask for all polygons (OPTIMIZED PART)
    print("Creating combined mask for all polygons...")
    polygon_indices = create_combined_mask(polygons_by_label, width, height)
    
    # Group by label for directory creation
    labels_to_process = defaultdict(list)
    for (label, mask_index), indices in polygon_indices.items():
        labels_to_process[label].append((mask_index, indices))
    
    # Process and save point clouds
    print("Extracting and saving point clouds...")
    for label, polygon_data in labels_to_process.items():
        print(f"Processing label: {label}...")
        
        # Create a subdirectory for this label
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        
        # Process each polygon for this label
        for mask_index, indices in polygon_data:
            # Filter out invalid points
            valid_indices = filter_valid_points(points, indices)
            
            print(f"  Polygon {mask_index}: Found {len(valid_indices)} valid points (from {len(indices)} total)")
            
            if len(valid_indices) > 0:
                # Extract the subset of points for this specific polygon
                extracted_points = points[valid_indices]
                
                # Create a new point cloud with only these points
                extracted_cloud = o3d.geometry.PointCloud()
                extracted_cloud.points = o3d.utility.Vector3dVector(extracted_points)
                
                # Copy colors if available
                if colors is not None:
                    extracted_colors = colors[valid_indices]
                    extracted_cloud.colors = o3d.utility.Vector3dVector(extracted_colors)
                
                # Copy normals if available
                if normals is not None:
                    extracted_normals = normals[valid_indices]
                    extracted_cloud.normals = o3d.utility.Vector3dVector(extracted_normals)
                
                # Save the extracted point cloud with the label and polygon index as filename
                output_file = os.path.join(label_dir, f"{label}_roi_{mask_index}.pcd")
                
                # Write the point cloud file
                o3d.io.write_point_cloud(output_file, extracted_cloud)
                
                print(f"  Saved polygon {mask_index} with {len(valid_indices)} points to {output_file}")
            else:
                print(f"  No valid points found for polygon {mask_index}")
    
    print("Processing complete.")

def main():
    # File paths - you should update these to match your file paths
    pcd_file_path = "input.pcd"  # Path to your PCD file
    json_file_path = "polygon.json"  # Path to your JSON file with polygon coordinates
    output_dir = "extracted_pcls"  # Directory to save the extracted point clouds
    
    # Parse command line arguments if provided
    import argparse
    parser = argparse.ArgumentParser(description='Extract point clouds for labeled objects from polygons in an image (optimized version).')
    parser.add_argument('--pcd', type=str, default=pcd_file_path, help='Path to the PCD file')
    parser.add_argument('--json', type=str, default=json_file_path, help='Path to the JSON file with polygon coordinates')
    parser.add_argument('--output', type=str, default=output_dir, help='Directory to save the extracted point clouds')
    args = parser.parse_args()
    
    # Extract and save point clouds for each labeled object (optimized version)
    extract_and_save_labeled_point_clouds_optimized(args.pcd, args.json, args.output)

if __name__ == "__main__":
    main()