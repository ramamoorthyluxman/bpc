import numpy as np
import json
import open3d as o3d
from shapely.geometry import Point, Polygon
import os

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
    for mask in masks:
        label = mask.get('label')
        points = mask.get('points', [])
        
        # Convert to a list of (x, y) tuples
        polygon_coords = [(point[0], point[1]) for point in points]
        
        # Add the polygon to the dictionary, creating a new list if the label doesn't exist yet
        if label not in polygons_by_label:
            polygons_by_label[label] = []
        
        polygons_by_label[label].append(polygon_coords)
    
    return polygons_by_label, width, height

def extract_points_in_polygon(points, width, height, polygon_coords):
    """
    Extract points from a point cloud that correspond to pixels within a polygon.
    Uses a more efficient approach with rasterization and bounding box optimization.
    
    Args:
        points (numpy.ndarray): Point cloud points as numpy array
        width (int): Width of the organized point cloud
        height (int): Height of the organized point cloud
        polygon_coords (list): List of (x, y) tuples representing the polygon vertices
        
    Returns:
        list: Indices of points that fall within the polygon
    """
    # Create a Shapely polygon
    polygon = Polygon(polygon_coords)
    
    # Get the bounding box of the polygon to limit our search area
    minx, miny, maxx, maxy = polygon.bounds
    minx, miny = max(0, int(minx)), max(0, int(miny))
    maxx, maxy = min(width-1, int(maxx)+1), min(height-1, int(maxy)+1)
    
    # Initialize an empty list for indices
    indices_in_polygon = []
    
    # Only iterate over the bounding box area
    for y in range(miny, maxy):
        # Calculate the row offset once per row
        row_offset = y * width
        for x in range(minx, maxx):
            # Use Point.within(polygon) which is faster than polygon.contains(Point)
            if Point(x, y).within(polygon):
                # Calculate index directly
                idx = row_offset + x
                # Check if the point is valid (not NaN or infinite)
                if idx < len(points) and not (np.isnan(points[idx]).any() or np.isinf(points[idx]).any()):
                    indices_in_polygon.append(idx)
    
    return indices_in_polygon

def get_pcd_dimensions(pcd_file_path):
    """
    Get the width and height of an organized point cloud from a PCD file.
    Handles both ASCII and binary PCD formats.
    
    Args:
        pcd_file_path (str): Path to the PCD file
        
    Returns:
        tuple: (width, height) dimensions of the organized point cloud
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
    Extract point clouds for each labeled object in the JSON and save them separately.
    
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
    
    # Process each label and its polygons
    for label, polygons in polygons_by_label.items():
        print(f"Processing label: {label} with {len(polygons)} polygons...")
        all_indices = []
        
        # Process each polygon for this label
        for i, polygon_coords in enumerate(polygons):
            # Extract indices of points within this polygon
            indices = extract_points_in_polygon(points, width, height, polygon_coords)
            print(f"  Polygon {i+1}: Found {len(indices)} points")
            all_indices.extend(indices)
        
        # Remove duplicate indices
        all_indices = list(set(all_indices))
        
        if all_indices:
            # Extract the subset of points for this label
            extracted_points = points[all_indices]
            
            # Create a new point cloud with only these points
            extracted_cloud = o3d.geometry.PointCloud()
            extracted_cloud.points = o3d.utility.Vector3dVector(extracted_points)
            
            # Copy colors if available
            if colors is not None:
                extracted_colors = colors[all_indices]
                extracted_cloud.colors = o3d.utility.Vector3dVector(extracted_colors)
            
            # Copy normals if available
            if normals is not None:
                extracted_normals = normals[all_indices]
                extracted_cloud.normals = o3d.utility.Vector3dVector(extracted_normals)
            
            # Save the extracted point cloud with the label as filename
            # Save as PCD to preserve organization information
            output_file = os.path.join(output_dir, f"{label}.pcd")
            
            # Write the point cloud file
            o3d.io.write_point_cloud(output_file, extracted_cloud)
            
            print(f"Extracted {len(all_indices)} total points for {label}")
            print(f"Saved to {output_file}")
        else:
            print(f"No points found for {label}")
    
    print("Processing complete.")

def main():
    # File paths - you should update these to match your file paths
    pcd_file_path = "input.pcd"  # Path to your PCD file
    json_file_path = "polygon.json"  # Path to your JSON file with polygon coordinates
    output_dir = "extracted_pcls"  # Directory to save the extracted point clouds
    
    # Parse command line arguments if provided
    import argparse
    parser = argparse.ArgumentParser(description='Extract point clouds for labeled objects from polygons in an image.')
    parser.add_argument('--pcd', type=str, default=pcd_file_path, help='Path to the PCD file')
    parser.add_argument('--json', type=str, default=json_file_path, help='Path to the JSON file with polygon coordinates')
    parser.add_argument('--output', type=str, default=output_dir, help='Directory to save the extracted point clouds')
    parser.add_argument('--no-verify', action='store_true', help='Skip mapping verification')
    args = parser.parse_args()
    
    # Extract and save point clouds for each labeled object
    extract_and_save_labeled_point_clouds(args.pcd, args.json, args.output, verify=not args.no_verify)

if __name__ == "__main__":
    main()