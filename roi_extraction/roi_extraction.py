import numpy as np
import json
import open3d as o3d
from matplotlib.path import Path
import os
from collections import defaultdict

def load_polygons_from_json_data(json_data):
    """
    Load all polygon points from JSON data (dict) with the provided structure.
    Returns a dictionary where keys are object labels and values are lists of polygons.
    """
    # Dictionary to store all polygons by label
    polygons_by_label = {}
    
    # Extract masks from the JSON
    masks = json_data.get('masks', [])
    
    # Extract width and height for the image
    width = json_data.get('width')
    height = json_data.get('height')
    
    # Process each mask in the JSON
    for mask_index, mask in enumerate(masks):
        label = mask.get('label')
        points = mask.get('points', [])
        geometric_center = mask.get('geometric_center', [0, 0])  # Get geometric center
        
        # Convert to a list of (x, y) tuples
        polygon_coords = [(point[0], point[1]) for point in points]
        
        # Add the polygon to the dictionary, creating a new list if the label doesn't exist yet
        if label not in polygons_by_label:
            polygons_by_label[label] = []
        
        # Store the polygon with its index and geometric center
        polygons_by_label[label].append((mask_index, polygon_coords, geometric_center))
    
    return polygons_by_label, width, height

def create_combined_mask(polygons_by_label, width, height):
    """
    Create a combined mask for all polygons using vectorized operations.
    Returns a dictionary mapping (label, mask_index) to pixel indices and geometric centers.
    """
    # Create coordinate grids
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    pixel_coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
    
    # Dictionary to store indices and centers for each polygon
    polygon_data = {}
    
    print(f"Processing {sum(len(polys) for polys in polygons_by_label.values())} polygons...")
    
    for label, polygons in polygons_by_label.items():
        for mask_index, polygon_coords, geometric_center in polygons:
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
                
                # Store polygon data with geometric center
                polygon_data[(label, mask_index)] = {
                    'indices': inside_indices,
                    'geometric_center': geometric_center
                }
                
                print(f"  {label} polygon {mask_index}: {len(inside_indices)} pixels, center at {geometric_center}")
    
    return polygon_data

def pixel_to_point_cloud_index(pixel_x, pixel_y, width):
    """
    Convert 2D pixel coordinates to 1D point cloud index.
    Assumes row-major ordering: index = y * width + x
    """
    return int(pixel_y * width + pixel_x)

def get_3d_center_location(points, pixel_x, pixel_y, width, height):
    """
    Get the 3D location corresponding to a 2D pixel coordinate.
    Returns None if the point is invalid (NaN or out of bounds).
    """
    # Check bounds
    if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
        return None
    
    # Convert to point cloud index
    pc_index = pixel_to_point_cloud_index(pixel_x, pixel_y, width)
    
    # Check if index is valid
    if pc_index >= len(points):
        return None
    
    # Get the 3D point
    point_3d = points[pc_index]
    
    # Check if point is valid (not NaN or infinite)
    if np.any(np.isnan(point_3d)) or np.any(np.isinf(point_3d)):
        return None
    
    return point_3d

def filter_valid_points(points, indices):
    """
    Filter out invalid points (NaN or infinite values) from the given indices.
    """
    valid_mask = np.logical_not(np.logical_or(
        np.isnan(points[indices]).any(axis=1),
        np.isinf(points[indices]).any(axis=1)
    ))
    return indices[valid_mask]

def extract_labeled_point_clouds(point_cloud, json_data, output_dir=None, save_rois=True):
    """
    Extract point clouds for all labeled objects based on polygon coordinates.
    
    Args:
        point_cloud (o3d.geometry.PointCloud): The input point cloud object
        json_data (dict): JSON data containing polygon coordinates and labels
        output_dir (str): Directory to save the extracted point clouds (optional if save_rois=False)
        save_rois (bool): Whether to save the ROI point clouds to files (default: True)
        
    Returns:
        list: Always returns list of tuples with extracted point clouds and their object IDs
              Format: [(point_cloud_object, object_id_dict), ...]
              where object_id_dict = {'label': str, 'mask_index': int, 'point_count': int, 'center_3d': [x, y, z] or None}
    """
    # Validate arguments
    if save_rois and output_dir is None:
        raise ValueError("output_dir is required when save_rois=True")
    
    # Create output directory if saving and it doesn't exist
    if save_rois:
        os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Load polygons from JSON data
    print(f"Loading polygon data from JSON...")
    polygons_by_label, width, height = load_polygons_from_json_data(json_data)
    print(f"Found {len(polygons_by_label)} unique labels in the JSON data.")
    print(f"Using JSON dimensions: {width}x{height}")
    
    # Create combined mask for all polygons (OPTIMIZED PART)
    print("Creating combined mask for all polygons...")
    polygon_data = create_combined_mask(polygons_by_label, width, height)
    
    # Group by label for processing
    labels_to_process = defaultdict(list)
    for (label, mask_index), data in polygon_data.items():
        labels_to_process[label].append((mask_index, data['indices'], data['geometric_center']))
    
    # List to store extracted point clouds with their IDs (ALWAYS created now)
    extracted_clouds_list = []

    # Process and save/return point clouds
    print("Extracting point clouds...")
    for label, polygon_data_list in labels_to_process.items():
        print(f"Processing label: {label}...")
        
        # Create a subdirectory for this label if saving
        if save_rois:
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)
        
        # Process each polygon for this label
        for mask_index, indices, geometric_center in polygon_data_list:
            # Filter out invalid points
            valid_indices = filter_valid_points(points, indices)
            
            # Get 3D center location from geometric center pixel
            center_3d = None
            if geometric_center and len(geometric_center) >= 2:
                pixel_x, pixel_y = geometric_center[0], geometric_center[1]
                center_3d = get_3d_center_location(points, pixel_x, pixel_y, width, height)
                if center_3d is not None:
                    center_3d = center_3d.tolist()  # Convert to list for JSON serialization
                    print(f"  Polygon {mask_index}: Center 2D: ({pixel_x}, {pixel_y}) -> 3D: ({center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f})")
                else:
                    print(f"  Polygon {mask_index}: Center 2D: ({pixel_x}, {pixel_y}) -> 3D: Invalid point")
            
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
                
                # ALWAYS store the extracted point cloud with its object ID in memory
                object_id = {
                    'label': label,
                    'mask_index': mask_index,
                    'point_count': len(valid_indices),
                    'center_3d': center_3d,  # Added 3D center location
                    'center_2d': geometric_center  # Also keep the original 2D center for reference
                }
                extracted_clouds_list.append((extracted_cloud, object_id))
                
                if save_rois:
                    # Save the extracted point cloud with the label and polygon index as filename
                    output_file = os.path.join(label_dir, f"{label}_roi_{mask_index}.pcd")
                    
                    # Write the point cloud file
                    o3d.io.write_point_cloud(output_file, extracted_cloud)
                    
                    print(f"  Saved polygon {mask_index} with {len(valid_indices)} points to {output_file}")
                else:
                    print(f"  Extracted polygon {mask_index} with {len(valid_indices)} points")
            else:
                print(f"  No valid points found for polygon {mask_index}")
    
    # ALWAYS return the extracted clouds list
    print(f"Processing complete. Returning {len(extracted_clouds_list)} extracted point clouds.")
    return extracted_clouds_list

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
    parser.add_argument('--no-save', action='store_true', help='Don\'t save ROI point clouds, just return data')
    args = parser.parse_args()
    
    # Load data from files for command line usage
    point_cloud = o3d.io.read_point_cloud(args.pcd)
    print(f"Loaded point cloud from {args.pcd}")
    
    with open(args.json, 'r') as f:
        json_data = json.load(f)
    print(f"Loaded JSON data from {args.json}")
    
    save_rois = not args.no_save
    
    # Extract point clouds using the function
    result = extract_labeled_point_clouds(
        point_cloud=point_cloud,
        json_data=json_data,
        output_dir=args.output if save_rois else None,
        save_rois=save_rois
    )
    
    # Result is ALWAYS a list now
    print(f"\nExtracted {len(result)} point cloud ROIs:")
    for i, (point_cloud_obj, object_id) in enumerate(result):
        label = object_id['label']
        mask_index = object_id['mask_index']
        point_count = object_id['point_count']
        center_3d = object_id['center_3d']
        center_2d = object_id['center_2d']
        
        center_str = f"3D: ({center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f})" if center_3d else "3D: Invalid"
        print(f"  ROI {i+1}: {label} (mask {mask_index}) - {point_count} points")
        print(f"    Center 2D: ({center_2d[0]}, {center_2d[1]}), {center_str}")

if __name__ == "__main__":
    main()