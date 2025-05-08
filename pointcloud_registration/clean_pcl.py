import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy import stats
import copy
import matplotlib.pyplot as plt
import os


def clean_point_cloud(points, 
                      depth_percentile=95.0,    # Remove points beyond this depth percentile
                      statistical_neighbors=50, # For statistical outlier removal
                      statistical_std=2.0,      # Standard deviation threshold
                      cluster_eps=0.02,         # DBSCAN clustering distance threshold
                      cluster_min_points=100,   # Minimum cluster size
                      depth_axis=2,             # Z-axis is typically the depth (0=X, 1=Y, 2=Z)
                      use_statistical=True,     # Enable statistical outlier removal
                      use_clustering=True,      # Enable cluster-based filtering
                      visualize=False):         # Show results
    """
    Clean a noisy point cloud by removing points that are wrongly segmented,
    particularly those that are far in depth.
    
    Args:
        points: numpy array of shape (N, 3) or Open3D PointCloud
        depth_percentile: Remove points beyond this depth percentile
        statistical_neighbors: Number of neighbors for statistical outlier removal
        statistical_std: Standard deviation multiplier for outlier removal
        cluster_eps: DBSCAN clustering distance threshold
        cluster_min_points: Minimum points to form a cluster
        depth_axis: Which axis represents depth (0=X, 1=Y, 2=Z)
        use_statistical: Enable statistical outlier removal
        use_clustering: Enable cluster-based filtering
        visualize: Show visualization of filtering process
        
    Returns:
        cleaned_points: numpy array of shape (M, 3) where M <= N
    """
    # Convert to Open3D PointCloud if input is numpy array
    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        pcd = copy.deepcopy(points)
    
    # Store original for visualization
    original_pcd = copy.deepcopy(pcd)
    
    print(f"Original point cloud: {len(pcd.points)} points")
    
    # Remove NaN points
    points_array = np.asarray(pcd.points)
    mask = ~np.isnan(points_array).any(axis=1)
    valid_points = points_array[mask]
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    print(f"After removing NaN points: {len(pcd.points)} points")
    
    # STEP 1: Remove depth outliers using percentile
    points_array = np.asarray(pcd.points)
    if len(points_array) > 0:
        depth_values = points_array[:, depth_axis]
        depth_threshold = np.percentile(depth_values, depth_percentile)
        mask = depth_values <= depth_threshold
        filtered_points = points_array[mask]
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        print(f"After depth filtering ({depth_percentile}th percentile): {len(pcd.points)} points")
    
    # STEP 2: Statistical outlier removal
    if use_statistical and len(pcd.points) > statistical_neighbors:
        pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=statistical_neighbors,
            std_ratio=statistical_std
        )
        print(f"After statistical outlier removal: {len(pcd.points)} points")
    
    # STEP 3: Cluster-based filtering (keep only the largest clusters)
    if use_clustering and len(pcd.points) > cluster_min_points:
        points_array = np.asarray(pcd.points)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=cluster_eps, min_samples=cluster_min_points).fit(points_array)
        labels = clustering.labels_
        
        # Count number of clusters (excluding noise with label -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Found {n_clusters} clusters")
        
        if n_clusters > 0:
            # Find the largest clusters (excluding noise with label -1)
            clusters = {}
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)
            
            # Sort clusters by size (largest first)
            sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
            
            # Keep only points in the largest cluster(s)
            # You can modify this to keep N largest clusters if needed
            largest_cluster_indices = sorted_clusters[0][1]
            filtered_points = points_array[largest_cluster_indices]
            pcd.points = o3d.utility.Vector3dVector(filtered_points)
            print(f"After keeping largest cluster: {len(pcd.points)} points")
    
    # Visualize the results if requested
    if visualize:
        # Create visualization of original vs cleaned point cloud
        original_pcd.paint_uniform_color([1, 0, 0])  # Red for original
        pcd.paint_uniform_color([0, 1, 0])           # Green for cleaned
        
        # Visualize
        o3d.visualization.draw_geometries([original_pcd, pcd], 
                                         window_name="Original (Red) vs Cleaned (Green)")
    
    # Return the cleaned point cloud
    return pcd


def process_pcl_file(input_file, output_file=None, 
                    depth_percentile=95.0,
                    statistical_neighbors=50,
                    statistical_std=2.0,
                    cluster_eps=0.02,
                    cluster_min_points=100,
                    depth_axis=2,
                    use_statistical=True,
                    use_clustering=True,
                    visualize=True):
    """
    Process a PCL/PCD file to remove outliers, especially those far in depth.
    
    Args:
        input_file: Path to input .pcd file
        output_file: Path to save cleaned .pcd file (if None, will use input_file_cleaned.pcd)
        Other parameters: Same as clean_point_cloud()
        
    Returns:
        Path to saved cleaned point cloud file
    """
    # Set default output file if not specified
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_cleaned{ext}"
    
    # Load the PCL/PCD file
    try:
        print(f"Loading point cloud from {input_file}...")
        pcd = o3d.io.read_point_cloud(input_file)
        print(f"Loaded {len(pcd.points)} points")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    # Check if the file was loaded correctly
    if len(pcd.points) == 0:
        print("Warning: Loaded point cloud has zero points!")
        return None
    
    # Clean the point cloud
    cleaned_pcd = clean_point_cloud(
        pcd,
        depth_percentile=depth_percentile,
        statistical_neighbors=statistical_neighbors,
        statistical_std=statistical_std,
        cluster_eps=cluster_eps,
        cluster_min_points=cluster_min_points,
        depth_axis=depth_axis,
        use_statistical=use_statistical,
        use_clustering=use_clustering,
        visualize=visualize
    )
    
    # Save the cleaned point cloud
    try:
        o3d.io.write_point_cloud(output_file, cleaned_pcd)
        print(f"Saved cleaned point cloud to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error saving file: {e}")
        return None


def advanced_pcl_cleaning(input_file, output_file=None, 
                         depth_axis=2,
                         object_depth_threshold=0.1,
                         visualize=True):
    """
    Advanced PCL file cleaning that maintains object depth consistency.
    
    Args:
        input_file: Path to input .pcd file
        output_file: Path to save cleaned .pcd file
        depth_axis: Which axis represents depth (0=X, 1=Y, 2=Z)
        object_depth_threshold: Maximum allowed depth variation
        visualize: Show visualization
        
    Returns:
        Path to saved cleaned point cloud file
    """
    # Set default output file if not specified
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_object_cleaned{ext}"
    
    # Load the PCL/PCD file
    try:
        print(f"Loading point cloud from {input_file}...")
        pcd = o3d.io.read_point_cloud(input_file)
        print(f"Loaded {len(pcd.points)} points")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    # First apply standard cleaning
    cleaned_pcd = clean_point_cloud(
        pcd,
        depth_percentile=95.0,
        statistical_neighbors=50,
        statistical_std=2.0,
        cluster_eps=0.02,
        depth_axis=depth_axis,
        visualize=False
    )
    
    # Store for visualization
    standard_cleaned = copy.deepcopy(cleaned_pcd)
    
    # Now apply object-specific depth consistency check
    points_array = np.asarray(cleaned_pcd.points)
    
    if len(points_array) > 0:
        # Calculate depth statistics
        depth_values = points_array[:, depth_axis]
        
        # Find median depth (more robust than mean)
        median_depth = np.median(depth_values)
        
        # Keep points that are within threshold of median depth
        depth_mask = np.abs(depth_values - median_depth) <= object_depth_threshold
        filtered_points = points_array[depth_mask]
        
        # Update colors if they exist
        if len(cleaned_pcd.colors) > 0:
            colors_array = np.asarray(cleaned_pcd.colors)
            filtered_colors = colors_array[depth_mask]
            cleaned_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        
        # Update point cloud
        cleaned_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        
        print(f"After depth consistency check: {len(cleaned_pcd.points)} points")
    
    # Visualize if requested
    if visualize:
        pcd.paint_uniform_color([1, 0, 0])  # Red - original
        standard_cleaned.paint_uniform_color([0, 0, 1])  # Blue - standard cleaned
        cleaned_pcd.paint_uniform_color([0, 1, 0])  # Green - object cleaned
        o3d.visualization.draw_geometries([pcd, standard_cleaned, cleaned_pcd], 
                                         window_name="Original (Red) vs Standard (Blue) vs Object-Cleaned (Green)")
    
    # Save the cleaned point cloud
    try:
        o3d.io.write_point_cloud(output_file, cleaned_pcd)
        print(f"Saved object-cleaned point cloud to {output_file}")
        return output_file
    except Exception as e:
        print(f"Error saving file: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean a PCL/PCD file by removing outliers far in depth")
    parser.add_argument("input_file", help="Path to input .pcd file")
    parser.add_argument("--output", "-o", help="Path to output .pcd file")
    parser.add_argument("--depth-percentile", "-p", type=float, default=98.0, 
                       help="Remove points beyond this depth percentile")
    parser.add_argument("--statistical-neighbors", "-n", type=int, default=50,
                       help="Number of neighbors for statistical outlier removal")
    parser.add_argument("--statistical-std", "-s", type=float, default=2.0,
                       help="Standard deviation threshold for statistical outlier removal")
    parser.add_argument("--cluster-eps", "-e", type=float, default=0.02,
                       help="DBSCAN clustering distance threshold")
    parser.add_argument("--cluster-min-points", "-m", type=int, default=100,
                       help="Minimum points to form a cluster")
    parser.add_argument("--depth-axis", "-d", type=int, default=2,
                       help="Axis representing depth (0=X, 1=Y, 2=Z)")
    parser.add_argument("--no-statistical", action="store_true",
                       help="Disable statistical outlier removal")
    parser.add_argument("--no-clustering", action="store_true",
                       help="Disable cluster-based filtering")
    parser.add_argument("--no-visualization", action="store_true",
                       help="Disable visualization")
    parser.add_argument("--advanced", "-a", action="store_true",
                       help="Use advanced object-specific cleaning")
    parser.add_argument("--object-depth-threshold", "-t", type=float, default=0.1,
                       help="Maximum allowed depth variation within object (for advanced mode)")
    
    args = parser.parse_args()
    
    if args.advanced:
        advanced_pcl_cleaning(
            args.input_file,
            args.output,
            depth_axis=args.depth_axis,
            object_depth_threshold=args.object_depth_threshold,
            visualize=not args.no_visualization
        )
    else:
        process_pcl_file(
            args.input_file,
            args.output,
            depth_percentile=args.depth_percentile,
            statistical_neighbors=args.statistical_neighbors,
            statistical_std=args.statistical_std,
            cluster_eps=args.cluster_eps,
            cluster_min_points=args.cluster_min_points,
            depth_axis=args.depth_axis,
            use_statistical=not args.no_statistical,
            use_clustering=not args.no_clustering,
            visualize=not args.no_visualization
        )