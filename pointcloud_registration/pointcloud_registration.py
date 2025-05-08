#!/usr/bin/env python3
"""
Robust Point Cloud Registration for Partially Visible Objects

This script performs robust registration between a noisy, partially visible point cloud
and a reference 3D model, with handling for outliers and partial visibility.

Requirements:
- numpy
- open3d
- scipy
- matplotlib

Usage:
    python pointcloud_registration.py --reference <reference_ply_file> --target <target_pcd_file> --initial_pose <initial_rotation_matrix.txt> --output <output_transformation.txt> --visualize --scale_target --use_centroid_alignment
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse
import os
import copy
from scipy.spatial import KDTree


def load_point_cloud(filename):
    """Load point cloud from file."""
    print(f"Loading point cloud from {filename}")
    if filename.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(filename)
    elif filename.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(filename)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    
    # Print detailed information about the point cloud
    print(f"  Points: {len(pcd.points)}")
    print(f"  Has normals: {pcd.has_normals()}")
    print(f"  Has colors: {pcd.has_colors()}")
    
    # Compute and print the bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    print(f"  Bounding box min: [{min_bound[0]:.3f}, {min_bound[1]:.3f}, {min_bound[2]:.3f}]")
    print(f"  Bounding box max: [{max_bound[0]:.3f}, {max_bound[1]:.3f}, {max_bound[2]:.3f}]")
    print(f"  Bounding box size: [{max_bound[0]-min_bound[0]:.3f}, {max_bound[1]-min_bound[1]:.3f}, {max_bound[2]-min_bound[2]:.3f}]")
    
    return pcd


def save_point_cloud(pcd, filename):
    """Save point cloud to file for inspection."""
    print(f"Saving point cloud to {filename}")
    try:
        success = o3d.io.write_point_cloud(filename, pcd)
        if success:
            print(f"  Successfully saved to {filename}")
        else:
            print(f"  Failed to save to {filename}")
    except Exception as e:
        print(f"  Error saving point cloud: {e}")


def load_initial_pose(filename):
    """Load initial rotation matrix from file."""
    print(f"Loading initial pose from {filename}")
    try:
        rotation_matrix = np.loadtxt(filename)
        # Check if the file contains just a rotation matrix (3x3) or a full transformation matrix (4x4)
        if rotation_matrix.shape == (3, 3):
            # Create a 4x4 transformation matrix with identity translation
            transformation = np.eye(4)
            transformation[:3, :3] = rotation_matrix
        elif rotation_matrix.shape == (4, 4):
            transformation = rotation_matrix
        else:
            raise ValueError(f"Invalid matrix dimensions: {rotation_matrix.shape}")
        return transformation
    except Exception as e:
        print(f"Error loading initial pose: {e}")
        print("Using identity transformation as fallback")
        return np.eye(4)


def normalize_rotation_matrix(rotation_matrix):
    """Ensure the rotation matrix is orthogonal."""
    print("Normalizing rotation matrix...")
    
    # Print the original matrix
    print("  Original rotation matrix:")
    for row in rotation_matrix[:3, :3]:
        print(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")
    
    # Check if already orthogonal
    is_orthogonal = np.allclose(np.dot(rotation_matrix[:3, :3], rotation_matrix[:3, :3].T), np.eye(3), rtol=1e-3)
    if is_orthogonal:
        print("  Rotation matrix is already orthogonal")
        return rotation_matrix
    
    # Normalize using SVD
    u, _, vh = np.linalg.svd(rotation_matrix[:3, :3], full_matrices=True)
    orthogonal_rotation = np.dot(u, vh)
    
    # Create a new transformation matrix
    normalized_matrix = np.eye(4)
    normalized_matrix[:3, :3] = orthogonal_rotation
    normalized_matrix[:3, 3] = rotation_matrix[:3, 3]  # Keep the translation
    
    # Print the normalized matrix
    print("  Normalized rotation matrix:")
    for row in normalized_matrix[:3, :3]:
        print(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")
    
    return normalized_matrix


def remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """Remove statistical outliers from the point cloud."""
    print("Removing statistical outliers...")
    cleaned_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cleaned_pcd


def detect_and_remove_outliers_advanced(pcd, aggressive=False):
    """More advanced outlier removal to handle overlap cases.
    
    Uses multiple methods to filter outliers while preserving the core structure.
    
    Args:
        pcd: The point cloud to clean
        aggressive: Whether to use more aggressive filtering
    
    Returns:
        Cleaned point cloud
    """
    print("Performing advanced outlier removal...")
    
    # Create a copy to work with
    result = copy.deepcopy(pcd)
    original_points = len(result.points)
    
    # Step 1: Statistical outlier removal
    try:
        print("  Step 1: Statistical outlier removal")
        nb_neighbors = 20
        std_ratio = 1.5 if aggressive else 2.0
        result, _ = result.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        print(f"    Points after statistical outlier removal: {len(result.points)} "
              f"(removed {original_points - len(result.points)} points)")
    except Exception as e:
        print(f"    Error in statistical outlier removal: {e}")
    
    # Step 2: Radius outlier removal (removes isolated points)
    try:
        print("  Step 2: Radius outlier removal")
        nb_points = 5
        radius = 0.1  # Adjust based on your point cloud scale
        result, _ = result.remove_radius_outlier(nb_points=nb_points, radius=radius)
        print(f"    Points after radius outlier removal: {len(result.points)} "
              f"(removed {original_points - len(result.points)} points)")
    except Exception as e:
        print(f"    Error in radius outlier removal: {e}")
    
    # Step 3: DBSCAN clustering to keep the largest cluster (helps with overlap)
    if aggressive:
        try:
            print("  Step 3: DBSCAN clustering")
            # Compute DBSCAN clustering
            points = np.asarray(result.points)
            
            # Estimate a good eps parameter based on point cloud density
            bbox = result.get_axis_aligned_bounding_box()
            vol = np.prod(bbox.get_max_bound() - bbox.get_min_bound())
            density = len(points) / vol
            eps = 0.05 * np.cbrt(vol / len(points))  # Adaptive epsilon
            
            labels = np.array(result.cluster_dbscan(eps=eps, min_points=5))
            
            # Skip if clustering failed
            if labels.max() < 0:
                print("    DBSCAN clustering failed - no clusters found")
            else:
                # Find the largest cluster
                unique_labels, counts = np.unique(labels, return_counts=True)
                if unique_labels[0] == -1:  # Remove noise label (-1)
                    unique_labels = unique_labels[1:]
                    counts = counts[1:]
                
                if len(counts) > 0:
                    largest_cluster = unique_labels[np.argmax(counts)]
                    # Keep only the largest cluster
                    cluster_mask = (labels == largest_cluster)
                    cluster_points = points[cluster_mask]
                    
                    # Only use clustering if we keep at least 50% of the points
                    if len(cluster_points) > len(points) * 0.5:
                        filtered_pcd = o3d.geometry.PointCloud()
                        filtered_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                        
                        # Copy normals and colors if they exist
                        if result.has_normals():
                            normals = np.asarray(result.normals)
                            filtered_pcd.normals = o3d.utility.Vector3dVector(normals[cluster_mask])
                        if result.has_colors():
                            colors = np.asarray(result.colors)
                            filtered_pcd.colors = o3d.utility.Vector3dVector(colors[cluster_mask])
                        
                        result = filtered_pcd
                        print(f"    Points after DBSCAN clustering: {len(result.points)} "
                              f"(kept largest cluster with {len(result.points)} points)")
                    else:
                        print(f"    Skipping DBSCAN result - largest cluster too small ({len(cluster_points)} points)")
        except Exception as e:
            print(f"    Error in DBSCAN clustering: {e}")
    
    print(f"  Total points removed: {original_points - len(result.points)} "
          f"({(original_points - len(result.points)) / original_points * 100:.1f}%)")
    
    return result


def preprocess_point_cloud(pcd, voxel_size=0.005, is_reference=False):
    """Preprocess point cloud by downsampling and computing normals."""
    print("Preprocessing point cloud...")
    
    # For reference model (sparse mesh), densify first
    if is_reference:
        print("  Densifying reference mesh model...")
        # Try to create a triangle mesh from points
        try:
            # Create a triangle mesh from the point cloud
            # This uses the fact that the PLY likely has proper triangle information
            pcd_bbox = pcd.get_axis_aligned_bounding_box()
            pcd_volume = np.prod(pcd_bbox.get_extent())
            
            # Sample points uniformly on the surface of the mesh
            # Note that Open3D has a function to directly sample from PLY triangles
            # We calculate desired points based on volume and voxel size
            target_points = int(pcd_volume / (voxel_size**3) * 0.5)  # Adjustable density factor
            target_points = min(50000, max(5000, target_points))  # Reasonable limits
            
            # Convert to triangle mesh if it isn't already
            try:
                # This approach works if the input is already a proper mesh with triangles
                dense_pcd = pcd.sample_points_uniformly(number_of_points=target_points)
                print(f"  Sampled {len(dense_pcd.points)} points from mesh")
            except:
                # If not, try to create a mesh from the points (more error-prone)
                # First ensure we have normals
                if not pcd.has_normals():
                    pcd.estimate_normals()
                
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.5)
                dense_pcd = mesh.sample_points_uniformly(number_of_points=target_points)
                print(f"  Created and sampled mesh with {len(dense_pcd.points)} points")
            
            # Replace sparse point cloud with densified version
            pcd = dense_pcd
            
        except Exception as e:
            print(f"  Error in mesh sampling: {e}")
            print("  Continuing with original points")
    
    # Now perform normal downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals if they don't exist
    if not pcd_down.has_normals():
        try:
            pcd_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
            pcd_down.normalize_normals()
        except Exception as e:
            print(f"Warning: Could not estimate normals: {e}")
            print("Continuing without normals...")
    
    print(f"  Downsampled to {len(pcd_down.points)} points")
    return pcd_down


def prepare_dataset(source, target, voxel_size):
    """Prepare point clouds for registration by computing FPFH features."""
    print("Preparing dataset with FPFH features...")
    source_down = preprocess_point_cloud(source, voxel_size, is_reference=True)
    target_down = preprocess_point_cloud(target, voxel_size, is_reference=False)
    
    # Check if there are enough points for feature computation
    if len(source_down.points) < 10 or len(target_down.points) < 10:
        print("Warning: Not enough points after preprocessing!")
        print(f"Source points: {len(source_down.points)}, Target points: {len(target_down.points)}")
        print("Adjusting voxel size to ensure sufficient points...")
        
        # If we don't have enough points, try with a smaller voxel size
        if len(source_down.points) < 10:
            source_down = preprocess_point_cloud(source, voxel_size/2, is_reference=True)
            print(f"Adjusted source points: {len(source_down.points)}")
            
        if len(target_down.points) < 10:
            target_down = preprocess_point_cloud(target, voxel_size/2, is_reference=False)
            print(f"Adjusted target points: {len(target_down.points)}")
    
    print(f"Computing features for source ({len(source_down.points)} points) and target ({len(target_down.points)} points)")
    
    # Compute FPFH features
    try:
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    except Exception as e:
        print(f"Error computing FPFH features: {e}")
        print("Using a more conservative approach...")
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*10, max_nn=30))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*10, max_nn=30))
    
    return source_down, target_down, source_fpfh, target_fpfh


def scale_point_cloud(pcd, scale_factor=None, target_size=None, reference_pcd=None):
    """Scale a point cloud either by a specific factor or to match a target size/reference cloud.
    
    Args:
        pcd: The point cloud to scale
        scale_factor: Direct scaling factor to apply (if specified)
        target_size: Target diagonal size to scale to (if specified)
        reference_pcd: Reference point cloud to match scale (if specified)
    
    Returns:
        Scaled point cloud and the scaling transformation matrix
    """
    print(f"Scaling point cloud...")
    
    # Get current points
    points = np.asarray(pcd.points)
    
    # Get current size
    bbox = pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    diagonal = np.linalg.norm(max_bound - min_bound)
    print(f"  Current diagonal size: {diagonal:.3f}")
    
    # Determine scale factor
    if scale_factor is not None:
        # Use direct scale factor
        factor = scale_factor
        print(f"  Using specified scale factor: {factor:.3f}")
    elif target_size is not None:
        # Scale to target diagonal size
        factor = target_size / diagonal
        print(f"  Scaling to target size {target_size:.3f}, factor: {factor:.3f}")
    elif reference_pcd is not None:
        # Scale to match reference cloud
        ref_bbox = reference_pcd.get_axis_aligned_bounding_box()
        ref_min_bound = ref_bbox.get_min_bound()
        ref_max_bound = ref_bbox.get_max_bound()
        ref_diagonal = np.linalg.norm(ref_max_bound - ref_min_bound)
        factor = ref_diagonal / diagonal
        print(f"  Scaling to match reference diagonal {ref_diagonal:.3f}, factor: {factor:.3f}")
    else:
        factor = 1.0
        print("  No scaling applied (factor = 1.0)")
    
    # Apply scaling
    scaled_points = points * factor
    
    # Create new point cloud
    scaled_pcd = copy.deepcopy(pcd)
    scaled_pcd.points = o3d.utility.Vector3dVector(scaled_points)
    
    # Create scaling transformation matrix
    transformation = np.eye(4)
    transformation[0, 0] = factor
    transformation[1, 1] = factor
    transformation[2, 2] = factor
    
    # Check new size
    scaled_bbox = scaled_pcd.get_axis_aligned_bounding_box()
    scaled_min_bound = scaled_bbox.get_min_bound()
    scaled_max_bound = scaled_bbox.get_max_bound()
    scaled_diagonal = np.linalg.norm(scaled_max_bound - scaled_min_bound)
    print(f"  New diagonal size: {scaled_diagonal:.3f}")
    
    return scaled_pcd, transformation


def align_point_clouds_by_centroid(source, target):
    """Align point clouds by their centroids."""
    print("Aligning point clouds by centroid...")
    
    # Get centroids
    source_centroid = np.mean(np.asarray(source.points), axis=0)
    target_centroid = np.mean(np.asarray(target.points), axis=0)
    
    # Compute translation vector
    translation = target_centroid - source_centroid
    print(f"  Translation vector: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
    
    # Create transformation matrix
    transformation = np.eye(4)
    transformation[:3, 3] = translation
    
    # Apply transformation
    source_aligned = copy.deepcopy(source)
    source_aligned.transform(transformation)
    
    return source_aligned, transformation


def center_point_clouds(source, target):
    """Center both point clouds at the origin."""
    print("Centering point clouds at origin...")
    
    # Get centroids
    source_centroid = np.mean(np.asarray(source.points), axis=0)
    target_centroid = np.mean(np.asarray(target.points), axis=0)
    
    # Create transformation matrices
    source_transform = np.eye(4)
    source_transform[:3, 3] = -source_centroid
    
    target_transform = np.eye(4)
    target_transform[:3, 3] = -target_centroid
    
    # Apply transformations
    source_centered = copy.deepcopy(source)
    source_centered.transform(source_transform)
    
    target_centered = copy.deepcopy(target)
    target_centered.transform(target_transform)
    
    print(f"  Source centroid: [{source_centroid[0]:.3f}, {source_centroid[1]:.3f}, {source_centroid[2]:.3f}]")
    print(f"  Target centroid: [{target_centroid[0]:.3f}, {target_centroid[1]:.3f}, {target_centroid[2]:.3f}]")
    
    return source_centered, target_centered, source_transform, target_transform


def global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, initial_transformation=np.eye(4)):
    """Perform global registration using RANSAC and FPFH features."""
    print("Performing global registration with RANSAC...")
    
    # Set up RANSAC parameters
    distance_threshold = voxel_size * 1.5
    
    try:
        # Try running RANSAC registration
        print("Running RANSAC feature matching...")
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, # 3 points to estimate a rigid transformation
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        print(f"RANSAC result - fitness: {result.fitness}, RMSE: {result.inlier_rmse}")
        
        # If RANSAC failed (very low fitness), fall back to initial transformation
        if result.fitness < 0.05:
            print("RANSAC produced poor results, using initial transformation instead")
            result.transformation = initial_transformation
    except Exception as e:
        print(f"Error in RANSAC: {e}")
        print("Using initial transformation as fallback")
        # Create a dummy result object with the initial transformation
        result = type('obj', (object,), {
            'transformation': initial_transformation,
            'fitness': 0.0,
            'inlier_rmse': float('inf')
        })
    
    # If we have an initial estimate, we compare it with RANSAC result
    # and choose the better one
    if not np.array_equal(initial_transformation, np.eye(4)):
        try:
            eval_init = o3d.pipelines.registration.evaluate_registration(
                source_down, target_down, distance_threshold, initial_transformation)
            
            eval_ransac = o3d.pipelines.registration.evaluate_registration(
                source_down, target_down, distance_threshold, result.transformation)
            
            print(f"Initial pose fitness: {eval_init.fitness}, RANSAC fitness: {eval_ransac.fitness}")
            
            # Choose the better transformation as starting point
            if eval_init.fitness > eval_ransac.fitness:
                print("Initial pose estimation is better than RANSAC result, using initial pose")
                result.transformation = initial_transformation
        except Exception as e:
            print(f"Error comparing transformations: {e}")
            print("Continuing with best available transformation")
    
    return result


def local_registration(source, target, global_transform, voxel_size):
    """Perform robust ICP registration with outlier rejection."""
    print("Performing robust ICP registration...")
    # Apply global transformation first
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(global_transform)
    
    # Parameters for ICP
    distance_threshold = voxel_size * 2.0
    
    try:
        # First ensure target has normals for point-to-plane ICP
        has_target_normals = target.has_normals()
        if not has_target_normals:
            print("Target cloud does not have normals, computing them...")
            try:
                target.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
                target.normalize_normals()
                has_target_normals = True
            except Exception as e:
                print(f"Could not compute target normals: {e}")
                has_target_normals = False
        
        # First try point-to-plane ICP if normals are available
        if has_target_normals:
            print("Attempting point-to-plane ICP...")
            try:
                result = o3d.pipelines.registration.registration_icp(
                    source_transformed, target, distance_threshold, global_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
                )
                print(f"Point-to-plane ICP fitness: {result.fitness}")
            except Exception as e:
                print(f"Point-to-plane ICP failed: {e}")
                print("Falling back to point-to-point ICP...")
                result = o3d.pipelines.registration.registration_icp(
                    source_transformed, target, distance_threshold, global_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
                )
                print(f"Point-to-point ICP fitness: {result.fitness}")
        else:
            # Use point-to-point ICP if no normals
            print("Using point-to-point ICP (no normals available)...")
            result = o3d.pipelines.registration.registration_icp(
                source_transformed, target, distance_threshold, global_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
            )
            print(f"Point-to-point ICP fitness: {result.fitness}")
        
        # If standard ICP gave poor results, try a more robust approach
        if result.fitness < 0.3:
            print("Standard ICP gave low fitness. Trying with larger threshold...")
            result = o3d.pipelines.registration.registration_icp(
                source_transformed, target, distance_threshold * 2, global_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
            )
            print(f"More permissive ICP fitness: {result.fitness}")
        
        # Try TrimmedICP which is more robust to outliers, but check Open3D version first
        # as some versions don't support the trimmed_rmse parameter
        robust_distance_threshold = voxel_size * 3.0
        
        try:
            # Check if the version supports trimmed ICP with the outlier_ratio parameter
            # by examining the function signature
            import inspect
            sig = inspect.signature(o3d.pipelines.registration.registration_icp)
            if 'outlier_ratio' in sig.parameters:
                # Version supports outlier_ratio parameter
                print("Attempting trimmed ICP for outlier rejection...")
                result_trimmed = o3d.pipelines.registration.registration_icp(
                    source_transformed, target, robust_distance_threshold, result.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
                    outlier_ratio=0.2  # Trimmed ICP parameter - consider 80% of points
                )
                print(f"Trimmed ICP fitness: {result_trimmed.fitness}")
                
                # Check if trimmed ICP improved the result
                if result_trimmed.fitness > result.fitness:
                    return result_trimmed
            else:
                print("Trimmed ICP not supported in this Open3D version")
        except Exception as e:
            print(f"Could not use trimmed ICP: {e}")
            print("Using standard ICP result")
        
        return result
    
    except Exception as e:
        print(f"All ICP methods failed: {e}")
        print("Returning global transformation as fallback")
        # Create a dummy result with the global transformation
        fallback_result = type('obj', (object,), {
            'transformation': global_transform,
            'fitness': 0.0,
            'inlier_rmse': float('inf')
        })
        return fallback_result


def evaluate_registration(source, target, result, voxel_size, draw_results=True):
    """Evaluate and visualize registration results."""
    print("Evaluating registration result...")
    # Transform source to target for evaluation
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(result.transformation)
    
    # Evaluate using distance threshold
    distance_threshold = voxel_size * 2.0
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_transformed, target, distance_threshold, result.transformation)
    
    print(f"Registration Result:")
    print(f"Fitness: {evaluation.fitness}")
    print(f"Inlier RMSE: {evaluation.inlier_rmse}")
    
    # Check if correspondence_set attribute exists before accessing it
    if hasattr(evaluation, 'correspondence_set_size'):
        print(f"Correspondence Set Size: {evaluation.correspondence_set_size}")
    elif hasattr(evaluation, 'correspondence_set'):
        print(f"Correspondence Set Size: {len(evaluation.correspondence_set)}")
    else:
        print("Correspondence set information not available")
    
    if draw_results:
        try:
            # Visualization
            source_color = copy.deepcopy(source_transformed)
            target_color = copy.deepcopy(target)
            
            # Color the point clouds for visualization
            source_color.paint_uniform_color([1, 0.706, 0])  # Orange for source
            target_color.paint_uniform_color([0, 0.651, 0.929])  # Blue for target
            
            # Draw both point clouds
            o3d.visualization.draw_geometries([source_color, target_color],
                                            window_name="Registration Result",
                                            width=1280, height=720)
        except Exception as e:
            print(f"Visualization error: {e}")
            print("Continuing without visualization...")
    
    return evaluation


def compute_inlier_mask(source_transformed, target, distance_threshold):
    """Compute a mask for points in source that have a close match in target."""
    source_points = np.asarray(source_transformed.points)
    target_points = np.asarray(target.points)
    
    # Build KDTree for target points
    target_tree = KDTree(target_points)
    
    # Find nearest neighbors for each source point
    distances, _ = target_tree.query(source_points, k=1)
    
    # Create mask for inliers
    inlier_mask = distances < distance_threshold
    
    return inlier_mask


def export_results(transformation, filename):
    """Export the transformation matrix to a file."""
    print(f"Saving transformation to {filename}")
    np.savetxt(filename, transformation, fmt='%.6f')

def normalize_point_clouds(source, target):
    """Fully normalize point clouds to the origin and uniform scale.
    
    This function:
    1. Centers both clouds at origin
    2. Scales both to a standardized size 
    3. Ensures both are in the same approximate coordinate range
    """
    print("Performing full point cloud normalization...")
    
    # Create copies to work with
    source_norm = copy.deepcopy(source)
    target_norm = copy.deepcopy(target)
    
    # Step 1: Center both at origin (exactly at 0,0,0)
    source_centroid = np.mean(np.asarray(source_norm.points), axis=0)
    target_centroid = np.mean(np.asarray(target_norm.points), axis=0)
    
    print(f"  Source centroid: [{source_centroid[0]:.3f}, {source_centroid[1]:.3f}, {source_centroid[2]:.3f}]")
    print(f"  Target centroid: [{target_centroid[0]:.3f}, {target_centroid[1]:.3f}, {target_centroid[2]:.3f}]")
    
    # Create transformation matrices for centering
    source_center_transform = np.eye(4)
    source_center_transform[:3, 3] = -source_centroid
    
    target_center_transform = np.eye(4)
    target_center_transform[:3, 3] = -target_centroid
    
    # Apply centering transformations
    source_norm.transform(source_center_transform)
    target_norm.transform(target_center_transform)
    
    # Step 2: Scale both to a standard size (unit sphere)
    source_points = np.asarray(source_norm.points)
    target_points = np.asarray(target_norm.points)
    
    source_scale = np.max(np.linalg.norm(source_points, axis=1))
    target_scale = np.max(np.linalg.norm(target_points, axis=1))
    
    print(f"  Source max radius: {source_scale:.3f}")
    print(f"  Target max radius: {target_scale:.3f}")
    
    # Create scaling matrices
    source_scale_transform = np.eye(4)
    source_scale_transform[0, 0] = 1.0 / source_scale if source_scale > 0 else 1.0
    source_scale_transform[1, 1] = 1.0 / source_scale if source_scale > 0 else 1.0
    source_scale_transform[2, 2] = 1.0 / source_scale if source_scale > 0 else 1.0
    
    target_scale_transform = np.eye(4)
    target_scale_transform[0, 0] = 1.0 / target_scale if target_scale > 0 else 1.0
    target_scale_transform[1, 1] = 1.0 / target_scale if target_scale > 0 else 1.0
    target_scale_transform[2, 2] = 1.0 / target_scale if target_scale > 0 else 1.0
    
    # Apply scaling transformations
    source_norm.transform(source_scale_transform)
    target_norm.transform(target_scale_transform)
    
    # Combine transformations for later inversion
    source_norm_transform = np.dot(source_scale_transform, source_center_transform)
    target_norm_transform = np.dot(target_scale_transform, target_center_transform)
    
    return source_norm, target_norm, source_norm_transform, target_norm_transform


# Add this function to your code
def load_and_prepare_reference_model(filename, n_points=10000):
    """Load reference PLY as a mesh and sample points on its surface."""
    print(f"Loading reference model from {filename}")
    
    # Load as triangle mesh instead of point cloud
    mesh = o3d.io.read_triangle_mesh(filename)
    print(f"  Mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    # Sample points on the mesh surface
    pcd = mesh.sample_points_poisson_disk(n_points)
    print(f"  Sampled {len(pcd.points)} points on mesh surface")
    
    return pcd


def main():
    parser = argparse.ArgumentParser(description="Robust point cloud registration for partially visible objects")
    parser.add_argument("--reference", required=True, help="Path to the reference 3D model (PLY file)")
    parser.add_argument("--target", required=True, help="Path to the target point cloud (PLY or PCD file)")
    parser.add_argument("--initial_pose", required=True, help="Path to the initial rotation matrix (TXT file)")
    parser.add_argument("--output", default="transformation.txt", help="Output path for the final transformation matrix")
    parser.add_argument("--voxel_size", type=float, default=0.005, help="Voxel size for downsampling")
    parser.add_argument("--visualize", action="store_true", help="Visualize the registration results")
    parser.add_argument("--debug_dir", default="debug_output", help="Directory to save debug files")
    parser.add_argument("--results_dir", default="results", help="Directory to save results")
    parser.add_argument("--use_centroid_alignment", action="store_true", help="Align point clouds by centroid first")
    parser.add_argument("--scale_target", action="store_true", help="Scale target point cloud to match reference")
    parser.add_argument("--aggressive_filtering", action="store_true", help="Use more aggressive outlier filtering")
    parser.add_argument("--scale_factor", type=float, default=None, help="Manual scaling factor to apply to target")
    parser.add_argument("--use_full_normalization", action="store_true", help="Use full point cloud normalization")
    args = parser.parse_args()
    
    # Extract the target file basename to use as subfolder name
    target_basename = os.path.basename(args.target)
    # Remove the extension to get a clean name
    target_name = os.path.splitext(target_basename)[0]
    
    # Create debug directory with a subfolder named after the target file
    debug_subfolder = os.path.join(args.debug_dir, target_name)
    os.makedirs(debug_subfolder, exist_ok=True)
    
    # Also create a subfolder in the results directory
    # Extract the directory part from the output path
    output_dir = os.path.dirname(args.output)
    if not output_dir:  # If output is just a filename with no directory
        output_dir = "."
    results_subfolder = os.path.join(output_dir, target_name)
    os.makedirs(results_subfolder, exist_ok=True)
    
    # Modify the output path to be in the new subfolder
    output_filename = os.path.basename(args.output)
    new_output_path = os.path.join(results_subfolder, output_filename)
    
    try:
        # Load data
        # Load reference as mesh and sample points
        reference_pcd = load_and_prepare_reference_model(args.reference, n_points=10000)
        target_pcd = load_point_cloud(args.target)
        initial_transformation = load_initial_pose(args.initial_pose)
        
        # Save initial point clouds for debugging
        save_point_cloud(reference_pcd, os.path.join(debug_subfolder, "reference_original.ply"))
        save_point_cloud(target_pcd, os.path.join(debug_subfolder, "target_original.ply"))
        
        # Check if we need to handle scale differences
        reference_bbox = reference_pcd.get_axis_aligned_bounding_box()
        target_bbox = target_pcd.get_axis_aligned_bounding_box()
        reference_size = np.linalg.norm(reference_bbox.get_max_bound() - reference_bbox.get_min_bound())
        target_size = np.linalg.norm(target_bbox.get_max_bound() - target_bbox.get_min_bound())
        scale_ratio = reference_size / target_size
        print(f"Reference diagonal: {reference_size:.3f}, Target diagonal: {target_size:.3f}")
        print(f"Reference/Target scale ratio: {scale_ratio:.3f}")
        
        # Normalize the rotation matrix to ensure it's orthogonal
        initial_transformation = normalize_rotation_matrix(initial_transformation)
        
        # The critical change: Scale first if significant scale difference detected
        # If the Z scale is very different (like in this case with Z around 1600 vs Z around 60)
        # we need to scale first before outlier removal
        scale_before_filtering = False
        target_scale_factor = None
        
        if args.scale_target or args.scale_factor is not None:
            # Check if there's a significant scale difference in the Z dimension
            ref_z_range = reference_bbox.get_max_bound()[2] - reference_bbox.get_min_bound()[2]
            target_z_range = target_bbox.get_max_bound()[2] - target_bbox.get_min_bound()[2]
            
            z_scale_ratio = ref_z_range / target_z_range if target_z_range != 0 else 1.0
            z_offset_ratio = min(abs(reference_bbox.get_min_bound()[2]), abs(reference_bbox.get_max_bound()[2])) / \
                             min(abs(target_bbox.get_min_bound()[2]), abs(target_bbox.get_max_bound()[2]))
            
            print(f"Z-range ratio: {z_scale_ratio:.3f}")
            print(f"Z-offset ratio: {z_offset_ratio:.6f}")
            
            if abs(z_scale_ratio - 1.0) > 0.2 or z_offset_ratio < 0.01:
                print("Significant scale/offset detected - scaling before filtering")
                scale_before_filtering = True
                
                # Determine appropriate scale factor for target point cloud
                if args.scale_factor is not None:
                    target_scale_factor = args.scale_factor
                else:
                    # For extreme cases (like z around 1600 vs z around 60), use z-scale
                    if abs(target_bbox.get_min_bound()[2]) > 100 * abs(reference_bbox.get_min_bound()[2]):
                        # This is a special case for your specific data where Z is around 1600
                        target_scale_factor = ref_z_range / target_z_range * 0.01
                        print(f"Using z-scale approach with factor: {target_scale_factor:.6f}")
                    else:
                        target_scale_factor = scale_ratio
                
                print(f"Scaling target with factor: {target_scale_factor}")
                target_scaled, scale_transform = scale_point_cloud(target_pcd, scale_factor=target_scale_factor)
                save_point_cloud(target_scaled, os.path.join(args.debug_dir, "target_prescaled.ply"))
                
                # Replace the target with scaled version
                target_pcd = target_scaled
        
        # Basic outlier removal with statistical method only (much less aggressive)
        print("Performing basic outlier removal...")
        reference_clean = remove_statistical_outliers(reference_pcd)
        target_clean = remove_statistical_outliers(target_pcd)
        
        # Save basic cleaned point clouds
        save_point_cloud(reference_clean, os.path.join(args.debug_dir, "reference_basic_clean.ply"))
        save_point_cloud(target_clean, os.path.join(args.debug_dir, "target_basic_clean.ply"))
        
        # Now scale if we haven't already and it's requested
        if (args.scale_target or args.scale_factor is not None) and not scale_before_filtering:
            if args.scale_factor is not None:
                target_scale_factor = args.scale_factor
            else:
                # Calculate from diagonal ratio of cleaned point clouds
                ref_clean_bbox = reference_clean.get_axis_aligned_bounding_box()
                target_clean_bbox = target_clean.get_axis_aligned_bounding_box()
                ref_clean_size = np.linalg.norm(ref_clean_bbox.get_max_bound() - ref_clean_bbox.get_min_bound())
                target_clean_size = np.linalg.norm(target_clean_bbox.get_max_bound() - target_clean_bbox.get_min_bound())
                target_scale_factor = ref_clean_size / target_clean_size
            
            print(f"Scaling target after basic cleaning with factor: {target_scale_factor}")
            target_scaled, scale_transform = scale_point_cloud(target_clean, scale_factor=target_scale_factor)
            save_point_cloud(target_scaled, os.path.join(args.debug_dir, "target_scaled.ply"))
            
            # Replace the target with scaled version
            target_clean = target_scaled
        
        # Now we can do more aggressive filtering if requested, but with adapted parameters
        if args.aggressive_filtering:
            print("Performing additional outlier filtering...")
            
            # Get point cloud scale to adapt radius parameter
            ref_bbox = reference_clean.get_axis_aligned_bounding_box()
            ref_scale = np.mean(ref_bbox.get_extent())
            # Use a radius proportional to the point cloud scale - 5% of average dimension
            ref_radius = ref_scale * 0.05
            print(f"Adaptive reference radius: {ref_radius}")
            
            # More gentle radius outlier removal for reference
            try:
                ref_filtered, _ = reference_clean.remove_radius_outlier(nb_points=3, radius=ref_radius)
                print(f"Reference after radius filtering: {len(ref_filtered.points)} points")
                if len(ref_filtered.points) < 10:
                    print("Too few points after filtering, using basic cleaned reference")
                else:
                    reference_clean = ref_filtered
            except Exception as e:
                print(f"Reference radius filtering failed: {e}")
            
            # Same for target
            if len(target_clean.points) > 0:
                target_bbox = target_clean.get_axis_aligned_bounding_box()
                target_scale = np.mean(target_bbox.get_extent())
                target_radius = target_scale * 0.05
                print(f"Adaptive target radius: {target_radius}")
                
                try:
                    target_filtered, _ = target_clean.remove_radius_outlier(nb_points=3, radius=target_radius)
                    print(f"Target after radius filtering: {len(target_filtered.points)} points")
                    if len(target_filtered.points) < 10:
                        print("Too few points after filtering, using basic cleaned target")
                    else:
                        target_clean = target_filtered
                except Exception as e:
                    print(f"Target radius filtering failed: {e}")
        
        # Save cleaned point clouds
        save_point_cloud(reference_clean, os.path.join(args.debug_dir, "reference_clean_final.ply"))
        save_point_cloud(target_clean, os.path.join(args.debug_dir, "target_clean_final.ply"))
        
        # NEW APPROACH: Use full normalization instead of just centroid alignment
        # This should handle extreme coordinate differences better
        print("Performing full point cloud normalization...")
        reference_normalized, target_normalized, source_norm_transform, target_norm_transform = normalize_point_clouds(reference_clean, target_clean)
        
        # Save normalized point clouds
        save_point_cloud(reference_normalized, os.path.join(args.debug_dir, "reference_normalized.ply"))
        save_point_cloud(target_normalized, os.path.join(args.debug_dir, "target_normalized.ply"))
        
        # Apply initial transformation to normalized reference
        reference_initial = copy.deepcopy(reference_normalized)
        reference_initial.transform(initial_transformation)
        save_point_cloud(reference_initial, os.path.join(args.debug_dir, "reference_initial_transform.ply"))
        
        # Use the normalized point clouds for registration
        reference_pcd = reference_normalized
        target_pcd = target_normalized
        
        # Print point cloud information
        print(f"Reference point cloud has {len(reference_pcd.points)} points")
        print(f"Target point cloud has {len(target_pcd.points)} points")
        
        # Check if we have enough points to proceed
        if len(reference_pcd.points) < 10 or len(target_pcd.points) < 10:
            print("ERROR: Not enough points in the point clouds after preprocessing!")
            print("Try with different parameters or less aggressive filtering.")
            raise ValueError("Insufficient points for registration")
        
        # Set voxel size for downsampling - adapt based on point cloud density
        if args.voxel_size <= 0:
            # Auto-determine voxel size based on point cloud density
            bbox = reference_pcd.get_axis_aligned_bounding_box()
            volume = np.prod(bbox.get_extent())
            point_density = len(reference_pcd.points) / volume
            voxel_size = max(0.001, min(0.02, 5.0 / np.cbrt(point_density)))
            print(f"Auto-determined voxel size: {voxel_size}")
        else:
            voxel_size = args.voxel_size
            print(f"Using provided voxel size: {voxel_size}")
        
        # Prepare dataset with features
        print("Preparing dataset with features...")
        source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(reference_pcd, target_pcd, voxel_size)
        
        # Save downsampled point clouds
        save_point_cloud(source_down, os.path.join(args.debug_dir, "reference_downsampled.ply"))
        save_point_cloud(target_down, os.path.join(args.debug_dir, "target_downsampled.ply"))
        
        # Try direct centroid alignment for comparison
        source_aligned, centroid_transform = align_point_clouds_by_centroid(source_down, target_down)
        save_point_cloud(source_aligned, os.path.join(args.debug_dir, "reference_centroid_aligned.ply"))
        
        # Global registration using RANSAC with initial pose
        print("Step 3: Global registration...")
        global_result = global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, initial_transformation)
        
        # Save result of global registration
        source_global = copy.deepcopy(source_down)
        source_global.transform(global_result.transformation)
        save_point_cloud(source_global, os.path.join(args.debug_dir, "reference_after_global.ply"))
        
        # Refine alignment using robust ICP
        print("Step 4: Local registration with robust ICP...")
        icp_result = local_registration(reference_pcd, target_pcd, global_result.transformation, voxel_size)
        
        # Save result of ICP
        source_icp = copy.deepcopy(reference_pcd)
        source_icp.transform(icp_result.transformation)
        save_point_cloud(source_icp, os.path.join(args.debug_dir, "reference_after_icp.ply"))
        
        # Evaluate registration
        print("Step 5: Evaluating results...")
        evaluation = evaluate_registration(reference_pcd, target_pcd, icp_result, voxel_size, args.visualize)
        
        # If registration failed, try centroid alignment as fallback
        if evaluation.fitness < 0.01:
            print("Registration failed with ICP. Trying centroid alignment as fallback...")
            
            # Create a dummy result with centroid alignment
            centroid_result = type('obj', (object,), {
                'transformation': centroid_transform,
                'fitness': 0.0,
                'inlier_rmse': float('inf')
            })
            
            # Evaluate centroid alignment
            centroid_eval = evaluate_registration(source_down, target_down, centroid_result, voxel_size, False)
            print(f"Centroid alignment fitness: {centroid_eval.fitness}")
            
            # Use whichever is better
            if centroid_eval.fitness > evaluation.fitness:
                print("Centroid alignment is better than ICP. Using centroid alignment.")
                icp_result = centroid_result
                evaluation = centroid_eval
        
        # Reconstruct the full transformation including any preprocessing transformations
        # The complete transformation sequence is:
        # 1. source_norm_transform (normalization)
        # 2. registration transform
        # 3. inverse target_norm_transform (denormalization)
        
        final_transform = np.dot(np.linalg.inv(target_norm_transform), 
                                np.dot(icp_result.transformation, source_norm_transform))
        
        # Create a final result object with the combined transformation
        combined_result = type('obj', (object,), {
            'transformation': final_transform,
            'fitness': evaluation.fitness,
            'inlier_rmse': evaluation.inlier_rmse
        })
        
        # Save transformation matrix
        export_results(combined_result.transformation, new_output_path)
        
        print("\nRegistration pipeline completed!")
        print(f"Final transformation saved to: {new_output_path}")
        print(f"Registration fitness score: {evaluation.fitness}")
        print(f"Debug files saved to: {debug_subfolder}/")
        
    except Exception as e:
        print(f"Error in registration pipeline: {e}")
        import traceback
        traceback.print_exc()
        print("\nFalling back to initial transformation...")
        if 'initial_transformation' in locals():
            try:
                export_results(initial_transformation, new_output_path)
                print(f"Initial transformation saved to: {new_output_path}")
            except Exception as e2:
                print(f"Error saving initial transformation: {e2}")
                print("Registration failed completely.")

main()