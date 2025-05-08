#!/usr/bin/env python3
"""
Robust Point Cloud Registration

This script implements a robust approach for registering a test point cloud to a reference 3D model.
It uses a combination of techniques for achieving high accuracy:
1. Feature-based global registration for good initial alignment
2. RANSAC for outlier rejection
3. Point-to-plane ICP for fine registration
4. Adaptive correspondence rejection
5. Coarse-to-fine strategy for optimal convergence

Dependencies:
- numpy
- open3d (primary library for point cloud processing)
- scipy
- matplotlib (for visualization if needed)
"""

import numpy as np
import open3d as o3d
import copy
import time
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RobustPointCloudRegistration:
    def __init__(self, voxel_size=0.05):
        """
        Initialize the registration class with parameters.
        
        Args:
            voxel_size: Base voxel size for downsampling (will be adapted in coarse-to-fine approach)
        """
        self.voxel_size = voxel_size
        
        # ICP parameters
        self.icp_distance_threshold = 2.0 * voxel_size  # Initial threshold
        self.icp_max_iterations = 100
        self.icp_convergence_criteria = 1e-6
        
        # RANSAC parameters for global registration
        self.ransac_n = 4  # Number of points to use for RANSAC
        self.ransac_max_iterations = 4000000
        self.ransac_confidence = 0.999
        self.ransac_distance_threshold = 1.5 * voxel_size
        
        # Feature matching parameters
        self.normal_radius = 2.0 * voxel_size
        self.feature_radius = 5.0 * voxel_size
        
        # Visualization flag
        self.visualize = False

    def load_point_clouds(self, test_pointcloud_path, reference_model_path):
        """
        Load test point cloud and reference 3D model.
        
        Args:
            test_pointcloud_path: Path to the test point cloud file
            reference_model_path: Path to the reference 3D model file (PLY)
            
        Returns:
            Tuple of (test_pcd, reference_pcd)
        """
        logger.info(f"Loading test point cloud from {test_pointcloud_path}")
        
        # Handle different file formats for test pointcloud
        if test_pointcloud_path.endswith('.txt'):
            # Assuming space-separated XYZ coordinates
            points = np.loadtxt(test_pointcloud_path, delimiter=' ')
            test_pcd = o3d.geometry.PointCloud()
            test_pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            if points.shape[1] >= 6:  # If file contains normals
                test_pcd.normals = o3d.utility.Vector3dVector(points[:, 3:6])
        elif test_pointcloud_path.endswith('.ply') or test_pointcloud_path.endswith('.pcd'):
            test_pcd = o3d.io.read_point_cloud(test_pointcloud_path)
        else:
            raise ValueError(f"Unsupported file format for test point cloud: {test_pointcloud_path}")
        
        logger.info(f"Loading reference model from {reference_model_path}")
        reference_pcd = o3d.io.read_triangle_mesh(reference_model_path)
        
        # Convert mesh to point cloud if needed
        if isinstance(reference_pcd, o3d.geometry.TriangleMesh):
            # Sample points from mesh surface
            reference_pcd = reference_pcd.sample_points_uniformly(
                number_of_points=min(1000000, int(len(test_pcd.points) * 2))
            )
        
        logger.info(f"Test point cloud: {len(test_pcd.points)} points")
        logger.info(f"Reference model: {len(reference_pcd.points)} points")
        
        return test_pcd, reference_pcd
    
    def load_initial_transform(self, transform_file_path):
        """
        Load initial transformation from file.
        
        Args:
            transform_file_path: Path to file containing transformation matrix
            
        Returns:
            4x4 numpy array with the transformation matrix
        """
        try:
            # Try to load as 3x3 rotation matrix
            R = np.loadtxt(transform_file_path)
            if R.shape == (3, 3):
                # Convert to 4x4 transform matrix
                transform = np.eye(4)
                transform[:3, :3] = R
                return transform
            elif R.shape == (4, 4):
                # Already a 4x4 transform
                return R
            else:
                raise ValueError(f"Unexpected transform matrix shape: {R.shape}")
        except Exception as e:
            logger.error(f"Error loading transform file: {e}")
            logger.warning("Using identity transform as fallback")
            return np.eye(4)

    def preprocess_point_clouds(self, source, target):
        """
        Preprocess point clouds for registration:
        1. Remove outliers
        2. Estimate normals if they don't exist
        3. Compute FPFH features for global registration
        
        Args:
            source: Source point cloud (test)
            target: Target point cloud (reference)
            
        Returns:
            Tuple of processed point clouds and their features
        """
        logger.info("Preprocessing point clouds...")
        
        # Create copies to avoid modifying the original data
        source_processed = copy.deepcopy(source)
        target_processed = copy.deepcopy(target)
        
        # Remove outliers (statistical)
        source_processed = source_processed.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)[0]
        target_processed = target_processed.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)[0]
        
        # Downsample for efficiency
        source_down = source_processed.voxel_down_sample(self.voxel_size)
        target_down = target_processed.voxel_down_sample(self.voxel_size)
        
        # Estimate normals if they don't exist
        if not source_down.has_normals():
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30))
        if not target_down.has_normals():
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30))
        
        # Orient normals consistently
        source_down.orient_normals_towards_camera_location(np.array([0, 0, 0]))
        target_down.orient_normals_towards_camera_location(np.array([0, 0, 0]))
        
        # Compute FPFH features
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.feature_radius, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.feature_radius, max_nn=100))
        
        return source_down, target_down, source_fpfh, target_fpfh

    def global_registration(self, source_down, target_down, source_fpfh, target_fpfh, initial_transform=None):
        """
        Perform global registration to get a good initial alignment.
        
        Args:
            source_down: Downsampled source point cloud
            target_down: Downsampled target point cloud
            source_fpfh: FPFH features for source
            target_fpfh: FPFH features for target
            initial_transform: Optional initial transform to apply first
            
        Returns:
            Transformation matrix from global registration
        """
        logger.info("Starting global registration...")
        
        source_transformed = copy.deepcopy(source_down)
        
        # Apply initial transform if provided
        if initial_transform is not None:
            source_transformed.transform(initial_transform)
            
            # Check if initial transform is already good enough
            result = o3d.pipelines.registration.evaluate_registration(
                source_transformed, target_down, self.ransac_distance_threshold)
            
            logger.info(f"Initial transform fitness: {result.fitness}, RMSE: {result.inlier_rmse}")
            
            if result.fitness > 0.5:  # If initial alignment is good
                logger.info("Initial transform is already good, skipping global registration")
                return initial_transform
        
        # Use RANSAC for robust global registration
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=self.ransac_distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=self.ransac_n,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.ransac_distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                self.ransac_max_iterations, self.ransac_confidence)
        )
        
        logger.info(f"Global registration result: fitness={result.fitness}, inlier_rmse={result.inlier_rmse}")
        
        # If we have an initial transform, compare results and choose the better one
        if initial_transform is not None:
            source_with_ransac = copy.deepcopy(source_down)
            source_with_ransac.transform(result.transformation)
            
            ransac_eval = o3d.pipelines.registration.evaluate_registration(
                source_with_ransac, target_down, self.ransac_distance_threshold)
            
            initial_eval = o3d.pipelines.registration.evaluate_registration(
                source_transformed, target_down, self.ransac_distance_threshold)
            
            if initial_eval.fitness > ransac_eval.fitness:
                logger.info("Initial transform is better than RANSAC result, using initial")
                return initial_transform
        
        return result.transformation

    def fine_registration(self, source, target, init_transform):
        """
        Perform fine registration using point-to-plane ICP with a coarse-to-fine approach.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            init_transform: Initial transformation from global registration
            
        Returns:
            Final transformation matrix
        """
        logger.info("Starting fine registration (ICP)...")
        
        # Initial transform
        current_transform = init_transform
        
        # Make copies of source and target
        source_copy = copy.deepcopy(source)
        target_copy = copy.deepcopy(target)
        
        # Make sure normals are computed for point-to-plane
        if not target_copy.has_normals():
            target_copy.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=30))
        
        # Apply initial transformation
        source_copy.transform(current_transform)
        
        # Coarse-to-fine approach with different voxel sizes
        voxel_sizes = [self.voxel_size*4, self.voxel_size*2, self.voxel_size]
        
        for i, voxel_size in enumerate(voxel_sizes):
            logger.info(f"ICP Stage {i+1}/{len(voxel_sizes)} (voxel_size={voxel_size})")
            
            # Downsample at current resolution
            source_down = source_copy.voxel_down_sample(voxel_size)
            target_down = target_copy.voxel_down_sample(voxel_size)
            
            # Ensure normals are available
            if not source_down.has_normals():
                source_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
            if not target_down.has_normals():
                target_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
            
            # Adjust distance threshold for current resolution
            distance_threshold = max(voxel_size * 2, 0.01)
            
            # Point-to-plane ICP
            result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, distance_threshold, current_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=self.icp_convergence_criteria,
                    relative_rmse=self.icp_convergence_criteria,
                    max_iteration=self.icp_max_iterations)
            )
            
            # Update current transformation for next stage
            current_transform = result.transformation
            source_copy.transform(result.transformation)
            
            logger.info(f"  Stage {i+1} result: fitness={result.fitness}, inlier_rmse={result.inlier_rmse}")
        
        # Final registration with adaptive rejection and increased max_iterations
        logger.info("Performing final refinement...")
        
        # Use point-to-point for final fine adjustment
        final_result = o3d.pipelines.registration.registration_icp(
            source_copy, target_copy, self.voxel_size * 0.5, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-8,
                relative_rmse=1e-8,
                max_iteration=200)
        )
        
        # Compute total transformation
        final_transform = np.matmul(final_result.transformation, current_transform)
        
        logger.info(f"Final registration result: fitness={final_result.fitness}, inlier_rmse={final_result.inlier_rmse}")
        
        return final_transform

    def evaluate_registration(self, source, target, transformation):
        """
        Evaluate the quality of registration.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            transformation: Transformation matrix
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating registration quality...")
        
        # Transform source to target frame
        source_transformed = copy.deepcopy(source)
        source_transformed.transform(transformation)
        
        # Evaluation at different thresholds to assess quality at different scales
        thresholds = [self.voxel_size * 0.5, self.voxel_size, self.voxel_size * 2]
        results = {}
        
        for threshold in thresholds:
            eval_result = o3d.pipelines.registration.evaluate_registration(
                source_transformed, target, threshold)
            
            results[f'threshold_{threshold:.5f}'] = {
                'fitness': eval_result.fitness,
                'inlier_rmse': eval_result.inlier_rmse,
                'correspondence_set_size': len(eval_result.correspondence_set)
            }
            
            logger.info(f"Evaluation at threshold {threshold:.5f}:")
            logger.info(f"  Fitness: {eval_result.fitness:.4f}")
            logger.info(f"  Inlier RMSE: {eval_result.inlier_rmse:.4f}")
            logger.info(f"  Correspondence set size: {len(eval_result.correspondence_set)}")
        
        # Calculate point-wise distances to estimate accuracy
        target_tree = KDTree(np.asarray(target.points))
        distances, _ = target_tree.query(np.asarray(source_transformed.points), k=1)
        
        results['mean_distance'] = np.mean(distances)
        results['median_distance'] = np.median(distances)
        results['max_distance'] = np.max(distances)
        results['std_distance'] = np.std(distances)
        
        logger.info(f"Mean distance: {results['mean_distance']:.4f}")
        logger.info(f"Median distance: {results['median_distance']:.4f}")
        logger.info(f"Max distance: {results['max_distance']:.4f}")
        logger.info(f"Std deviation: {results['std_distance']:.4f}")
        
        return results

    def visualize_registration(self, source, target, transformation):
        """
        Visualize registration results.
        
        Args:
            source: Source point cloud
            target: Target point cloud
            transformation: Final transformation matrix
        """
        source_transformed = copy.deepcopy(source)
        source_transformed.transform(transformation)
        
        # Color the point clouds
        source_transformed.paint_uniform_color([1, 0.706, 0])  # Yellow
        target.paint_uniform_color([0, 0.651, 0.929])  # Blue
        
        # Visualize point clouds
        o3d.visualization.draw_geometries([source_transformed, target],
                                          window_name="Registration Result",
                                          width=1280, height=720)

    def register(self, test_pointcloud_path, reference_model_path, initial_transform_path=None, output_path=None):
        """
        Main registration method.
        
        Args:
            test_pointcloud_path: Path to test point cloud file
            reference_model_path: Path to reference 3D model file
            initial_transform_path: Optional path to initial transform file
            output_path: Optional path to save results
            
        Returns:
            Tuple of (transformation_matrix, evaluation_metrics)
        """
        start_time = time.time()
        
        # Load point clouds
        source, target = self.load_point_clouds(test_pointcloud_path, reference_model_path)
        
        # Load initial transform if provided
        initial_transform = None
        if initial_transform_path:
            initial_transform = self.load_initial_transform(initial_transform_path)
            logger.info(f"Loaded initial transform:\n{initial_transform}")
        
        # Preprocess point clouds
        source_down, target_down, source_fpfh, target_fpfh = self.preprocess_point_clouds(source, target)
        
        # Global registration
        global_transform = self.global_registration(
            source_down, target_down, source_fpfh, target_fpfh, initial_transform)
        
        # Fine registration (ICP)
        final_transform = self.fine_registration(source, target, global_transform)
        
        # Evaluate registration quality
        evaluation = self.evaluate_registration(source, target, final_transform)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Registration completed in {elapsed_time:.2f} seconds")
        
        # Visualize if enabled
        if self.visualize:
            self.visualize_registration(source, target, final_transform)
        
        # Save results if output path is provided
        if output_path:
            # Save transformation matrix
            np.savetxt(f"{output_path}_transform.txt", final_transform)
            
            # Save transformed point cloud
            source_transformed = copy.deepcopy(source)
            source_transformed.transform(final_transform)
            o3d.io.write_point_cloud(f"{output_path}_registered.ply", source_transformed)
            
            # Save evaluation metrics
            with open(f"{output_path}_evaluation.txt", "w") as f:
                f.write(f"Registration Evaluation Results:\n")
                f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n\n")
                
                for threshold, metrics in evaluation.items():
                    if isinstance(metrics, dict):
                        f.write(f"Threshold {threshold}:\n")
                        for k, v in metrics.items():
                            f.write(f"  {k}: {v}\n")
                    else:
                        f.write(f"{threshold}: {metrics}\n")
            
            logger.info(f"Results saved to {output_path}")
        
        return final_transform, evaluation


def main():
    """
    Example usage of the RobustPointCloudRegistration class
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Robust Point Cloud Registration")
    parser.add_argument("--test", required=True, help="Path to test point cloud file (txt/ply/pcd)")
    parser.add_argument("--reference", required=True, help="Path to reference 3D model file (ply)")
    parser.add_argument("--init-transform", help="Path to initial transformation matrix file")
    parser.add_argument("--output", help="Path prefix for output files")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Base voxel size for registration")
    parser.add_argument("--visualize", action="store_true", help="Visualize registration results")
    
    args = parser.parse_args()
    
    # Create registration object
    registration = RobustPointCloudRegistration(voxel_size=args.voxel_size)
    registration.visualize = args.visualize
    
    # Perform registration
    transform, evaluation = registration.register(
        args.test, args.reference, args.init_transform, args.output)
    
    # Print final transformation
    print("\nFinal Transformation Matrix:")
    print(np.array2string(transform, precision=6, suppress_small=True))
    
    # Print key evaluation metrics
    print("\nRegistration Quality:")
    print(f"Mean distance: {evaluation['mean_distance']:.6f}")
    print(f"Median distance: {evaluation['median_distance']:.6f}")
    print(f"Max distance: {evaluation['max_distance']:.6f}")
    
    return 0


if __name__ == "__main__":
    main()