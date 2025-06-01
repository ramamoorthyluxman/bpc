import numpy as np
import open3d as o3d
import copy
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
import os
import argparse
import sys
from PIL import Image

class PointCloudProcessor:
    def __init__(self):
        # Registration parameters
        self.voxel_size = 0.01  # Downsampling voxel size for registration
        self.nb_neighbors = 20  # Neighbors for normal estimation
        self.std_ratio = 2.0    # Standard deviation ratio for outlier removal
        self.distance_threshold = 0.1  # For RANSAC
        self.ransac_n = 3       # Minimum points for RANSAC
        self.ransac_iter = 100000  # RANSAC iterations
        self.icp_threshold = 0.1  # ICP convergence threshold
        self.icp_max_iter = 100    # Maximum ICP iterations
        
        # Size reduction parameters
        self.reference_points = 5000  # Target points for reference model
        self.max_scene_points = 25000  # Maximum points for scene
        self.max_file_size_mb = 10  # Maximum combined file size in MB
        
        # Path settings
        self.output_dir = 'registration_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Tracking registration progress
        self.fitness_history = []
        self.rmse_history = []
        self.convergence_iterations = 0
    
    def compute_registration_confidence(self, source_processed, target_processed, final_transform):
        """
        Compute a confidence score for the registration result.
        
        Args:
            source_processed: Processed source point cloud
            target_processed: Processed target point cloud  
            final_transform: Final transformation matrix
            
        Returns:
            confidence_score: Float between 0 and 1 (higher is better)
            confidence_details: Dictionary with detailed confidence metrics
        """
        # Apply final transformation to source
        source_transformed = copy.deepcopy(source_processed)
        source_transformed.transform(final_transform)
        
        # Compute final alignment metrics
        final_fitness = self.fitness_history[-1] if self.fitness_history else 0.0
        final_rmse = self.rmse_history[-1] if self.rmse_history else float('inf')
        
        # 1. Fitness score component (0 to 1, higher is better)
        fitness_score = min(final_fitness, 1.0)
        
        # 2. RMSE component (normalize by point cloud scale)
        source_points = np.asarray(source_processed.points)
        target_points = np.asarray(target_processed.points)
        
        # Estimate point cloud scale for RMSE normalization
        source_scale = np.std(np.linalg.norm(source_points, axis=1))
        target_scale = np.std(np.linalg.norm(target_points, axis=1))
        avg_scale = (source_scale + target_scale) / 2
        
        # Normalize RMSE by scale (lower normalized RMSE is better)
        normalized_rmse = final_rmse / max(avg_scale, 1e-6)
        rmse_score = max(0.0, 1.0 - min(normalized_rmse, 1.0))
        
        # 3. Convergence quality component
        convergence_score = 0.5  # Default neutral score
        if len(self.fitness_history) > 1:
            # Check if fitness improved during ICP
            fitness_improvement = self.fitness_history[-1] - self.fitness_history[0]
            convergence_score = min(1.0, max(0.0, 0.5 + fitness_improvement))
            
            # Penalize if RMSE got worse during refinement
            if len(self.rmse_history) > 1:
                rmse_change = self.rmse_history[0] - self.rmse_history[-1]  # Positive if RMSE improved
                if rmse_change < 0:  # RMSE got worse
                    convergence_score *= 0.8
        
        # 4. Geometric consistency check
        geometry_score = 0.5  # Default neutral score
        try:
            # Check overlap between transformed source and target
            source_tree = o3d.geometry.KDTreeFlann(target_processed)
            source_trans_points = np.asarray(source_transformed.points)
            
            overlap_count = 0
            overlap_threshold = self.voxel_size * 2  # Reasonable overlap threshold
            
            for point in source_trans_points:
                _, indices, distances = source_tree.search_radius_vector_3d(point, overlap_threshold)
                if len(indices) > 0:
                    overlap_count += 1
            
            overlap_ratio = overlap_count / len(source_trans_points)
            geometry_score = min(1.0, overlap_ratio)
            
        except Exception as e:
            geometry_score = 0.5
        
        # 5. Transformation reasonableness check
        transform_score = 1.0
        try:
            # Check if transformation is reasonable (not too extreme)
            rotation = final_transform[:3, :3]
            translation = final_transform[:3, 3]
            
            # Check rotation matrix properties
            det = np.linalg.det(rotation)
            if abs(det - 1.0) > 0.1:  # Should be close to 1 for proper rotation
                transform_score *= 0.7
            
            # Check orthogonality
            orthogonality_error = np.linalg.norm(np.dot(rotation, rotation.T) - np.eye(3))
            if orthogonality_error > 0.1:
                transform_score *= 0.8
            
            # Check if translation is reasonable relative to point cloud scale
            translation_magnitude = np.linalg.norm(translation)
            if translation_magnitude > avg_scale * 3:  # Very large translation
                transform_score *= 0.9
                
        except Exception as e:
            transform_score = 0.8
        
        # Combine all components with weights
        weights = {
            'fitness': 0.3,
            'rmse': 0.25,
            'convergence': 0.2,
            'geometry': 0.15,
            'transform': 0.1
        }
        
        confidence_score = (
            weights['fitness'] * fitness_score +
            weights['rmse'] * rmse_score +
            weights['convergence'] * convergence_score +
            weights['geometry'] * geometry_score +
            weights['transform'] * transform_score
        )
        
        # Compile detailed metrics
        confidence_details = {
            'overall_confidence': confidence_score,
            'final_fitness': final_fitness,
            'final_rmse': final_rmse,
            'normalized_rmse': normalized_rmse,
            'convergence_iterations': self.convergence_iterations,
            'overlap_ratio': geometry_score,
            'component_scores': {
                'fitness_score': fitness_score,
                'rmse_score': rmse_score,
                'convergence_score': convergence_score,
                'geometry_score': geometry_score,
                'transform_score': transform_score
            },
            'weights': weights
        }
        
        return confidence_score, confidence_details
    
    def save_confidence_report(self, confidence_score, confidence_details):
        """Save detailed confidence report to file"""
        confidence_file = os.path.join(self.output_dir, 'confidence_report.txt')
        
        with open(confidence_file, 'w') as f:
            f.write("Registration Confidence Report\n")
            f.write("=============================\n\n")
            
            f.write(f"Overall Confidence Score: {confidence_score:.3f}\n")
            f.write(f"Confidence Level: ")
            if confidence_score >= 0.8:
                f.write("High (Excellent registration)\n")
            elif confidence_score >= 0.6:
                f.write("Medium-High (Good registration)\n")
            elif confidence_score >= 0.4:
                f.write("Medium (Acceptable registration)\n")
            elif confidence_score >= 0.2:
                f.write("Low (Poor registration)\n")
            else:
                f.write("Very Low (Failed registration)\n")
            
            f.write("\nDetailed Metrics:\n")
            f.write(f"  Final Fitness: {confidence_details['final_fitness']:.6f}\n")
            f.write(f"  Final RMSE: {confidence_details['final_rmse']:.6f}\n")
            f.write(f"  Normalized RMSE: {confidence_details['normalized_rmse']:.6f}\n")
            f.write(f"  Convergence Iterations: {confidence_details['convergence_iterations']}\n")
            f.write(f"  Overlap Ratio: {confidence_details['overlap_ratio']:.3f}\n")
            
            f.write("\nComponent Scores:\n")
            for component, score in confidence_details['component_scores'].items():
                weight = confidence_details['weights'].get(component.replace('_score', ''), 0)
                f.write(f"  {component}: {score:.3f} (weight: {weight:.2f})\n")
            
            f.write("\nRecommendations:\n")
            if confidence_score < 0.4:
                f.write("- Registration quality is poor. Consider:\n")
                f.write("  * Adjusting registration parameters\n")
                f.write("  * Using better initial alignment\n")
                f.write("  * Checking point cloud quality and overlap\n")
            elif confidence_score < 0.6:
                f.write("- Registration quality is acceptable but could be improved:\n")
                f.write("  * Fine-tune ICP parameters\n")
                f.write("  * Consider multi-scale registration\n")
            else:
                f.write("- Registration quality is good to excellent\n")

    def transform_to_rt_format(self, transform):
        """
        Convert 4x4 transformation matrix to R, t format.
        
        Args:
            transform: 4x4 transformation matrix
            
        Returns:
            R_str: Rotation matrix as space-separated string (row-wise)
            t_str: Translation vector as space-separated string (in mm)
        """
        # Extract rotation matrix (3x3) and translation vector (3x1)
        R = transform[:3, :3]
        t = transform[:3, 3]
        
        # Convert translation to mm (assuming input is in meters)
        t_mm = t * 1000
        
        # Format as space-separated strings
        R_flat = R.flatten()
        R_str = ' '.join([f'{val:.6f}' for val in R_flat])
        t_str = ' '.join([f'{val:.6f}' for val in t_mm])
        
        return R_str, t_str
    
    def save_transformation_rt_format(self, transform, filename):
        """
        Save transformation matrix in R, t format as requested.
        
        Args:
            transform: 4x4 transformation matrix
            filename: Output file path
        """
        R_str, t_str = self.transform_to_rt_format(transform)
        
        with open(filename, 'w') as f:
            f.write(R_str + '\n')
            f.write(t_str + '\n')
    
    def load_transformation_rt_format(self, filename):
        """
        Load transformation from R, t format file.
        
        Args:
            filename: Input file path
            
        Returns:
            transform: 4x4 transformation matrix
        """
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Parse rotation matrix
        R_values = [float(x) for x in lines[0].strip().split()]
        R = np.array(R_values).reshape(3, 3)
        
        # Parse translation vector (convert from mm to meters)
        t_values = [float(x) for x in lines[1].strip().split()]
        t = np.array(t_values) / 1000  # Convert mm to meters
        
        # Construct 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        return transform
        
    def load_point_clouds(self, source_file, reference_file, scene_file=None):
        """Load all point clouds"""
        source = o3d.io.read_point_cloud(source_file)
        source_center = np.mean(np.asarray(source.points), axis=0)
        
        if reference_file.endswith('.ply'):
            # Load reference as point cloud
            reference = o3d.io.read_point_cloud(reference_file)
            
            # Also try loading as mesh to check if it contains triangle information
            try:
                reference_mesh = o3d.io.read_triangle_mesh(reference_file)
                if len(reference_mesh.triangles) > 0:
                    # If it's a proper mesh, sample points for registration
                    if len(reference_mesh.vertices) > self.reference_points:
                        reference_for_reg = reference_mesh.sample_points_uniformly(self.reference_points)
                    else:
                        reference_for_reg = reference
                else:
                    reference_for_reg = reference
            except Exception as e:
                reference_for_reg = reference
        else:
            reference = o3d.io.read_point_cloud(reference_file)
            reference_for_reg = reference
        
        reference_center = np.mean(np.asarray(reference_for_reg.points), axis=0)
        
        # Load scene if provided
        if scene_file:
            scene = o3d.io.read_point_cloud(scene_file)
        else:
            scene = None
        
        return source, reference, reference_for_reg, scene, source_center, reference_center
    
    def downsample_point_cloud(self, pcd, target_points, name="point cloud"):
        """Downsample point cloud to target number of points"""
        original_points = len(pcd.points)
        
        if original_points <= target_points:
            return copy.deepcopy(pcd)
        
        # Calculate voxel size to achieve target point count (approximation)
        reduction_factor = original_points / target_points
        voxel_size = 0.01 * (reduction_factor ** (1/3))
        
        # Try voxel downsampling first
        downsampled = pcd.voxel_down_sample(voxel_size)
        current_points = len(downsampled.points)

        if current_points < 10:
            downsampled = copy.deepcopy(pcd)
            current_points = len(downsampled.points)
        
        # If we're still over the limit, use random sampling as fallback
        if current_points > target_points:
            indices = np.random.choice(current_points, target_points, replace=False)
            points = np.asarray(downsampled.points)[indices]
            
            # Create a new point cloud with random samples
            random_sampled = o3d.geometry.PointCloud()
            random_sampled.points = o3d.utility.Vector3dVector(points)
            
            # Copy colors if available
            if len(downsampled.colors) > 0:
                colors = np.asarray(downsampled.colors)[indices]
                random_sampled.colors = o3d.utility.Vector3dVector(colors)
            
            downsampled = random_sampled
        
        return downsampled
    
    def preprocess_for_registration(self, pcd, name="point cloud"):
        """Prepare point cloud for registration"""
        # Make a copy to avoid modifying the original
        processed = copy.deepcopy(pcd)
        
        # Store original center
        original_center = np.mean(np.asarray(processed.points), axis=0)
        
        # Center the point cloud
        processed.points = o3d.utility.Vector3dVector(
            np.asarray(processed.points) - original_center)
        
        # Scale normalization
        points = np.asarray(processed.points)
        scale = np.max([np.linalg.norm(points, axis=1).max(), 1e-8])
        processed.points = o3d.utility.Vector3dVector(points / scale)
        
        # Voxel downsampling
        downsampled = processed.voxel_down_sample(self.voxel_size)
        
        # Statistical outlier removal for noise handling
        cleaned, ind = downsampled.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio)
        
        # Estimate normals for feature computation
        cleaned.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=self.voxel_size * 2, max_nn=30))
        
        # Compute FPFH features
        features = o3d.pipelines.registration.compute_fpfh_feature(
            cleaned, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100))
        
        return cleaned, features, scale, original_center
    
    def create_3part_2d_visualization(self, reference_original, source_original, reference_transformed):
        """
        Create a 3-part 2D visualization showing:
        Left: Neutral pose of reference model
        Center: Source pose 
        Right: Transformed pose of reference model
        """
        print("Creating 3-part 2D visualization...")
        
        # Extract points and debug
        ref_points = np.asarray(reference_original.points)
        src_points = np.asarray(source_original.points)
        trans_points = np.asarray(reference_transformed.points)
        
        print(f"Point counts - Reference: {len(ref_points)}, Source: {len(src_points)}, Transformed: {len(trans_points)}")
        
        if len(ref_points) == 0 or len(src_points) == 0 or len(trans_points) == 0:
            print("Error: One or more point clouds are empty!")
            return None
        
        # Use XY projection (front view) - most common and reliable
        ref_2d = ref_points[:, [0, 1]]
        src_2d = src_points[:, [0, 1]]
        trans_2d = trans_points[:, [0, 1]]
        
        # Calculate overall bounds for consistent scaling
        all_2d = np.vstack([ref_2d, src_2d, trans_2d])
        x_min, x_max = all_2d[:, 0].min(), all_2d[:, 0].max()
        y_min, y_max = all_2d[:, 1].min(), all_2d[:, 1].max()
        
        # Add padding
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0
        padding = 0.1
        x_pad = max(x_range * padding, 0.01)
        y_pad = max(y_range * padding, 0.01)
        
        xlims = [x_min - x_pad, x_max + x_pad]
        ylims = [y_min - y_pad, y_max + y_pad]
        
        # Create individual plots that we know work
        plot_configs = [
            (ref_2d, 'reference', 'green', 'Reference Model\n(Neutral Pose)'),
            (src_2d, 'source', 'red', 'Source Point Cloud\n(Target Pose)'),
            (trans_2d, 'transformed', 'blue', 'Reference Model\n(Transformed Pose)')
        ]
        
        individual_files = []
        
        for points_2d, name, color, title in plot_configs:
            plt.figure(figsize=(6, 6))
            plt.scatter(points_2d[:, 0], points_2d[:, 1], c=color, s=15, alpha=0.8)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True, alpha=0.3)
            plt.xlim(xlims)
            plt.ylim(ylims)
            
            individual_file = os.path.join(self.output_dir, f'{name}_individual.png')
            plt.savefig(individual_file, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            individual_files.append(individual_file)
            print(f"Individual {name} plot saved to {individual_file}")
        
        # Now combine the three working images side by side
        print("Combining individual images...")
        
        # Load the three images
        img1 = Image.open(individual_files[0])  # reference
        img2 = Image.open(individual_files[1])  # source  
        img3 = Image.open(individual_files[2])  # transformed
        
        # Get dimensions (they should all be the same)
        width, height = img1.size
        
        # Create new image with triple width
        combined_img = Image.new('RGB', (width * 3, height), 'white')
        
        # Paste the three images side by side
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (width, 0))
        combined_img.paste(img3, (width * 2, 0))
        
        # Save the combined image
        vis_filename = os.path.join(self.output_dir, 'registration_comparison_2d.png')
        combined_img.save(vis_filename)
        
        print(f"Combined 3-part visualization saved to {vis_filename}")
        return vis_filename
    
    def register_point_clouds(self, source, target):
        """Perform registration between source and target point clouds"""
        print("Starting registration process...")
        
        # Preprocess point clouds
        source_processed, source_feat, source_scale, source_center = self.preprocess_for_registration(source, "source")
        target_processed, target_feat, target_scale, target_center = self.preprocess_for_registration(target, "target")
        
        # Global registration (RANSAC)
        print("RANSAC Global Registration...")
        start = time.time()
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_processed, target_processed, source_feat, target_feat,
            mutual_filter=True,
            max_correspondence_distance=self.distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=self.ransac_n,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(self.distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                self.ransac_iter, 0.999))
        
        print(f"RANSAC registration took {time.time() - start:.3f} seconds")
        print(f"RANSAC Fitness: {result.fitness}")
        print(f"RANSAC RMSE: {result.inlier_rmse}")
        
        # Save results for progress tracking
        self.fitness_history = [result.fitness]
        self.rmse_history = [result.inlier_rmse]
        
        initial_transform = result.transformation
        
        # Refine with ICP
        print("Refining registration with ICP...")
        current_transform = initial_transform
        source_transformed = copy.deepcopy(source_processed)
        source_transformed.transform(current_transform)
        
        # Iterative ICP
        self.convergence_iterations = 0
        for i in range(self.icp_max_iter):
            # Run one ICP iteration
            result = o3d.pipelines.registration.registration_icp(
                source_transformed, target_processed, 
                self.icp_threshold, np.eye(4),  # Use identity as initial to avoid compounding
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=1))
            
            # Update the current transformation
            iter_transform = result.transformation
            current_transform = np.matmul(iter_transform, current_transform)
            
            # Transform the source for next iteration
            source_transformed = copy.deepcopy(source_processed)
            source_transformed.transform(current_transform)
            
            # Save results for progress tracking
            self.fitness_history.append(result.fitness)
            self.rmse_history.append(result.inlier_rmse)
            self.convergence_iterations = i + 1
            
            # Check for convergence
            if i > 0 and abs(self.fitness_history[-1] - self.fitness_history[-2]) < 1e-10:
                print(f"ICP converged after {i+1} iterations")
                break
        
        final_transform_processed = current_transform
        
        # Scale back the transformation
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = scale_matrix[1, 1] = scale_matrix[2, 2] = source_scale / target_scale
        final_transform = np.matmul(final_transform_processed, scale_matrix)
        
        # Compute confidence score
        confidence_score, confidence_details = self.compute_registration_confidence(
            source_processed, target_processed, final_transform_processed)
        
        # Save confidence report
        self.save_confidence_report(confidence_score, confidence_details)
        
        return final_transform, source_center, target_center, confidence_score
    
    def place_reference_in_scene(self, reference, scene, transform_src_to_ref, source_center, reference_center):
        """Place reference model in scene with correct transformation"""
        print("Placing reference model in scene...")
        
        # Calculate inverse transformation (reference to source)
        transform_ref_to_src = np.linalg.inv(transform_src_to_ref)
        
        # Adjust transformation to account for centering
        T_src = np.eye(4)
        T_src[:3, 3] = source_center
        
        T_ref = np.eye(4)
        T_ref[:3, 3] = -reference_center
        
        # The complete transformation: T_src * T_ref_to_src * T_ref
        adjusted_transform = np.matmul(T_src, np.matmul(transform_ref_to_src, T_ref))
        
        # Apply transformation to reference
        reference_transformed = copy.deepcopy(reference)
        reference_transformed.transform(adjusted_transform)
        
        # Color the reference for distinction
        reference_transformed.paint_uniform_color([1, 0.3, 0])  # Bright orange-red
        
        return reference_transformed
    
    def combine_and_reduce_size(self, reference_transformed, scene, output_file=None):
        """Combine reference and scene and reduce to target file size"""
        print("Combining and reducing size...")
        
        # Calculate target points for scene based on file size limit
        # Rough estimate: assume each point with RGB takes about 24 bytes
        # Target 90% of size limit to allow for file format overhead
        target_bytes = self.max_file_size_mb * 1024 * 1024 * 0.9
        reference_bytes = len(reference_transformed.points) * 24  # RGB points
        scene_bytes = target_bytes - reference_bytes
        scene_points = max(1000, int(scene_bytes / 24))  # At least 1000 points
        
        print(f"Target file size: {self.max_file_size_mb} MB")
        print(f"Allocating {len(reference_transformed.points)} points to reference")
        print(f"Allocating ~{scene_points} points to scene")
        
        # Downsample scene if needed
        if len(scene.points) > scene_points:
            scene_downsampled = self.downsample_point_cloud(scene, scene_points, "scene")
        else:
            scene_downsampled = copy.deepcopy(scene)
        
        # Combine point clouds
        combined = o3d.geometry.PointCloud()
        combined.points = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(reference_transformed.points), np.asarray(scene_downsampled.points))))
        
        # Combine colors
        if len(reference_transformed.colors) > 0 and len(scene_downsampled.colors) > 0:
            combined.colors = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(reference_transformed.colors), np.asarray(scene_downsampled.colors))))
        
        # Save final registration.ply file
        if output_file:
            print(f"Saving combined result to {output_file}...")
            o3d.io.write_point_cloud(output_file, combined, write_ascii=False, compressed=True)
            
            # Check file size
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"Final file size: {size_mb:.2f} MB")
            
            # Emergency reduction if still over limit
            if size_mb > self.max_file_size_mb:
                print(f"Warning: File size exceeds target of {self.max_file_size_mb} MB")
                print("Applying emergency size reduction...")
                
                # Keep reference points intact, reduce scene points further
                reduction_factor = size_mb / self.max_file_size_mb
                reference_points = len(reference_transformed.points)
                total_points = len(combined.points)
                reduced_total = int(total_points / reduction_factor)
                scene_points = max(1000, reduced_total - reference_points)
                
                print(f"Emergency reduction: Keeping reference ({reference_points} points) and reducing scene to {scene_points} points")
                
                # Create a new downsampled scene from the original
                scene_emergency = self.downsample_point_cloud(scene, scene_points, "scene (emergency)")
                
                # Combine again
                reduced_combined = o3d.geometry.PointCloud()
                reduced_combined.points = o3d.utility.Vector3dVector(
                    np.vstack((np.asarray(reference_transformed.points), np.asarray(scene_emergency.points))))
                
                if len(reference_transformed.colors) > 0 and len(scene_emergency.colors) > 0:
                    reduced_combined.colors = o3d.utility.Vector3dVector(
                        np.vstack((np.asarray(reference_transformed.colors), np.asarray(scene_emergency.colors))))
                
                # Save the emergency reduced result
                print(f"Saving emergency reduced result to {output_file}...")
                o3d.io.write_point_cloud(output_file, reduced_combined, write_ascii=False, compressed=True)
                
                # Check final size
                size_mb = os.path.getsize(output_file) / (1024 * 1024)
                print(f"Final file size after emergency reduction: {size_mb:.2f} MB")
                
                return reduced_combined
        
        return combined
    
    def plot_registration_metrics(self):
        """Plot registration metrics over iterations"""
        if len(self.fitness_history) < 2:
            print("Not enough iterations to plot metrics")
            return
            
        iterations = range(len(self.fitness_history))
        
        plt.figure(figsize=(12, 5))
        
        # Plot fitness
        plt.subplot(1, 2, 1)
        plt.plot(iterations, self.fitness_history, 'b-')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title('Registration Fitness Progress')
        plt.grid(True)
        
        # Plot RMSE
        plt.subplot(1, 2, 2)
        plt.plot(iterations, self.rmse_history, 'r-')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.title('Registration RMSE Progress')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        metrics_file = os.path.join(self.output_dir, 'registration_metrics.png')
        plt.savefig(metrics_file)
        print(f"Registration metrics plot saved to {metrics_file}")
    
    def run_registration_only(self, source_file=None, reference_file=None, 
                              source_pcl=None, reference_pcl=None):
        """
        Run registration without placement
        
        Args:
            source_file: Path to source (ROI) point cloud file (optional if source_pcl provided)
            reference_file: Path to reference point cloud/mesh file (optional if reference_pcl provided)
            source_pcl: Source point cloud object (optional if source_file provided)
            reference_pcl: Reference point cloud object (optional if reference_file provided)
            
        Returns:
            R_str, t_str: Transformation in R,t format
            source_center, reference_center: Point cloud centers
            confidence_score: Registration confidence score (0-1)
        """
        
        # Load or use provided point clouds
        if source_pcl is not None and reference_pcl is not None:
            # Use provided point cloud objects
            #print("Using provided point cloud objects...")
            source = copy.deepcopy(source_pcl)
            reference = copy.deepcopy(reference_pcl)
            reference_for_reg = copy.deepcopy(reference_pcl)
            
            # Calculate centers
            source_center = np.mean(np.asarray(source.points), axis=0)
            reference_center = np.mean(np.asarray(reference_for_reg.points), axis=0)
            
            #print(f"Source point cloud has {len(source.points)} points with center at {source_center}")
            #print(f"Reference has {len(reference_for_reg.points)} points with center at {reference_center}")
            
        else:
            # Load from files
            if source_file is None or reference_file is None:
                raise ValueError("Either provide point cloud objects (source_pcl, reference_pcl) or file paths (source_file, reference_file)")
            
            source, reference, reference_for_reg, _, source_center, reference_center = self.load_point_clouds(
                source_file, reference_file)
        
        # Register (now returns confidence score too)
        transform, source_center, reference_center, confidence_score = self.register_point_clouds(source, reference_for_reg)
        
        # Create transformed reference for visualization
        reference_transformed = copy.deepcopy(reference)
        reference_transformed.transform(transform)
        
        # Create 3-part 2D visualization
        self.create_3part_2d_visualization(reference, source, reference_transformed)
        
        # Save transformation in R,t format
        rt_transform_file = os.path.join(self.output_dir, 'transformation_rt.txt')
        self.save_transformation_rt_format(transform, rt_transform_file)
        
        # Also save the original 4x4 matrix for backward compatibility
        transform_file = os.path.join(self.output_dir, 'transformation_matrix.txt')
        np.savetxt(transform_file, transform)
        
        # Save centers
        source_center_file = os.path.join(self.output_dir, 'source_center.txt')
        reference_center_file = os.path.join(self.output_dir, 'reference_center.txt')
        np.savetxt(source_center_file, source_center)
        np.savetxt(reference_center_file, reference_center)
        
        # Plot metrics
        self.plot_registration_metrics()
        
        #print("\nRegistration completed successfully!")
        #print(f"Transformation matrix saved to {transform_file}")
        #print(f"Transformation R,t format saved to {rt_transform_file}")
        #print(f"Registration confidence: {confidence_score:.3f}")
        
        # Convert to R,t format for display and return
        R_str, t_str = self.transform_to_rt_format(transform)
        
        #print("\nTransformation in R,t format:")
        #print(f"R: {R_str}")
        #print(f"t: {t_str}")
        
        # Return confidence score as well
        return R_str, t_str, source_center, reference_center, confidence_score
    
    def run_placement_only(self, source_file, reference_file, scene_file, transform_file, output_file):
        """Run placement using pre-computed transformation"""
        # Check if transform file is in R,t format or 4x4 matrix format
        try:
            # Try loading as R,t format first
            transform = self.load_transformation_rt_format(transform_file)
            print("Loaded transformation from R,t format")
        except:
            # Fallback to loading as 4x4 matrix
            transform = np.loadtxt(transform_file)
            print("Loaded transformation from 4x4 matrix format")
        
        # Try to load centers from same directory as transform
        transform_dir = os.path.dirname(transform_file)
        source_center_file = os.path.join(transform_dir, 'source_center.txt')
        reference_center_file = os.path.join(transform_dir, 'reference_center.txt')
        
        if os.path.exists(source_center_file) and os.path.exists(reference_center_file):
            source_center = np.loadtxt(source_center_file)
            reference_center = np.loadtxt(reference_center_file)
            print("Loaded centers from files")
            
            # Load point clouds (without loading source if centers are available)
            _, reference, _, scene, _, _ = self.load_point_clouds(source_file, reference_file, scene_file)
        else:
            # Load all point clouds to compute centers
            _, reference, _, scene, source_center, reference_center = self.load_point_clouds(
                source_file, reference_file, scene_file)
            print("Computed centers from point clouds")
        
        # Place reference in scene
        reference_transformed = self.place_reference_in_scene(
            reference, scene, transform, source_center, reference_center)
        
        # Combine and reduce size - this creates the final registration.ply
        combined = self.combine_and_reduce_size(reference_transformed, scene, output_file)
        
        print("\nPlacement completed successfully!")
        print(f"Combined result saved to {output_file}")
        
        return combined
    
    def run_full_pipeline(self, source_pcl, reference_file_path, scene_pcl, 
                     output_file=None, visualize_and_save_results=True):
        """
        Run both registration and placement
        
        Args:
            source_pcl: Source point cloud object
            reference_file_path: Path to reference point cloud/mesh file
            scene_pcl: Scene point cloud object
            output_file: Path for combined output file (optional)
            visualize_and_save_results: If True, save visualizations and files; if False, skip unnecessary saves
            
        Returns:
            R_str, t_str: The computed transformation in R,t format
            confidence_score: Registration confidence score (0-1)
        """
        
        # Use provided point cloud objects and load reference from file
        #print("Using provided source and scene point cloud objects...")
        #print(f"Loading reference model: {reference_file_path}")
        
        # Load reference from file
        if reference_file_path.endswith('.ply'):
            # Load reference as point cloud
            reference = o3d.io.read_point_cloud(reference_file_path)
            
            # Also try loading as mesh to check if it contains triangle information
            try:
                reference_mesh = o3d.io.read_triangle_mesh(reference_file_path)
                if len(reference_mesh.triangles) > 0:
                    #print(f"Reference loaded as mesh with {len(reference_mesh.triangles)} triangles")
                    # If it's a proper mesh, sample points for registration
                    if len(reference_mesh.vertices) > self.reference_points:
                        reference_for_reg = reference_mesh.sample_points_uniformly(self.reference_points)
                        #print(f"Sampled {self.reference_points} points from mesh for registration")
                    else:
                        reference_for_reg = reference
                else:
                    reference_for_reg = reference
            except Exception as e:
                #print(f"Could not load reference as mesh: {str(e)}")
                reference_for_reg = reference
        else:
            reference = o3d.io.read_point_cloud(reference_file_path)
            reference_for_reg = reference
        
        # Use provided point cloud objects
        source = copy.deepcopy(source_pcl)
        scene = copy.deepcopy(scene_pcl)
        
        # Calculate centers
        source_center = np.mean(np.asarray(source.points), axis=0)
        reference_center = np.mean(np.asarray(reference_for_reg.points), axis=0)
        
        #print(f"Source point cloud has {len(source.points)} points with center at {source_center}")
        #print(f"Reference has {len(reference_for_reg.points)} points with center at {reference_center}")
        #print(f"Scene point cloud has {len(scene.points)} points")
        
        # Register (now returns confidence score too)
        transform, source_center, reference_center, confidence_score = self.register_point_clouds(source, reference_for_reg)
        
        if visualize_and_save_results:
            # Create transformed reference for visualization
            reference_transformed_for_vis = copy.deepcopy(reference)
            reference_transformed_for_vis.transform(transform)
            
            # Create 3-part 2D visualization
            self.create_3part_2d_visualization(reference, source, reference_transformed_for_vis)
            
            # Save transformation in R,t format
            rt_transform_file = os.path.join(self.output_dir, 'transformation_rt.txt')
            self.save_transformation_rt_format(transform, rt_transform_file)
            
            # Also save the original 4x4 matrix for backward compatibility
            transform_file = os.path.join(self.output_dir, 'transformation_matrix.txt')
            np.savetxt(transform_file, transform)
            
            # Save centers
            source_center_file = os.path.join(self.output_dir, 'source_center.txt')
            reference_center_file = os.path.join(self.output_dir, 'reference_center.txt')
            np.savetxt(source_center_file, source_center)
            np.savetxt(reference_center_file, reference_center)
            
            # Plot metrics
            self.plot_registration_metrics()
            
            #print("\nRegistration completed. Proceeding with placement...")
            #print(f"Registration confidence: {confidence_score:.3f}")
        else:
            print("\nRegistration completed (visualization skipped). Proceeding with placement...")
            print(f"Registration confidence: {confidence_score:.3f}")
        
        # Place reference in scene
        reference_transformed = self.place_reference_in_scene(
            reference, scene, transform, source_center, reference_center)
        
        # Combine and reduce size (conditionally save to file)
        output_file_param = output_file if visualize_and_save_results else None
        combined = self.combine_and_reduce_size(reference_transformed, scene, output_file_param)
        
        # Convert to R,t format for return
        R_str, t_str = self.transform_to_rt_format(transform)
        
        if visualize_and_save_results:
            #print("\nFull pipeline completed successfully!")
            if output_file:
                print(f"Combined result saved to {output_file}")
        else:
            print("\nFull pipeline completed successfully!")
            print("Results computed but files not saved (visualization disabled)")
        
        print(f"\nTransformation in R,t format:")
        print(f"R: {R_str}")
        print(f"t: {t_str}")
        print(f"Registration confidence: {confidence_score:.3f}")
        
        # Always return the transformation in R,t format plus confidence score
        return R_str, t_str, confidence_score

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration and Placement with Size Reduction')
    
    # Required arguments
    parser.add_argument('--source', type=str, required=True, help='Path to source (ROI) PCD file')
    parser.add_argument('--reference', type=str, required=True, help='Path to reference PLY/PCD file')
    
    # Optional arguments
    parser.add_argument('--scene', type=str, help='Path to scene PCD file')
    parser.add_argument('--transform', type=str, help='Path to pre-computed transformation file (R,t format or 4x4 matrix)')
    parser.add_argument('--output', type=str, default='registration.ply', help='Output file path for combined result')
    parser.add_argument('--output_dir', type=str, default='registration_results', help='Directory for output files')
    parser.add_argument('--max_size', type=float, default=10, help='Maximum file size in MB')
    parser.add_argument('--no_save', action='store_true', help='Skip saving visualizations and intermediate files')
    
    args = parser.parse_args()
    
    # Create processor
    processor = PointCloudProcessor()
    processor.output_dir = args.output_dir
    processor.max_file_size_mb = args.max_size
    os.makedirs(processor.output_dir, exist_ok=True)
    
    try:
        if args.transform and args.scene:
            # Placement only with pre-computed transformation
            processor.run_placement_only(
                args.source, args.reference, args.scene, args.transform, args.output)
        elif args.scene:
            # Full pipeline: registration + placement
            # Need to load point clouds as objects for the new interface
            source_pcl = o3d.io.read_point_cloud(args.source)
            scene_pcl = o3d.io.read_point_cloud(args.scene)
            
            R_str, t_str, confidence_score = processor.run_full_pipeline(
                source_pcl, args.reference, scene_pcl, args.output, 
                visualize_and_save_results=not args.no_save)
            
            print(f"\nFinal transformation in R,t format:")
            print(f"R: {R_str}")
            print(f"t: {t_str}")
            print(f"Registration confidence: {confidence_score:.3f}")
        else:
            # Registration only
            R_str, t_str, source_center, reference_center, confidence_score = processor.run_registration_only(args.source, args.reference)
            print(f"Registration confidence: {confidence_score:.3f}")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()