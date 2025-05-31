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
        
        print(f"Transformation saved in R,t format to {filename}")
    
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
        
        print(f"Transformation loaded from R,t format file: {filename}")
        return transform
        
    def load_point_clouds(self, source_file, reference_file, scene_file=None):
        """Load all point clouds"""
        print(f"Loading source point cloud (ROI): {source_file}")
        source = o3d.io.read_point_cloud(source_file)
        source_center = np.mean(np.asarray(source.points), axis=0)
        print(f"Source point cloud has {len(source.points)} points with center at {source_center}")
        
        print(f"Loading reference model: {reference_file}")
        if reference_file.endswith('.ply'):
            # Load reference as point cloud
            reference = o3d.io.read_point_cloud(reference_file)
            
            # Also try loading as mesh to check if it contains triangle information
            try:
                reference_mesh = o3d.io.read_triangle_mesh(reference_file)
                if len(reference_mesh.triangles) > 0:
                    print(f"Reference loaded as mesh with {len(reference_mesh.triangles)} triangles")
                    # If it's a proper mesh, sample points for registration
                    if len(reference_mesh.vertices) > self.reference_points:
                        reference_for_reg = reference_mesh.sample_points_uniformly(self.reference_points)
                        print(f"Sampled {self.reference_points} points from mesh for registration")
                    else:
                        reference_for_reg = reference
                else:
                    reference_for_reg = reference
            except Exception as e:
                print(f"Could not load reference as mesh: {str(e)}")
                reference_for_reg = reference
        else:
            reference = o3d.io.read_point_cloud(reference_file)
            reference_for_reg = reference
        
        reference_center = np.mean(np.asarray(reference_for_reg.points), axis=0)
        print(f"Reference has {len(reference_for_reg.points)} points with center at {reference_center}")
        
        # Load scene if provided
        if scene_file:
            print(f"Loading scene point cloud: {scene_file}")
            scene = o3d.io.read_point_cloud(scene_file)
            print(f"Scene point cloud has {len(scene.points)} points")
        else:
            scene = None
        
        return source, reference, reference_for_reg, scene, source_center, reference_center
    
    def downsample_point_cloud(self, pcd, target_points, name="point cloud"):
        """Downsample point cloud to target number of points"""
        original_points = len(pcd.points)
        print(f"Adaptively downsampling {name} from {original_points} points to ~{target_points} points")
        
        if original_points <= target_points:
            print(f"No downsampling needed for {name}")
            return copy.deepcopy(pcd)
        
        # Calculate voxel size to achieve target point count (approximation)
        reduction_factor = original_points / target_points
        voxel_size = 0.01 * (reduction_factor ** (1/3))
        
        # Try voxel downsampling first
        print(f"Trying voxel downsampling with size {voxel_size:.6f}...")
        downsampled = pcd.voxel_down_sample(voxel_size)
        current_points = len(downsampled.points)

        if current_points < 10:
            downsampled = copy.deepcopy(pcd)
            current_points = len(downsampled.points)
        
        # If we're still over the limit, use random sampling as fallback
        if current_points > target_points:
            print(f"Voxel downsampling not sufficient, using random sampling...")
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
        
        print(f"Downsampled {name} to {len(downsampled.points)} points")
        return downsampled
    
    def preprocess_for_registration(self, pcd, name="point cloud", save_visualizations=True):
        """Prepare point cloud for registration"""
        print(f"Preprocessing {name} for registration...")
        
        # Make a copy to avoid modifying the original
        processed = copy.deepcopy(pcd)
        
        # Store original center
        original_center = np.mean(np.asarray(processed.points), axis=0)
        
        if save_visualizations:
            # Visualize the input point cloud with its center
            vis_input = copy.deepcopy(processed)
            input_frame = self._create_coordinate_frame(original_center, size=0.05)
            vis_input.points = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(vis_input.points), np.asarray(input_frame.points))))
            if len(vis_input.colors) > 0:
                # Create default coloring if needed
                if len(vis_input.colors) == 0:
                    vis_input.paint_uniform_color([0.7, 0.7, 0.7])
                vis_input.colors = o3d.utility.Vector3dVector(
                    np.vstack((np.asarray(vis_input.colors), np.asarray(input_frame.colors))))
            else:
                vis_input.paint_uniform_color([0.7, 0.7, 0.7])
                # Add frame colors
                colors = np.ones((len(vis_input.points) - 4, 3)) * 0.7  # Default gray for points
                colors = np.vstack((colors, np.asarray(input_frame.colors)))
                vis_input.colors = o3d.utility.Vector3dVector(colors)
            
            vis_filename = os.path.join(self.output_dir, f'preprocess_input_{name}.pcd')
            o3d.io.write_point_cloud(vis_filename, vis_input, write_ascii=False, compressed=True)
        
        # Center the point cloud
        processed.points = o3d.utility.Vector3dVector(
            np.asarray(processed.points) - original_center)
        
        if save_visualizations:
            # Save the centered point cloud
            centered = copy.deepcopy(processed)
            centered_frame = self._create_coordinate_frame(np.zeros(3), size=0.05)
            centered.points = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(centered.points), np.asarray(centered_frame.points))))
            if len(centered.colors) > 0:
                centered.colors = o3d.utility.Vector3dVector(
                    np.vstack((np.asarray(centered.colors), np.asarray(centered_frame.colors))))
            else:
                centered.paint_uniform_color([0.7, 0.7, 0.7])
                # Add frame colors
                colors = np.ones((len(centered.points) - 4, 3)) * 0.7  # Default gray for points
                colors = np.vstack((colors, np.asarray(centered_frame.colors)))
                centered.colors = o3d.utility.Vector3dVector(colors)
            
            vis_filename = os.path.join(self.output_dir, f'preprocess_centered_{name}.pcd')
            o3d.io.write_point_cloud(vis_filename, centered, write_ascii=False, compressed=True)
        
        # Scale normalization
        points = np.asarray(processed.points)
        scale = np.max([np.linalg.norm(points, axis=1).max(), 1e-8])
        processed.points = o3d.utility.Vector3dVector(points / scale)
        
        if save_visualizations:
            # Save the scaled point cloud
            scaled = copy.deepcopy(processed)
            scaled_frame = self._create_coordinate_frame(np.zeros(3), size=0.05)
            scaled.points = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(scaled.points), np.asarray(scaled_frame.points))))
            if len(scaled.colors) > 0:
                scaled.colors = o3d.utility.Vector3dVector(
                    np.vstack((np.asarray(scaled.colors), np.asarray(scaled_frame.colors))))
            else:
                scaled.paint_uniform_color([0.7, 0.7, 0.7])
                # Add frame colors
                colors = np.ones((len(scaled.points) - 4, 3)) * 0.7  # Default gray for points
                colors = np.vstack((colors, np.asarray(scaled_frame.colors)))
                scaled.colors = o3d.utility.Vector3dVector(colors)
            
            vis_filename = os.path.join(self.output_dir, f'preprocess_scaled_{name}.pcd')
            o3d.io.write_point_cloud(vis_filename, scaled, write_ascii=False, compressed=True)
        
        # Voxel downsampling
        print(f"Downsampling with voxel size: {self.voxel_size}")
        downsampled = processed.voxel_down_sample(self.voxel_size)
        
        # Statistical outlier removal for noise handling
        print(f"Removing outliers with {self.nb_neighbors} neighbors and {self.std_ratio} std ratio")
        cleaned, ind = downsampled.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio)
        
        # Estimate normals for feature computation
        print("Estimating normals...")
        cleaned.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=self.voxel_size * 2, max_nn=30))
        
        # Compute FPFH features
        print("Computing FPFH features...")
        features = o3d.pipelines.registration.compute_fpfh_feature(
            cleaned, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100))
        
        if save_visualizations:
            # Save the final processed point cloud
            final = copy.deepcopy(cleaned)
            final_frame = self._create_coordinate_frame(np.zeros(3), size=0.05)
            final.points = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(final.points), np.asarray(final_frame.points))))
            if len(final.colors) > 0:
                final.colors = o3d.utility.Vector3dVector(
                    np.vstack((np.asarray(final.colors), np.asarray(final_frame.colors))))
            else:
                final.paint_uniform_color([0.7, 0.7, 0.7])
                # Add frame colors
                colors = np.ones((len(final.points) - 4, 3)) * 0.7  # Default gray for points
                colors = np.vstack((colors, np.asarray(final_frame.colors)))
                final.colors = o3d.utility.Vector3dVector(colors)
            
            vis_filename = os.path.join(self.output_dir, f'preprocess_final_{name}.pcd')
            o3d.io.write_point_cloud(vis_filename, final, write_ascii=False, compressed=True)
            
            # Save preprocessing details
            details_file = os.path.join(self.output_dir, f'preprocess_details_{name}.txt')
            with open(details_file, 'w') as f:
                f.write(f"Preprocessing Details for {name}\n")
                f.write("================================\n\n")
                f.write(f"Original center: {original_center}\n")
                f.write(f"Normalization scale: {scale}\n")
                f.write(f"Original point count: {len(pcd.points)}\n")
                f.write(f"After downsampling: {len(downsampled.points)}\n")
                f.write(f"After outlier removal: {len(cleaned.points)}\n")
            
            print(f"Preprocessing details saved to {details_file}")
        
        return cleaned, features, scale, original_center
    
    def register_point_clouds(self, source, target, save_visualizations=True):
        """Perform registration between source and target point clouds"""
        print("Starting registration process...")
        
        if save_visualizations:
            # Run initial debugging on the input point clouds
            self.debug_registration_process(source, target)
            
            # Save visualization before registration
            self.visualize_registration(source, target, step="before")
        
        # Preprocess point clouds
        source_processed, source_feat, source_scale, source_center = self.preprocess_for_registration(source, "source", save_visualizations)
        target_processed, target_feat, target_scale, target_center = self.preprocess_for_registration(target, "target", save_visualizations)
        
        if save_visualizations:
            # Also visualize the preprocessed (normalized) point clouds
            self.visualize_registration(source_processed, target_processed, step="preprocessed")
        
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
        
        # Store transformations for our sequence visualization
        if save_visualizations:
            transforms = [result.transformation]
            transform_names = ["After RANSAC"]
        
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
        for i in range(self.icp_max_iter):
            print(f"ICP iteration {i+1}/{self.icp_max_iter}")
            
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
            
            if save_visualizations:
                # Save major steps in the transform sequence
                if i == 0 or i == 4 or i == 9 or i == 19 or i == self.icp_max_iter-1:
                    transforms.append(current_transform)
                    transform_names.append(f"After ICP iteration {i+1}")
            
            # Check for convergence
            if i > 0 and abs(self.fitness_history[-1] - self.fitness_history[-2]) < 1e-10:
                if save_visualizations and i not in [0, 4, 9, 19]:
                    transforms.append(current_transform)
                    transform_names.append(f"Final (ICP iteration {i+1})")
                print(f"ICP converged after {i+1} iterations")
                break
        
        final_transform_processed = current_transform
        
        # Scale back the transformation
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = scale_matrix[1, 1] = scale_matrix[2, 2] = source_scale / target_scale
        final_transform = np.matmul(final_transform_processed, scale_matrix)
        
        if save_visualizations:
            # Add the final scaled-back transform
            transforms.append(final_transform)
            transform_names.append("Final (after scaling)")
            
            # Create sequence visualization
            self.visualize_transform_sequence(source_processed, target_processed, 
                                            transforms[:-1], transform_names[:-1])
            
            # Also create a sequence visualization on original (unprocessed) point clouds
            self.visualize_transform_sequence(source, target, [final_transform], ["Final Transform"])
            
            # Debug the final transformation
            self.debug_registration_process(source, target, final_transform)
            
            # Save visualization after registration
            self.visualize_registration(source, target, final_transform, step="after")
        
        return final_transform, source_center, target_center
    
    def visualize_transform_sequence(self, source, target, transforms, names):
        """
        Visualize a sequence of transformations applied to the source point cloud.
        
        Args:
            source: Original source point cloud
            target: Target point cloud
            transforms: List of transformation matrices
            names: List of transformation names/descriptions
        """
        assert len(transforms) == len(names), "Number of transforms must match number of names"
        
        # Create a multi-stage visualization
        all_points = []
        all_colors = []
        
        # Add the target point cloud (reference) in green
        target_copy = copy.deepcopy(target)
        target_copy.paint_uniform_color([0, 1, 0])
        all_points.append(np.asarray(target_copy.points))
        all_colors.append(np.asarray(target_copy.colors))
        
        # Create a color gradient for source transforms from blue to red
        colors = []
        for i in range(len(transforms)):
            # Gradient from blue to red
            r = i / (len(transforms) - 1) if len(transforms) > 1 else 1.0
            b = 1.0 - r
            colors.append([r, 0, b])
        
        # Apply each transformation to the source
        for i, (transform, name) in enumerate(zip(transforms, names)):
            source_copy = copy.deepcopy(source)
            source_copy.transform(transform)
            source_copy.paint_uniform_color(colors[i])
            
            all_points.append(np.asarray(source_copy.points))
            all_colors.append(np.asarray(source_copy.colors))
        
        # Combine all point clouds
        combined = o3d.geometry.PointCloud()
        combined.points = o3d.utility.Vector3dVector(np.vstack(all_points))
        combined.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
        
        # Save the visualization
        vis_filename = os.path.join(self.output_dir, 'transform_sequence.pcd')
        o3d.io.write_point_cloud(vis_filename, combined, write_ascii=False, compressed=True)
        
        # Create a legend text file
        legend_file = os.path.join(self.output_dir, 'transform_sequence_legend.txt')
        with open(legend_file, 'w') as f:
            f.write("Transform Sequence Visualization Legend\n")
            f.write("=====================================\n\n")
            f.write("Green: Target/Reference point cloud\n")
            
            for i, name in enumerate(names):
                r = i / (len(names) - 1) if len(names) > 1 else 1.0
                b = 1.0 - r
                f.write(f"RGB({int(r*255)}, 0, {int(b*255)}): {name}\n")
        
        print(f"Transform sequence visualization saved to {vis_filename}")
        print(f"Legend saved to {legend_file}")
    

    def visualize_transformation(self, transform, source_center, target_center):
        """
        Visualize the transformation by creating coordinate frames
        and showing how the transformation maps source to target space.
        """
        # Create a text file with transformation details
        transform_viz_file = os.path.join(self.output_dir, 'transformation_visualization.txt')
        
        # Extract rotation and translation
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        euler_angles = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True)
        
        with open(transform_viz_file, 'w') as f:
            f.write("Transformation Visualization\n")
            f.write("===========================\n\n")
            f.write(f"Source center: {source_center[0]:.4f}, {source_center[1]:.4f}, {source_center[2]:.4f}\n")
            f.write(f"Target center: {target_center[0]:.4f}, {target_center[1]:.4f}, {target_center[2]:.4f}\n\n")
            f.write("Transformation Matrix:\n")
            f.write(np.array_str(transform, precision=6))
            f.write("\n\n")
            f.write("Decomposed Transformation:\n")
            f.write(f"Translation (x, y, z): {translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}\n")
            f.write(f"Rotation (Euler angles in degrees, xyz): {euler_angles[0]:.6f}, {euler_angles[1]:.6f}, {euler_angles[2]:.6f}\n")
        
        print(f"Transformation visualization saved to {transform_viz_file}")
        
        # Create coordinate frames as point clouds and save them
        # This will create source, target, and transformed source coordinate frames
        self.visualize_coordinate_frames(transform, source_center, target_center)

    def visualize_coordinate_frames(self, transform, source_center, target_center):
        """
        Create a visualization of coordinate frames before and after transformation.
        """
        # Create coordinate frame point clouds
        def create_coordinate_frame(center, size=0.1, color_type="rgb"):
            points = []
            colors = []
            
            # Origin
            points.append(center)
            colors.append([0.5, 0.5, 0.5])  # Gray for origin
            
            # X axis (red)
            points.append(center + np.array([size, 0, 0]))
            colors.append([1, 0, 0])
            
            # Y axis (green)
            points.append(center + np.array([0, size, 0]))
            colors.append([0, 1, 0])
            
            # Z axis (blue)
            points.append(center + np.array([0, 0, size]))
            colors.append([0, 0, 1])
            
            frame = o3d.geometry.PointCloud()
            frame.points = o3d.utility.Vector3dVector(np.array(points))
            frame.colors = o3d.utility.Vector3dVector(np.array(colors))
            return frame
        
        # Create source frame
        source_frame = create_coordinate_frame(source_center, size=0.05)
        
        # Create target frame
        target_frame = create_coordinate_frame(target_center, size=0.05)
        
        # Create transformed source frame
        # First, we need to create a transformation that includes the centers
        T_src = np.eye(4)
        T_src[:3, 3] = source_center
        
        T_transform = np.eye(4)
        T_transform[:3, :3] = transform[:3, :3]
        T_transform[:3, 3] = transform[:3, 3]
        
        T_combined = np.matmul(T_transform, T_src)
        transformed_center = T_combined[:3, 3]
        
        # Create the transformed frame
        transformed_frame = create_coordinate_frame(transformed_center, size=0.05)
        
        # Combine all frames
        all_frames = o3d.geometry.PointCloud()
        all_frames.points = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(source_frame.points), 
                    np.asarray(target_frame.points),
                    np.asarray(transformed_frame.points))))
        all_frames.colors = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(source_frame.colors),
                    np.asarray(target_frame.colors),
                    np.asarray(transformed_frame.colors))))
        
        # Save the visualization
        frames_filename = os.path.join(self.output_dir, 'coordinate_frames.pcd')
        o3d.io.write_point_cloud(frames_filename, all_frames, write_ascii=False, compressed=True)
        print(f"Coordinate frames visualization saved to {frames_filename}")
                
    

    def visualize_registration(self, source, reference, transform=None, step="before"):
        """
        Visualize the registration between source and reference point clouds.
        
        Args:
            source: Source point cloud (ROI)
            reference: Reference point cloud
            transform: Optional transformation matrix to apply to source
            step: String indicating registration step ("before" or "after")
        """
        print(f"Creating visualization for {step} registration...")
        
        # Create copies to avoid modifying originals
        source_vis = copy.deepcopy(source)
        reference_vis = copy.deepcopy(reference)
        
        # For visualization clarity, ensure both point clouds are centered
        # This helps when the clouds are far apart in their original spaces
        if step == "before":
            # For "before" visualization, center both clouds at the origin
            # to see their relative shapes before alignment
            source_center = np.mean(np.asarray(source_vis.points), axis=0)
            reference_center = np.mean(np.asarray(reference_vis.points), axis=0)
            
            # Center both clouds for better visualization
            source_vis.points = o3d.utility.Vector3dVector(
                np.asarray(source_vis.points) - source_center)
            reference_vis.points = o3d.utility.Vector3dVector(
                np.asarray(reference_vis.points) - reference_center)
            
            # Log the centering operations for debugging
            print(f"Centering source from {source_center} and reference from {reference_center} for visualization")
        
        # Apply transformation if provided (for "after" step)
        if transform is not None and step == "after":
            # For "after" visualization, apply the transformation
            # but don't pre-center the clouds as we want to see the actual transformation result
            source_vis.transform(transform)
            
            # Log transformation details
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            euler_angles = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True)
            print(f"Applied transformation to source with:")
            print(f"  Translation: {translation}")
            print(f"  Rotation (Euler angles in degrees): {euler_angles}")
        
        # Color the point clouds distinctly
        source_vis.paint_uniform_color([1, 0, 0])       # Red for source/ROI
        reference_vis.paint_uniform_color([0, 1, 0])    # Green for reference
        
        # Add coordinate frames to visualize orientation
        frame_size = max(
            np.ptp(np.asarray(source_vis.points), axis=0).max(),
            np.ptp(np.asarray(reference_vis.points), axis=0).max()
        ) * 0.1  # 10% of the largest dimension
        
        source_frame = self._create_coordinate_frame(
            np.mean(np.asarray(source_vis.points), axis=0), size=frame_size)
        reference_frame = self._create_coordinate_frame(
            np.mean(np.asarray(reference_vis.points), axis=0), size=frame_size)
        
        # Combine the point clouds and frames
        combined = o3d.geometry.PointCloud()
        combined.points = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(source_vis.points), 
                    np.asarray(reference_vis.points),
                    np.asarray(source_frame.points),
                    np.asarray(reference_frame.points))))
        combined.colors = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(source_vis.colors), 
                    np.asarray(reference_vis.colors),
                    np.asarray(source_frame.colors),
                    np.asarray(reference_frame.colors))))
        
        # Save the visualization
        vis_filename = os.path.join(self.output_dir, f'registration_vis_{step}.pcd')
        o3d.io.write_point_cloud(vis_filename, combined, write_ascii=False, compressed=True)
        print(f"Registration visualization saved to {vis_filename}")
        
        # Save individual clouds for detailed inspection
        src_filename = os.path.join(self.output_dir, f'source_{step}.pcd')
        ref_filename = os.path.join(self.output_dir, f'reference_{step}.pcd')
        o3d.io.write_point_cloud(src_filename, source_vis, write_ascii=False, compressed=True)
        o3d.io.write_point_cloud(ref_filename, reference_vis, write_ascii=False, compressed=True)
        
        return combined

    def _create_coordinate_frame(self, center, size=0.1):
        """
        Create a coordinate frame point cloud at the specified center.
        """
        points = []
        colors = []
        
        # Origin
        points.append(center)
        colors.append([0.5, 0.5, 0.5])  # Gray for origin
        
        # X axis (red)
        points.append(center + np.array([size, 0, 0]))
        colors.append([1, 0, 0])
        
        # Y axis (green)
        points.append(center + np.array([0, size, 0]))
        colors.append([0, 1, 0])
        
        # Z axis (blue)
        points.append(center + np.array([0, 0, size]))
        colors.append([0, 0, 1])
        
        frame = o3d.geometry.PointCloud()
        frame.points = o3d.utility.Vector3dVector(np.array(points))
        frame.colors = o3d.utility.Vector3dVector(np.array(colors))
        return frame
    
    def debug_registration_process(self, source, target, transform=None):
        """
        Detailed debugging of the registration process to identify issues.
        """
        debug_file = os.path.join(self.output_dir, 'registration_debug.txt')
        
        with open(debug_file, 'w') as f:
            # Basic statistics
            source_points = np.asarray(source.points)
            target_points = np.asarray(target.points)
            
            source_center = np.mean(source_points, axis=0)
            target_center = np.mean(target_points, axis=0)
            
            source_min = np.min(source_points, axis=0)
            source_max = np.max(source_points, axis=0)
            target_min = np.min(target_points, axis=0)
            target_max = np.max(target_points, axis=0)
            
            f.write("Registration Debug Information\n")
            f.write("============================\n\n")
            
            f.write(f"Source Point Cloud Stats:\n")
            f.write(f"  Number of points: {len(source_points)}\n")
            f.write(f"  Center: {source_center}\n")
            f.write(f"  Min bounds: {source_min}\n")
            f.write(f"  Max bounds: {source_max}\n")
            f.write(f"  Size (range): {source_max - source_min}\n\n")
            
            f.write(f"Target Point Cloud Stats:\n")
            f.write(f"  Number of points: {len(target_points)}\n")
            f.write(f"  Center: {target_center}\n")
            f.write(f"  Min bounds: {target_min}\n")
            f.write(f"  Max bounds: {target_max}\n")
            f.write(f"  Size (range): {target_max - target_min}\n\n")
            
            # Analyze scale differences
            source_scale = np.max([np.linalg.norm(source_points, axis=1).max(), 1e-8])
            target_scale = np.max([np.linalg.norm(target_points, axis=1).max(), 1e-8])
            
            f.write(f"Scale Analysis:\n")
            f.write(f"  Source scale: {source_scale}\n")
            f.write(f"  Target scale: {target_scale}\n")
            f.write(f"  Scale ratio (source/target): {source_scale/target_scale}\n\n")
            
            # Analyze the transformation if provided
            if transform is not None:
                f.write(f"Transformation Analysis:\n")
                f.write(f"  Full matrix:\n{transform}\n\n")
                
                # Decompose transformation
                rotation = transform[:3, :3]
                translation = transform[:3, 3]
                
                # Check for reflection in the rotation matrix
                det = np.linalg.det(rotation)
                f.write(f"  Rotation determinant: {det}\n")
                if abs(det - 1.0) > 1e-6:
                    f.write(f"  WARNING: Rotation includes scaling or reflection!\n")
                
                # Check if rotation is orthogonal
                orthogonality_error = np.linalg.norm(np.dot(rotation, rotation.T) - np.eye(3))
                f.write(f"  Rotation orthogonality error: {orthogonality_error}\n")
                if orthogonality_error > 1e-6:
                    f.write(f"  WARNING: Rotation matrix is not orthogonal!\n")
                
                # Convert to Euler angles
                try:
                    euler_angles = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True)
                    f.write(f"  Euler angles (xyz, degrees): {euler_angles}\n")
                except Exception as e:
                    f.write(f"  ERROR computing Euler angles: {str(e)}\n")
                
                f.write(f"  Translation: {translation}\n")
                
                # Apply transform to source center and verify it approaches target center
                transformed_center = np.dot(rotation, source_center) + translation
                distance_between_centers = np.linalg.norm(transformed_center - target_center)
                f.write(f"  Source center after transform: {transformed_center}\n")
                f.write(f"  Distance between transformed source center and target center: {distance_between_centers}\n")
                
                if distance_between_centers > 0.5 * np.linalg.norm(target_max - target_min):
                    f.write(f"  WARNING: Large distance between centers after transformation!\n")
            
            f.write("\nRecommendations:\n")
            # Add recommendations based on analysis
            if transform is not None and (orthogonality_error > 1e-6 or abs(det - 1.0) > 1e-6):
                f.write("- Check the preprocessing normalization steps, especially centering and scaling\n")
                f.write("- The current transformation includes non-rigid components which may distort the model\n")
            
            if transform is not None and distance_between_centers > 0.5 * np.linalg.norm(target_max - target_min):
                f.write("- The centers are far apart after transformation, suggesting poor alignment\n")
                f.write("- Consider using a different initial alignment strategy\n")
                f.write("- Try increasing the number of RANSAC iterations\n")
            
            if abs(source_scale/target_scale - 1.0) > 0.5:
                f.write("- Large scale difference between source and target\n")
                f.write("- Consider explicit scaling normalization before registration\n")
        
        print(f"Detailed registration debug information saved to {debug_file}")
        
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
        
        print("Adjusted transformation matrix:")
        print(np.array_str(adjusted_transform, precision=4))
        
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

        print("scene points: ", len(scene.points))
        
        print(f"Target file size: {self.max_file_size_mb} MB")
        print(f"Allocating {len(reference_transformed.points)} points to reference")
        print(f"Allocating ~{scene_points} points to scene")
        
        # Downsample scene if needed
        if len(scene.points) > scene_points:
            scene_downsampled = self.downsample_point_cloud(scene, scene_points, "scene")
        else:
            scene_downsampled = copy.deepcopy(scene)

        print("downsampled scene points: ", len(scene_downsampled.points))
        
        # Combine point clouds
        combined = o3d.geometry.PointCloud()
        combined.points = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(reference_transformed.points), np.asarray(scene_downsampled.points))))
        
        # Combine colors
        if len(reference_transformed.colors) > 0 and len(scene_downsampled.colors) > 0:
            combined.colors = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(reference_transformed.colors), np.asarray(scene_downsampled.colors))))
        
        # Save result only if output_file is provided
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
        """
        
        # Load or use provided point clouds
        if source_pcl is not None and reference_pcl is not None:
            # Use provided point cloud objects
            print("Using provided point cloud objects...")
            source = copy.deepcopy(source_pcl)
            reference = copy.deepcopy(reference_pcl)
            reference_for_reg = copy.deepcopy(reference_pcl)
            
            # Calculate centers
            source_center = np.mean(np.asarray(source.points), axis=0)
            reference_center = np.mean(np.asarray(reference_for_reg.points), axis=0)
            
            print(f"Source point cloud has {len(source.points)} points with center at {source_center}")
            print(f"Reference has {len(reference_for_reg.points)} points with center at {reference_center}")
            
        else:
            # Load from files
            if source_file is None or reference_file is None:
                raise ValueError("Either provide point cloud objects (source_pcl, reference_pcl) or file paths (source_file, reference_file)")
            
            source, reference, reference_for_reg, _, source_center, reference_center = self.load_point_clouds(
                source_file, reference_file)
        
        # Register
        transform, source_center, reference_center = self.register_point_clouds(source, reference_for_reg)
        
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
        
        print("\nRegistration completed successfully!")
        print(f"Transformation matrix saved to {transform_file}")
        print(f"Transformation R,t format saved to {rt_transform_file}")
        
        # Convert to R,t format for display and return
        R_str, t_str = self.transform_to_rt_format(transform)
        
        print("\nTransformation in R,t format:")
        print(f"R: {R_str}")
        print(f"t: {t_str}")
        
        # Return in R,t format instead of 4x4 matrix
        return R_str, t_str, source_center, reference_center
    
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
        
        # Combine and reduce size
        combined = self.combine_and_reduce_size(reference_transformed, scene, output_file)
        
        print("\nPlacement completed successfully!")
        print(f"Combined result saved to {output_file}")
        
        return combined
    
    def visualize_scene_placement(self, reference, scene, step="before"):
        """
        Visualize the placement of the reference in the scene.
        
        Args:
            reference: Reference point cloud (original or transformed)
            scene: Scene point cloud
            step: String indicating placement step ("before" or "after")
        """
        print(f"Creating visualization for {step} placement...")
        
        # Create copies to avoid modifying originals
        reference_vis = copy.deepcopy(reference)
        scene_vis = copy.deepcopy(scene)
        
        # Color the point clouds distinctly
        if step == "before":
            reference_vis.paint_uniform_color([0, 1, 0])  # Green for reference before placement
        else:
            reference_vis.paint_uniform_color([1, 0.3, 0])  # Orange for reference after placement
        
        scene_vis.paint_uniform_color([0, 0, 1])  # Blue for scene
        
        # Combine the point clouds
        combined = o3d.geometry.PointCloud()
        combined.points = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(reference_vis.points), np.asarray(scene_vis.points))))
        combined.colors = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(reference_vis.colors), np.asarray(scene_vis.colors))))
        
        # Save the visualization
        vis_filename = os.path.join(self.output_dir, f'placement_vis_{step}.pcd')
        o3d.io.write_point_cloud(vis_filename, combined, write_ascii=False, compressed=True)
        print(f"Placement visualization saved to {vis_filename}")
        
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
        """
        
        # Use provided point cloud objects and load reference from file
        print("Using provided source and scene point cloud objects...")
        print(f"Loading reference model: {reference_file_path}")
        
        # Load reference from file
        if reference_file_path.endswith('.ply'):
            # Load reference as point cloud
            reference = o3d.io.read_point_cloud(reference_file_path)
            
            # Also try loading as mesh to check if it contains triangle information
            try:
                reference_mesh = o3d.io.read_triangle_mesh(reference_file_path)
                if len(reference_mesh.triangles) > 0:
                    print(f"Reference loaded as mesh with {len(reference_mesh.triangles)} triangles")
                    # If it's a proper mesh, sample points for registration
                    if len(reference_mesh.vertices) > self.reference_points:
                        reference_for_reg = reference_mesh.sample_points_uniformly(self.reference_points)
                        print(f"Sampled {self.reference_points} points from mesh for registration")
                    else:
                        reference_for_reg = reference
                else:
                    reference_for_reg = reference
            except Exception as e:
                print(f"Could not load reference as mesh: {str(e)}")
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
        
        print(f"Source point cloud has {len(source.points)} points with center at {source_center}")
        print(f"Reference has {len(reference_for_reg.points)} points with center at {reference_center}")
        print(f"Scene point cloud has {len(scene.points)} points")
        
        # Register with conditional visualization
        transform, source_center, reference_center = self.register_point_clouds(
            source, reference_for_reg, save_visualizations=visualize_and_save_results)
        
        if visualize_and_save_results:
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
            
            print("\nRegistration completed. Proceeding with placement...")
            
            # Visualize reference and scene before placement
            self.visualize_scene_placement(reference, scene, step="before")
        else:
            print("\nRegistration completed (visualization skipped). Proceeding with placement...")

        if visualize_and_save_results:        
            # Place reference in scene
            reference_transformed = self.place_reference_in_scene(
                reference, scene, transform, source_center, reference_center)
        
        if visualize_and_save_results:
            # Visualize reference and scene after placement
            self.visualize_scene_placement(reference_transformed, scene, step="after")
        
            # Combine and reduce size (conditionally save to file)
            output_file_param = output_file if visualize_and_save_results else None
            combined = self.combine_and_reduce_size(reference_transformed, scene, output_file_param)
        
        
        # Convert to R,t format for return
        R_str, t_str = self.transform_to_rt_format(transform)
        
        if visualize_and_save_results:
            print("\nFull pipeline completed successfully!")
            if output_file:
                print(f"Combined result saved to {output_file}")
        else:
            print("\nFull pipeline completed successfully!")
            print("Results computed but files not saved (visualization disabled)")
        
        print(f"\nTransformation in R,t format:")
        print(f"R: {R_str}")
        print(f"t: {t_str}")
        
        # Always return the transformation in R,t format
        return R_str, t_str

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration and Placement with Size Reduction')
    
    # Required arguments
    parser.add_argument('--source', type=str, required=True, help='Path to source (ROI) PCD file')
    parser.add_argument('--reference', type=str, required=True, help='Path to reference PLY/PCD file')
    
    # Optional arguments
    parser.add_argument('--scene', type=str, help='Path to scene PCD file')
    parser.add_argument('--transform', type=str, help='Path to pre-computed transformation file (R,t format or 4x4 matrix)')
    parser.add_argument('--output', type=str, default='combined_result.pcd', help='Output file path for combined result')
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
            R_str, t_str = processor.run_full_pipeline(
                args.source, args.reference, args.scene, args.output, 
                visualize_and_save_results=not args.no_save)
            
            print(f"\nFinal transformation in R,t format:")
            print(f"R: {R_str}")
            print(f"t: {t_str}")
        else:
            # Registration only
            R_str, t_str, source_center, reference_center = processor.run_registration_only(args.source, args.reference)
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()