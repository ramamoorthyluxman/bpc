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

class PointCloudRegistration:
    def __init__(self, use_visualization=False):
        # Parameters for preprocessing
        self.voxel_size = 0.01  # Downsampling voxel size
        self.nb_neighbors = 20  # Neighbors for normal estimation
        self.std_ratio = 2.0    # Standard deviation ratio for outlier removal
        
        # Parameters for registration
        self.distance_threshold = 0.05  # For RANSAC
        self.ransac_n = 3       # Minimum points for RANSAC
        self.ransac_iter = 100000  # RANSAC iterations
        self.icp_threshold = 0.005  # ICP convergence threshold
        self.icp_max_iter = 100    # Maximum ICP iterations
        
        # Transformation history
        self.transformation_history = []
        self.fitness_history = []
        self.rmse_history = []
        
        # Visualization settings
        self.use_visualization = use_visualization
        self.vis = None
        self.source_temp = None  # For visualization updates
        
        # Output directory
        self.output_dir = 'registration_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_point_clouds(self, source_file, target_file):
        """Load source (PCD) and target (PLY) point clouds"""
        print(f"Loading source point cloud: {source_file}")
        print(f"Loading target point cloud: {target_file}")
        
        # Determine file type and load accordingly
        if source_file.endswith('.pcd'):
            source = o3d.io.read_point_cloud(source_file)
        else:
            raise ValueError("Source file must be a PCD file")
            
        if target_file.endswith('.ply'):
            target = o3d.io.read_point_cloud(target_file)
        else:
            raise ValueError("Target file must be a PLY file")
            
        print(f"Source point cloud has {len(source.points)} points")
        print(f"Target point cloud has {len(target.points)} points")
        
        return source, target
    
    def preprocess_point_cloud(self, pcd, name="point cloud"):
        """Preprocess point cloud: downsampling, normal estimation, outlier removal"""
        print(f"Preprocessing {name}...")
        
        # Make a copy to avoid modifying the original
        processed = copy.deepcopy(pcd)
        
        # Check if point cloud is empty
        if len(processed.points) == 0:
            raise ValueError(f"Empty {name}")
        
        # Center the point cloud
        processed.points = o3d.utility.Vector3dVector(
            np.asarray(processed.points) - np.mean(np.asarray(processed.points), axis=0))
        
        # Scale normalization
        points = np.asarray(processed.points)
        scale = np.max([np.linalg.norm(points, axis=1).max(), 1e-8])
        processed.points = o3d.utility.Vector3dVector(points / scale)
        
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
        
        return cleaned, features, scale
    
    def execute_global_registration(self, source, target, source_feat, target_feat):
        """Perform global registration using RANSAC"""
        print("RANSAC Global Registration...")
        
        start = time.time()
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_feat, target_feat,
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
        
        # Save initial transformation to history
        self.transformation_history.append(result.transformation)
        self.fitness_history.append(result.fitness)
        self.rmse_history.append(result.inlier_rmse)
        
        return result.transformation
        
    def refine_registration(self, source, target, initial_transform):
        """Refine registration using ICP"""
        print("Refining registration with ICP...")
        
        current_transform = initial_transform
        source_transformed = copy.deepcopy(source)
        source_transformed.transform(current_transform)
        
        # Iterative ICP
        for i in range(self.icp_max_iter):
            print(f"ICP iteration {i+1}/{self.icp_max_iter}")
            
            # Run one ICP iteration
            result = o3d.pipelines.registration.registration_icp(
                source_transformed, target, 
                self.icp_threshold, np.eye(4),  # Use identity as initial to avoid compounding
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=1))
            
            # Update the current transformation
            iter_transform = result.transformation
            current_transform = np.matmul(iter_transform, current_transform)
            
            # Transform the source for next iteration
            source_transformed = copy.deepcopy(source)
            source_transformed.transform(current_transform)
            
            # Save transformation to history
            self.transformation_history.append(current_transform)
            self.fitness_history.append(result.fitness)
            self.rmse_history.append(result.inlier_rmse)
            
            # Check for convergence
            if i > 0 and abs(self.fitness_history[-1] - self.fitness_history[-2]) < 1e-6:
                print(f"ICP converged after {i+1} iterations")
                break
        
        return current_transform
    
    def plot_registration_metrics(self):
        """Plot registration metrics over iterations"""
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
        plt.savefig(f'{self.output_dir}/registration_metrics.png')
        
        print(f"Registration metrics plot saved to {self.output_dir}/registration_metrics.png")
    
    def visualize_transformation_evolution(self):
        """Visualize the evolution of transformations"""
        if len(self.transformation_history) < 2:
            print("Not enough transformations to visualize evolution")
            return
        
        # Create a visualization of how the transformations evolved
        plt.figure(figsize=(12, 6))
        
        # Extract translations over iterations
        translations = np.array([t[:3, 3] for t in self.transformation_history])
        iterations = range(len(translations))
        
        # Plot X, Y, Z translations
        plt.subplot(1, 2, 1)
        plt.plot(iterations, translations[:, 0], 'r-', label='X')
        plt.plot(iterations, translations[:, 1], 'g-', label='Y')
        plt.plot(iterations, translations[:, 2], 'b-', label='Z')
        plt.xlabel('Iteration')
        plt.ylabel('Translation')
        plt.title('Translation Evolution')
        plt.legend()
        plt.grid(True)
        
        # Extract rotations over iterations (convert to Euler angles)
        rotations = np.array([Rotation.from_matrix(t[:3, :3]).as_euler('xyz', degrees=True) 
                              for t in self.transformation_history])
        
        # Plot rotation angles
        plt.subplot(1, 2, 2)
        plt.plot(iterations, rotations[:, 0], 'r-', label='Roll')
        plt.plot(iterations, rotations[:, 1], 'g-', label='Pitch')
        plt.plot(iterations, rotations[:, 2], 'b-', label='Yaw')
        plt.xlabel('Iteration')
        plt.ylabel('Rotation (degrees)')
        plt.title('Rotation Evolution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/transformation_evolution.png')
        print(f"Transformation evolution saved to {self.output_dir}/transformation_evolution.png")
    
    def create_off_screen_renderer(self, width=1280, height=720):
        """Creates an off-screen renderer for Open3D visualization"""
        render = o3d.visualization.rendering.OffscreenRenderer(width, height)
        
        # Set up light
        render.scene.set_background([0, 0, 0, 1])  # Black background
        render.scene.add_directional_light("main_light", [0, 1, 1], [1, 1, 1], 100000)
        render.scene.add_directional_light("fill_light", [1, -1, -1], [0.5, 0.5, 0.5], 100000)
        
        return render
    
    def render_point_cloud_view(self, renderer, geometries, camera_position, look_at=[0, 0, 0], up=[0, 1, 0], filename=None):
        """Renders a view of the geometries from a specific camera position"""
        
        # Add geometries to the scene
        for i, geom in enumerate(geometries):
            if isinstance(geom, o3d.geometry.PointCloud):
                # Convert point cloud to triangle mesh for better rendering
                # Use ball pivoting or Poisson reconstruction depending on normals
                if len(geom.normals) > 0:
                    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(geom, depth=5)
                    material = o3d.visualization.rendering.MaterialRecord()
                    material.shader = "defaultLit"
                    
                    # Use the same colors as the point cloud
                    if len(geom.colors) > 0:
                        mesh.vertex_colors = geom.colors
                        
                    renderer.scene.add_geometry(f"mesh_{i}", mesh, material)
                else:
                    material = o3d.visualization.rendering.MaterialRecord()
                    material.shader = "defaultLit"
                    material.point_size = 3.0
                    
                    # Set material color if point cloud has colors
                    if len(geom.colors) > 0:
                        first_color = np.asarray(geom.colors)[0]
                        material.base_color = [first_color[0], first_color[1], first_color[2], 1.0]
                    
                    renderer.scene.add_geometry(f"pointcloud_{i}", geom, material)
            else:
                material = o3d.visualization.rendering.MaterialRecord()
                material.shader = "defaultLit"
                renderer.scene.add_geometry(f"geometry_{i}", geom, material)
        
        # Set up camera
        camera = o3d.visualization.rendering.Camera("perspective")
        camera.look_at(look_at, camera_position, up)
        renderer.scene.camera = camera
        
        # Render image
        img = renderer.render_to_image()
        
        # Save if filename is provided
        if filename:
            o3d.io.write_image(filename, img)
            
        # Clear scene for next render
        renderer.scene.clear_geometry()
        
        return img
    
    def create_2d_projection_visualizations(self, source, target, source_transformed):
        """Create 2D projections of point clouds for easy comparison"""
        print("Creating 2D projection visualizations...")
        
        # Convert to numpy arrays
        source_pts = np.asarray(source.points)
        target_pts = np.asarray(target.points)
        transformed_pts = np.asarray(source_transformed.points)
        
        # Sample points if there are too many (for clearer visualization)
        max_points = 5000
        if len(source_pts) > max_points:
            idx = np.random.choice(len(source_pts), max_points, replace=False)
            source_pts = source_pts[idx]
        if len(transformed_pts) > max_points:
            idx = np.random.choice(len(transformed_pts), max_points, replace=False)
            transformed_pts = transformed_pts[idx]
        if len(target_pts) > max_points:
            idx = np.random.choice(len(target_pts), max_points, replace=False)
            target_pts = target_pts[idx]
        
        # Create figure for 3 projections (XY, XZ, YZ), before and after registration
        plt.figure(figsize=(18, 12))
        
        # Define projection planes
        projections = [
            {'indices': [0, 1], 'title': 'XY Projection', 'labels': ['X', 'Y']},
            {'indices': [0, 2], 'title': 'XZ Projection', 'labels': ['X', 'Z']},
            {'indices': [1, 2], 'title': 'YZ Projection', 'labels': ['Y', 'Z']}
        ]
        
        # Plot each projection
        for i, proj in enumerate(projections):
            idx1, idx2 = proj['indices']
            
            # Before registration (source vs target)
            plt.subplot(2, 3, i+1)
            plt.scatter(source_pts[:, idx1], source_pts[:, idx2], c='red', marker='.', s=1, alpha=0.5, label='Source')
            plt.scatter(target_pts[:, idx1], target_pts[:, idx2], c='blue', marker='.', s=1, alpha=0.5, label='Target')
            plt.title(f"{proj['title']} - Before Registration")
            plt.xlabel(proj['labels'][0])
            plt.ylabel(proj['labels'][1])
            plt.grid(True)
            plt.legend()
            
            # After registration (transformed source vs target)
            plt.subplot(2, 3, i+4)
            plt.scatter(transformed_pts[:, idx1], transformed_pts[:, idx2], c='green', marker='.', s=1, alpha=0.5, label='Transformed Source')
            plt.scatter(target_pts[:, idx1], target_pts[:, idx2], c='blue', marker='.', s=1, alpha=0.5, label='Target')
            plt.title(f"{proj['title']} - After Registration")
            plt.xlabel(proj['labels'][0])
            plt.ylabel(proj['labels'][1])
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/2d_projections.png', dpi=300)
        print(f"2D projections saved to {self.output_dir}/2d_projections.png")
    
    def save_3d_visualizations(self, source, target, source_transformed):
        """Save 3D visualizations of the point clouds before and after registration"""
        try:
            print("Creating 3D visualizations...")
            
            # Prepare colored point clouds
            source_vis = copy.deepcopy(source)
            source_vis.paint_uniform_color([1, 0, 0])  # Red
            
            target_vis = copy.deepcopy(target)
            target_vis.paint_uniform_color([0, 0, 1])  # Blue
            
            source_transformed_vis = copy.deepcopy(source_transformed)
            source_transformed_vis.paint_uniform_color([0, 1, 0])  # Green
            
            # Create coordinate frames
            frame_size = 0.1
            source_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
            target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
            source_transformed_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
            
            # Create combined point clouds
            combined_before = copy.deepcopy(source_vis)
            combined_before.points = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(source_vis.points), np.asarray(target_vis.points))))
            combined_before.colors = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(source_vis.colors), np.asarray(target_vis.colors))))
            
            combined_after = copy.deepcopy(source_transformed_vis)
            combined_after.points = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(source_transformed_vis.points), np.asarray(target_vis.points))))
            combined_after.colors = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(source_transformed_vis.colors), np.asarray(target_vis.colors))))
            
            # Try to use off-screen rendering for 3D views
            try:
                renderer = self.create_off_screen_renderer()
                
                # Define camera positions for multiple views
                camera_positions = [
                    {"pos": [1, 1, 1], "name": "isometric"},
                    {"pos": [1, 0, 0], "name": "right"},
                    {"pos": [0, 1, 0], "name": "top"},
                    {"pos": [0, 0, 1], "name": "front"}
                ]
                
                # Render source point cloud
                os.makedirs(f'{self.output_dir}/3d_views', exist_ok=True)
                
                # Render before registration views
                for cam in camera_positions:
                    self.render_point_cloud_view(
                        renderer, [source_vis, target_vis, source_frame, target_frame],
                        camera_position=cam["pos"],
                        filename=f'{self.output_dir}/3d_views/before_registration_{cam["name"]}.png'
                    )
                
                # Render after registration views
                for cam in camera_positions:
                    self.render_point_cloud_view(
                        renderer, [source_transformed_vis, target_vis, source_transformed_frame, target_frame],
                        camera_position=cam["pos"],
                        filename=f'{self.output_dir}/3d_views/after_registration_{cam["name"]}.png'
                    )
                
                print(f"3D visualizations saved to {self.output_dir}/3d_views/")
            except Exception as e:
                print(f"Warning: Off-screen rendering failed: {str(e)}")
                print("Falling back to saving point clouds as PLY files...")
                
                # Save as PLY files if rendering fails
                o3d.io.write_point_cloud(f'{self.output_dir}/source.ply', source_vis)
                o3d.io.write_point_cloud(f'{self.output_dir}/target.ply', target_vis)
                o3d.io.write_point_cloud(f'{self.output_dir}/source_transformed.ply', source_transformed_vis)
                o3d.io.write_point_cloud(f'{self.output_dir}/combined_before.ply', combined_before)
                o3d.io.write_point_cloud(f'{self.output_dir}/combined_after.ply', combined_after)
                
                print(f"Point clouds saved as PLY files in {self.output_dir}/")
        
        except Exception as e:
            print(f"Error in 3D visualization: {str(e)}")
    
    def create_registration_error_visualization(self, source_transformed, target):
        """Create visualization showing point-wise registration error"""
        print("Creating error visualization...")
        
        # Convert to numpy arrays
        source_pts = np.asarray(source_transformed.points)
        target_pts = np.asarray(target.points)
        
        try:
            # Build a kd-tree for target points
            tree = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_pts)
            
            # Find the nearest target point for each source point
            distances, _ = tree.kneighbors(source_pts)
            distances = distances.flatten()
            
            # Create colored point cloud for visualization
            error_vis = copy.deepcopy(source_transformed)
            
            # Create a color map from blue (good) to red (bad)
            # Normalize distances
            max_dist = np.percentile(distances, 95)  # Use 95th percentile to avoid outliers
            normalized_distances = np.clip(distances / max_dist, 0, 1)
            
            # Create colors (blue for good alignment, red for poor alignment)
            colors = np.zeros((len(normalized_distances), 3))
            colors[:, 0] = normalized_distances  # Red channel (high for poor alignment)
            colors[:, 2] = 1 - normalized_distances  # Blue channel (high for good alignment)
            
            # Apply colors to the point cloud
            error_vis.colors = o3d.utility.Vector3dVector(colors)
            
            # Save the error visualization
            o3d.io.write_point_cloud(f'{self.output_dir}/registration_error.ply', error_vis)
            
            # Plot histogram of distances
            plt.figure(figsize=(10, 6))
            plt.hist(distances, bins=50, color='blue', alpha=0.7)
            plt.axvline(x=np.mean(distances), color='r', linestyle='--', label=f'Mean: {np.mean(distances):.4f}')
            plt.axvline(x=np.median(distances), color='g', linestyle='--', label=f'Median: {np.median(distances):.4f}')
            plt.xlabel('Distance to Nearest Target Point')
            plt.ylabel('Frequency')
            plt.title('Distribution of Registration Errors')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'{self.output_dir}/error_histogram.png')
            
            print(f"Error visualization saved to {self.output_dir}/registration_error.ply")
            print(f"Error histogram saved to {self.output_dir}/error_histogram.png")
            
            # Return error statistics
            error_stats = {
                'mean': np.mean(distances),
                'median': np.median(distances),
                'std': np.std(distances),
                'max': np.max(distances),
                'min': np.min(distances)
            }
            
            return error_stats
            
        except Exception as e:
            print(f"Error in registration error visualization: {str(e)}")
            return None
    
    def create_registration_summary(self, source, target, initial_transform, final_transform, error_stats=None):
        """Create a summary of the registration process"""
        print("Creating registration summary...")
        
        # Extract rotation and translation from initial and final transforms
        initial_rotation = initial_transform[:3, :3]
        initial_translation = initial_transform[:3, 3]
        initial_euler = Rotation.from_matrix(initial_rotation).as_euler('xyz', degrees=True)
        
        final_rotation = final_transform[:3, :3]
        final_translation = final_transform[:3, 3]
        final_euler = Rotation.from_matrix(final_rotation).as_euler('xyz', degrees=True)
        
        # Create summary document
        with open(f'{self.output_dir}/registration_summary.txt', 'w') as f:
            f.write("=== Point Cloud Registration Summary ===\n\n")
            
            # Point cloud information
            f.write("Point Cloud Information:\n")
            f.write(f"Source: {len(source.points)} points\n")
            f.write(f"Target: {len(target.points)} points\n\n")
            
            # Initial alignment (RANSAC)
            f.write("Initial Alignment (RANSAC):\n")
            f.write(f"Fitness: {self.fitness_history[0]:.6f}\n")
            f.write(f"RMSE: {self.rmse_history[0]:.6f}\n")
            f.write("Translation (X, Y, Z): ")
            f.write(f"{initial_translation[0]:.6f}, {initial_translation[1]:.6f}, {initial_translation[2]:.6f}\n")
            f.write("Rotation (Euler angles in degrees, XYZ): ")
            f.write(f"{initial_euler[0]:.6f}, {initial_euler[1]:.6f}, {initial_euler[2]:.6f}\n\n")
            
            # Final alignment (After ICP)
            f.write("Final Alignment (After ICP):\n")
            f.write(f"Fitness: {self.fitness_history[-1]:.6f}\n")
            f.write(f"RMSE: {self.rmse_history[-1]:.6f}\n")
            f.write("Translation (X, Y, Z): ")
            f.write(f"{final_translation[0]:.6f}, {final_translation[1]:.6f}, {final_translation[2]:.6f}\n")
            f.write("Rotation (Euler angles in degrees, XYZ): ")
            f.write(f"{final_euler[0]:.6f}, {final_euler[1]:.6f}, {final_euler[2]:.6f}\n\n")
            
            # Registration improvement
            f.write("Registration Improvement:\n")
            f.write(f"Fitness improvement: {(self.fitness_history[-1] - self.fitness_history[0]) / self.fitness_history[0] * 100:.2f}%\n")
            f.write(f"RMSE improvement: {(self.rmse_history[0] - self.rmse_history[-1]) / self.rmse_history[0] * 100:.2f}%\n\n")
            
            # Error statistics if available
            if error_stats:
                f.write("Registration Error Statistics:\n")
                f.write(f"Mean distance: {error_stats['mean']:.6f}\n")
                f.write(f"Median distance: {error_stats['median']:.6f}\n")
                f.write(f"Standard deviation: {error_stats['std']:.6f}\n")
                f.write(f"Minimum distance: {error_stats['min']:.6f}\n")
                f.write(f"Maximum distance: {error_stats['max']:.6f}\n\n")
            
            # Final transformation matrix
            f.write("Final Transformation Matrix:\n")
            for i in range(4):
                f.write(f"{final_transform[i, 0]:.6f} {final_transform[i, 1]:.6f} ")
                f.write(f"{final_transform[i, 2]:.6f} {final_transform[i, 3]:.6f}\n")
            
        print(f"Registration summary saved to {self.output_dir}/registration_summary.txt")
    
    def create_composite_visualization(self):
        """Create a composite visualization of all results"""
        try:
            print("Creating composite visualization...")
            
            # Define paths to all generated images
            image_paths = {
                'metrics': f'{self.output_dir}/registration_metrics.png',
                'evolution': f'{self.output_dir}/transformation_evolution.png',
                'projections': f'{self.output_dir}/2d_projections.png',
                'error_hist': f'{self.output_dir}/error_histogram.png',
            }
            
            # Add 3D views if they exist
            view_dir = f'{self.output_dir}/3d_views'
            if os.path.exists(view_dir):
                for view in ['isometric', 'front', 'top', 'right']:
                    before_path = f'{view_dir}/before_registration_{view}.png'
                    after_path = f'{view_dir}/after_registration_{view}.png'
                    
                    if os.path.exists(before_path):
                        image_paths[f'3d_before_{view}'] = before_path
                    if os.path.exists(after_path):
                        image_paths[f'3d_after_{view}'] = after_path
            
            # Check which images actually exist
            existing_images = {}
            for key, path in image_paths.items():
                if os.path.exists(path):
                    existing_images[key] = path
            
            # If there are no images, return
            if not existing_images:
                print("No visualization images found to create composite.")
                return
            
            # Create an HTML report
            with open(f'{self.output_dir}/registration_report.html', 'w') as f:
                f.write('<!DOCTYPE html>\n')
                f.write('<html lang="en">\n')
                f.write('<head>\n')
                f.write('    <meta charset="UTF-8">\n')
                f.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
                f.write('    <title>Point Cloud Registration Report</title>\n')
                f.write('    <style>\n')
                f.write('        body { font-family: Arial, sans-serif; margin: 20px; }\n')
                f.write('        h1, h2 { color: #333; }\n')
                f.write('        .section { margin-bottom: 30px; }\n')
                f.write('        .image-container { text-align: center; margin: 20px 0; }\n')
                f.write('        img { max-width: 100%; border: 1px solid #ddd; }\n')
                f.write('        .flex-container { display: flex; flex-wrap: wrap; justify-content: center; }\n')
                f.write('        .flex-item { margin: 10px; max-width: 45%; }\n')
                f.write('        .caption { font-style: italic; margin-top: 5px; }\n')
                f.write('        pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }\n')
                f.write('    </style>\n')
                f.write('</head>\n')
                f.write('<body>\n')
                f.write('    <h1>Point Cloud Registration Report</h1>\n')
                
                # Add registration summary
                f.write('    <div class="section">\n')
                f.write('        <h2>Registration Summary</h2>\n')
                if os.path.exists(f'{self.output_dir}/registration_summary.txt'):
                    with open(f'{self.output_dir}/registration_summary.txt', 'r') as summary_file:
                        f.write('        <pre>\n')
                        f.write(summary_file.read())
                        f.write('        </pre>\n')
                else:
                    f.write('        <p>Registration summary not available.</p>\n')
                f.write('    </div>\n')
                
                # Add 3D Visualization section
                if any(key.startswith('3d_') for key in existing_images.keys()):
                    f.write('    <div class="section">\n')
                    f.write('        <h2>3D Visualization</h2>\n')
                    
                    # Before registration views
                    f.write('        <h3>Before Registration</h3>\n')
                    f.write('        <div class="flex-container">\n')
                    for view in ['isometric', 'front', 'top', 'right']:
                        key = f'3d_before_{view}'
                        if key in existing_images:
                            f.write(f'            <div class="flex-item">\n')
                            f.write(f'                <img src="{existing_images[key]}" alt="Before Registration - {view} view">\n')
                            f.write(f'                <div class="caption">{view.capitalize()} View</div>\n')
                            f.write(f'            </div>\n')
                    f.write('        </div>\n')
                    
                    # After registration views
                    f.write('        <h3>After Registration</h3>\n')
                    f.write('        <div class="flex-container">\n')
                    for view in ['isometric', 'front', 'top', 'right']:
                        key = f'3d_after_{view}'
                        if key in existing_images:
                            f.write(f'            <div class="flex-item">\n')
                            f.write(f'                <img src="{existing_images[key]}" alt="After Registration - {view} view">\n')
                            f.write(f'                <div class="caption">{view.capitalize()} View</div>\n')
                            f.write(f'            </div>\n')
                    f.write('        </div>\n')
                    f.write('    </div>\n')
                
                # Add 2D Projections
                if 'projections' in existing_images:
                    f.write('    <div class="section">\n')
                    f.write('        <h2>2D Projections</h2>\n')
                    f.write('        <div class="image-container">\n')
                    f.write(f'            <img src="{existing_images["projections"]}" alt="2D Projections">\n')
                    f.write('        </div>\n')
                    f.write('    </div>\n')
                
                # Add Registration Metrics
                if 'metrics' in existing_images or 'evolution' in existing_images:
                    f.write('    <div class="section">\n')
                    f.write('        <h2>Registration Metrics</h2>\n')
                    f.write('        <div class="flex-container">\n')
                    if 'metrics' in existing_images:
                        f.write('            <div class="flex-item">\n')
                        f.write(f'                <img src="{existing_images["metrics"]}" alt="Registration Metrics">\n')
                        f.write('                <div class="caption">Registration Fitness and RMSE</div>\n')
                        f.write('            </div>\n')
                    if 'evolution' in existing_images:
                        f.write('            <div class="flex-item">\n')
                        f.write(f'                <img src="{existing_images["evolution"]}" alt="Transformation Evolution">\n')
                        f.write('                <div class="caption">Transformation Evolution</div>\n')
                        f.write('            </div>\n')
                    f.write('        </div>\n')
                    f.write('    </div>\n')
                
                # Add Error Analysis
                if 'error_hist' in existing_images:
                    f.write('    <div class="section">\n')
                    f.write('        <h2>Error Analysis</h2>\n')
                    f.write('        <div class="image-container">\n')
                    f.write(f'            <img src="{existing_images["error_hist"]}" alt="Error Histogram">\n')
                    f.write('            <div class="caption">Distribution of Registration Errors</div>\n')
                    f.write('        </div>\n')
                    f.write('    </div>\n')
                
                f.write('</body>\n')
                f.write('</html>\n')
            
            print(f"Composite visualization saved to {self.output_dir}/registration_report.html")
            
        except Exception as e:
            print(f"Error creating composite visualization: {str(e)}")
    
    def run_registration(self, source_file, target_file):
        """Run the complete registration pipeline"""
        # Load point clouds
        source, target = self.load_point_clouds(source_file, target_file)
        original_source = copy.deepcopy(source)
        original_target = copy.deepcopy(target)
        
        # Preprocess point clouds
        source_processed, source_feat, source_scale = self.preprocess_point_cloud(source, "source")
        target_processed, target_feat, target_scale = self.preprocess_point_cloud(target, "target")
        
        # Global registration (RANSAC)
        initial_transform = self.execute_global_registration(
            source_processed, target_processed, source_feat, target_feat)
        
        # Apply initial transformation to source for visualization
        source_initial = copy.deepcopy(original_source)
        source_initial.transform(initial_transform)
        
        # Refine with ICP
        final_transform_processed = self.refine_registration(
            source_processed, target_processed, initial_transform)
        
        # Scale back the transformation
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = scale_matrix[1, 1] = scale_matrix[2, 2] = source_scale / target_scale
        final_transform = np.matmul(final_transform_processed, scale_matrix)
        
        # Apply final transformation to original source
        source_final = copy.deepcopy(original_source)
        source_final.transform(final_transform)
        
        # Create all visualizations
        self.plot_registration_metrics()
        self.visualize_transformation_evolution()
        self.create_2d_projection_visualizations(original_source, original_target, source_final)
        self.save_3d_visualizations(original_source, original_target, source_final)
        error_stats = self.create_registration_error_visualization(source_final, original_target)
        self.create_registration_summary(original_source, original_target, initial_transform, final_transform, error_stats)
        self.create_composite_visualization()
        
        # Print final transformation matrix
        print("\nFinal Transformation Matrix:")
        print(np.array_str(final_transform, precision=4))
        
        # Extract rotation and translation
        rotation = final_transform[:3, :3]
        translation = final_transform[:3, 3]
        
        # Convert rotation matrix to Euler angles (in degrees)
        euler_angles = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True)
        
        print("\nTransformation Parameters:")
        print(f"Translation (x, y, z): {translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}")
        print(f"Rotation (Euler angles in degrees, xyz): {euler_angles[0]:.4f}, {euler_angles[1]:.4f}, {euler_angles[2]:.4f}")
        
        print(f"\nAll results and visualizations saved to {self.output_dir}/")
        print(f"Open {self.output_dir}/registration_report.html in a web browser to view all visualizations.")
        
        return final_transform

def main():
    parser = argparse.ArgumentParser(description='Robust Point Cloud Registration')
    parser.add_argument('--source', type=str, required=True, help='Path to source PCD file')
    parser.add_argument('--target', type=str, required=True, help='Path to target PLY file')
    parser.add_argument('--vis', action='store_true', help='Enable Open3D visualization if possible')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel size for downsampling')
    parser.add_argument('--ransac_iter', type=int, default=100000, help='RANSAC iterations')
    parser.add_argument('--icp_iter', type=int, default=100, help='Maximum ICP iterations')
    parser.add_argument('--output_dir', type=str, default='registration_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create registration object
    registration = PointCloudRegistration(use_visualization=args.vis)
    registration.voxel_size = args.voxel_size
    registration.ransac_iter = args.ransac_iter
    registration.icp_max_iter = args.icp_iter
    registration.output_dir = args.output_dir
    os.makedirs(registration.output_dir, exist_ok=True)
    
    try:
        transform = registration.run_registration(args.source, args.target)
        print("\nRegistration completed successfully!")
    except Exception as e:
        print(f"\nError during registration: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()