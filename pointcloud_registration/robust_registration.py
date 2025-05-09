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

class PointCloudProcessing:
    def __init__(self):
        # Parameters for preprocessing
        self.voxel_size = 0.01  # Downsampling voxel size for registration
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
        
        # Parameters for placement
        self.max_scene_points = 50000  # Maximum number of points in scene after downsampling
        
        # Output directory
        self.output_dir = 'registration_results'
        os.makedirs(self.output_dir, exist_ok=True)
    
    # ============ LOADING AND PREPROCESSING ============
    
    def load_point_clouds(self, source_file, reference_file, scene_file=None):
        """Load all point clouds"""
        print(f"Loading source point cloud (ROI): {source_file}")
        if source_file.endswith('.pcd'):
            source = o3d.io.read_point_cloud(source_file)
        else:
            raise ValueError("Source file must be a PCD file")
        
        print(f"Loading reference model: {reference_file}")
        if reference_file.endswith('.ply'):
            reference = o3d.io.read_point_cloud(reference_file)
        else:
            raise ValueError("Reference file must be a PLY file")
        
        # Check if scene file is provided
        if scene_file:
            print(f"Loading scene point cloud: {scene_file}")
            if scene_file.endswith('.pcd'):
                scene = o3d.io.read_point_cloud(scene_file)
            else:
                raise ValueError("Scene file must be a PCD file")
            print(f"Scene point cloud has {len(scene.points)} points")
        else:
            scene = None
        
        print(f"Source point cloud has {len(source.points)} points")
        print(f"Reference model has {len(reference.points)} points")
        
        return source, reference, scene
    
    def adaptive_downsample_point_cloud(self, pcd, target_points, name="point cloud"):
        """Downsample point cloud to approximately target_points"""
        original_points = len(pcd.points)
        print(f"Adaptively downsampling {name} from {original_points} points to ~{target_points} points")
        
        if original_points <= target_points:
            print(f"No downsampling needed, {name} already has fewer points than target")
            return copy.deepcopy(pcd)
        
        # Calculate voxel size to achieve approximately target_points
        # This is an approximation based on uniform distribution assumption
        estimated_voxel_size = (original_points / target_points) ** (1/3) * 0.01
        
        # Start with estimated voxel size
        voxel_size = estimated_voxel_size
        downsampled = pcd.voxel_down_sample(voxel_size)
        current_points = len(downsampled.points)
        
        # Binary search to find better voxel size if estimation is far off
        if abs(current_points - target_points) > 0.2 * target_points:
            min_voxel = 0.001
            max_voxel = 0.5
            
            if current_points > target_points:
                min_voxel = voxel_size
            else:
                max_voxel = voxel_size
            
            # Binary search with max 10 iterations
            for _ in range(10):
                voxel_size = (min_voxel + max_voxel) / 2
                downsampled = pcd.voxel_down_sample(voxel_size)
                current_points = len(downsampled.points)
                
                if abs(current_points - target_points) < 0.1 * target_points:
                    break
                
                if current_points > target_points:
                    min_voxel = voxel_size
                else:
                    max_voxel = voxel_size
        
        print(f"Downsampled {name} to {len(downsampled.points)} points using voxel size {voxel_size:.6f}")
        return downsampled
    
    def preprocess_point_cloud(self, pcd, name="point cloud"):
        """Preprocess point cloud: downsampling, normal estimation, outlier removal"""
        print(f"Preprocessing {name}...")
        
        # Make a copy to avoid modifying the original
        processed = copy.deepcopy(pcd)
        
        # Check if point cloud is empty
        if len(processed.points) == 0:
            raise ValueError(f"Empty {name}")
        
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
        
        return cleaned, features, scale, original_center
    
    # ============ REGISTRATION ============
    
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
        self.transformation_history = [result.transformation]
        self.fitness_history = [result.fitness]
        self.rmse_history = [result.inlier_rmse]
        
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
    
    # ============ MODEL PLACEMENT ============
    
    def place_reference_in_scene(self, reference, scene, transformation_src_to_ref, source_center, reference_center):
        """Place reference model in scene using the inverse transformation with correct centering"""
        print("Placing reference model in scene...")
        
        # Calculate inverse transformation (reference to source)
        transformation_ref_to_src = np.linalg.inv(transformation_src_to_ref)
        
        # Adjust transformation to account for centering
        # Create translation matrices for the centers
        T_src = np.eye(4)
        T_src[:3, 3] = source_center
        
        T_ref = np.eye(4)
        T_ref[:3, 3] = -reference_center
        
        # The complete transformation: T_src * T_ref_to_src * T_ref
        adjusted_transform = np.matmul(T_src, np.matmul(transformation_ref_to_src, T_ref))
        
        # Apply transformation to reference
        reference_in_scene = copy.deepcopy(reference)
        reference_in_scene.transform(adjusted_transform)
        
        # Color the reference model with a distinct bright color
        reference_in_scene.paint_uniform_color([1, 0.3, 0])  # Bright orange-red
        
        # Create a combined point cloud
        # First, make a copy of the scene to preserve its RGB information
        scene_copy = copy.deepcopy(scene)
        
        # Combine reference and scene
        combined = copy.deepcopy(reference_in_scene)
        combined.points = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(reference_in_scene.points), np.asarray(scene_copy.points))))
        
        # Combine colors - keeping original scene colors
        if len(scene_copy.colors) > 0:
            combined.colors = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(reference_in_scene.colors), np.asarray(scene_copy.colors))))
        
        return reference_in_scene, combined, adjusted_transform
    
    def save_results(self, reference_in_scene, scene, combined):
        """Save placement results"""
        print("Saving results...")
        
        # Save individual point clouds
        o3d.io.write_point_cloud(f'{self.output_dir}/reference_in_scene.pcd', reference_in_scene)
        o3d.io.write_point_cloud(f'{self.output_dir}/scene.pcd', scene)
        o3d.io.write_point_cloud(f'{self.output_dir}/combined_result.pcd', combined)
        
        print(f"Results saved to {self.output_dir}/")
    
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
            material = o3d.visualization.rendering.MaterialRecord()
            material.shader = "defaultLit"
            material.point_size = 5.0  # Larger point size for better visibility
            
            # Set material color if point cloud has colors
            if len(geom.colors) > 0:
                first_color = np.asarray(geom.colors)[0]
                material.base_color = [first_color[0], first_color[1], first_color[2], 1.0]
            
            renderer.scene.add_geometry(f"pointcloud_{i}", geom, material)
        
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
    
    def create_visualizations(self, reference_in_scene, scene_downsampled):
        """Create visualizations of the placement result"""
        print("Creating visualizations...")
        
        try:
            # Try to use off-screen rendering for 3D views
            renderer = self.create_off_screen_renderer()
            
            # Define camera positions for multiple views
            camera_positions = [
                {"pos": [1, 1, 1], "name": "isometric"},
                {"pos": [1, 0, 0], "name": "right"},
                {"pos": [0, 1, 0], "name": "top"},
                {"pos": [0, 0, 1], "name": "front"}
            ]
            
            # Create directory for 3D views
            os.makedirs(f'{self.output_dir}/views', exist_ok=True)
            
            # Calculate center of reference model for focus
            ref_center = np.mean(np.asarray(reference_in_scene.points), axis=0)
            
            # Render views of placement result
            for cam in camera_positions:
                self.render_point_cloud_view(
                    renderer, [reference_in_scene, scene_downsampled],
                    camera_position=[pos * 2 for pos in cam["pos"]],  # Scale to ensure good view
                    look_at=ref_center,
                    filename=f'{self.output_dir}/views/placement_{cam["name"]}.png'
                )
                
                # Also render close-up of reference in scene
                self.render_point_cloud_view(
                    renderer, [reference_in_scene, scene_downsampled],
                    camera_position=[ref_center[i] + cam["pos"][i] * 0.5 for i in range(3)],  # Closer view
                    look_at=ref_center,
                    filename=f'{self.output_dir}/views/placement_closeup_{cam["name"]}.png'
                )
            
            print(f"3D visualizations saved to {self.output_dir}/views/")
            
        except Exception as e:
            print(f"Warning: Visualization failed: {str(e)}")
            print("Falling back to saving point clouds only.")
    
    # ============ MAIN PROCESSES ============
    
    def run_registration(self, source_file, reference_file):
        """Run the registration pipeline"""
        print("Starting registration process...")
        
        # Load point clouds
        source, reference, _ = self.load_point_clouds(source_file, reference_file)
        
        # Preprocess point clouds
        source_processed, source_feat, source_scale, source_center = self.preprocess_point_cloud(source, "source")
        reference_processed, reference_feat, reference_scale, reference_center = self.preprocess_point_cloud(reference, "reference")
        
        # Global registration (RANSAC)
        initial_transform = self.execute_global_registration(
            source_processed, reference_processed, source_feat, reference_feat)
        
        # Refine with ICP
        final_transform_processed = self.refine_registration(
            source_processed, reference_processed, initial_transform)
        
        # Scale back the transformation
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = scale_matrix[1, 1] = scale_matrix[2, 2] = source_scale / reference_scale
        final_transform = np.matmul(final_transform_processed, scale_matrix)
        
        # Save the transformation matrix
        np.savetxt(f'{self.output_dir}/transformation_src_to_ref.txt', final_transform)
        
        # Save centers for later use in placement
        np.savetxt(f'{self.output_dir}/source_center.txt', source_center)
        np.savetxt(f'{self.output_dir}/reference_center.txt', reference_center)
        
        # Visualize registration progress
        self.plot_registration_metrics()
        self.visualize_transformation_evolution()
        
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
        
        return final_transform, source_center, reference_center
    
    def run_model_placement(self, source_file, reference_file, scene_file, transform_file=None):
        """Run the model placement pipeline"""
        print("Starting model placement process...")
        
        # Load source centers and transformation matrix
        if transform_file and os.path.exists(transform_file):
            print(f"Loading transformation from file: {transform_file}")
            transformation_src_to_ref = np.loadtxt(transform_file)
            
            # Try to load centers if they exist
            source_center_file = os.path.join(os.path.dirname(transform_file), 'source_center.txt')
            reference_center_file = os.path.join(os.path.dirname(transform_file), 'reference_center.txt')
            
            if os.path.exists(source_center_file) and os.path.exists(reference_center_file):
                source_center = np.loadtxt(source_center_file)
                reference_center = np.loadtxt(reference_center_file)
                print("Loaded source and reference centers from files")
            else:
                # If centers don't exist, load the clouds and compute centers
                source_temp, reference_temp, _ = self.load_point_clouds(source_file, reference_file)
                source_center = np.mean(np.asarray(source_temp.points), axis=0)
                reference_center = np.mean(np.asarray(reference_temp.points), axis=0)
                print("Computed centers from point clouds")
        else:
            # Perform registration
            transformation_src_to_ref, source_center, reference_center = self.run_registration(source_file, reference_file)
        
        # Load point clouds
        _, reference, scene = self.load_point_clouds(source_file, reference_file, scene_file)
        
        # Downsample scene to max points
        scene_downsampled = self.adaptive_downsample_point_cloud(scene, self.max_scene_points, "scene")
        
        # Place reference in scene
        reference_in_scene, combined, adjusted_transform = self.place_reference_in_scene(
            reference, scene_downsampled, transformation_src_to_ref, source_center, reference_center)
        
        # Save results
        self.save_results(reference_in_scene, scene_downsampled, combined)
        
        # Create visualizations
        self.create_visualizations(reference_in_scene, scene_downsampled)
        
        # Save the adjusted transformation
        np.savetxt(f'{self.output_dir}/adjusted_transform.txt', adjusted_transform)
        
        print("\nModel placement completed successfully!")
        print(f"Results saved to {self.output_dir}/")
        print("\nOutput files:")
        print(f"- Reference in scene: {self.output_dir}/reference_in_scene.pcd")
        print(f"- Scene: {self.output_dir}/scene.pcd")
        print(f"- Combined result: {self.output_dir}/combined_result.pcd")
        print(f"- Transformation matrices: {self.output_dir}/transformation_src_to_ref.txt")
        print(f"- Adjusted transformation: {self.output_dir}/adjusted_transform.txt")
        
        if os.path.exists(f'{self.output_dir}/views'):
            print(f"- Visualizations: {self.output_dir}/views/*.png")
        
        return adjusted_transform
    
    def run_full_pipeline(self, source_file, reference_file, scene_file=None):
        """Run the full pipeline: registration and placement if scene is provided"""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Run registration
        transformation, source_center, reference_center = self.run_registration(source_file, reference_file)
        
        # If scene file is provided, also run model placement
        if scene_file:
            print("\n" + "="*50)
            print("Registration complete. Proceeding with model placement...")
            print("="*50 + "\n")
            adjusted_transform = self.run_model_placement(source_file, reference_file, scene_file)
            return adjusted_transform
        
        return transformation

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration and Placement')
    parser.add_argument('--source', type=str, required=True, help='Path to source PCD file')
    parser.add_argument('--reference', type=str, required=True, help='Path to reference PLY file')
    parser.add_argument('--scene', type=str, help='Path to scene PCD file (optional)')
    parser.add_argument('--transform', type=str, help='Path to transformation matrix file (optional)')
    parser.add_argument('--output_dir', type=str, default='processing_results', help='Output directory')
    parser.add_argument('--max_scene_points', type=int, default=75000, help='Maximum number of points in downsampled scene')
    
    args = parser.parse_args()
    
    # Create processing object
    processor = PointCloudProcessing()
    processor.output_dir = args.output_dir
    processor.max_scene_points = args.max_scene_points
    
    try:
        if args.transform and args.scene:
            # Run placement with provided transformation
            processor.run_model_placement(args.source, args.reference, args.scene, args.transform)
        elif args.scene:
            # Run full pipeline
            processor.run_full_pipeline(args.source, args.reference, args.scene)
        else:
            # Run registration only
            processor.run_registration(args.source, args.reference)
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()