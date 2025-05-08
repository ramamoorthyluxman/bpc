import open3d as o3d
import open3d.visualization.rendering as rendering
import numpy as np
import os
import argparse


def render_pcd_neutral_pose(pcd_path, output_dir, point_size=5):
    """
    Render a point cloud file in a neutral pose (zero rotations).
    
    Args:
        pcd_path: Path to the point cloud file
        output_dir: Directory to save the rendered image
        point_size: Size of the points in the rendering
    """
    # Create the base output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a subdirectory with the same name as the point cloud file (without extension)
    pcd_basename = os.path.splitext(os.path.basename(pcd_path))[0]
    pcd_output_dir = os.path.join(output_dir, pcd_basename)
    os.makedirs(pcd_output_dir, exist_ok=True)
    
    print(f"Created output directory: {pcd_output_dir}")
    
    # Load the point cloud
    print(f"Loading point cloud from {pcd_path}...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # Get the center and scale of the point cloud to normalize
    center = pcd.get_center()
    max_bound = pcd.get_max_bound()
    min_bound = pcd.get_min_bound()
    extent = np.linalg.norm(max_bound - min_bound)
    
    # Set up renderer
    width, height = 640, 480
    renderer = rendering.OffscreenRenderer(width, height)
    
    # Set up material for points
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size
    
    # Set up scene
    renderer.scene.set_background([0.5, 0.5, 0.5, 1.0])  # Medium gray background
    
    # Add lighting
    renderer.scene.scene.set_sun_light([1, 1, 1], [1, 1, 1], 75000)
    renderer.scene.scene.enable_sun_light(True)
    
    # Add the point cloud to the scene
    renderer.scene.clear_geometry()
    renderer.scene.add_geometry("pcd", pcd, mat)
    
    # Calculate camera position - look at the point cloud from the front
    eye = center + np.array([0, 0, extent * 1.5])  # Camera position
    up = np.array([0, 1, 0], dtype=np.float32)     # Up direction (Y axis)
    
    # Set up the camera
    renderer.setup_camera(60.0, center.astype(np.float32), eye.astype(np.float32), up)
    
    # Render the image
    print("Rendering neutral pose image...")
    img = renderer.render_to_image()
    
    # Save the image to the PCD-specific directory
    output_path = os.path.join(pcd_output_dir, "neutral_pose.png")
    o3d.io.write_image(output_path, img)
    print(f"Image saved to {output_path}")
    
    # Also save views from other angles if needed
    angles = [
        ("front", np.array([0, 0, extent * 1.5])),
        ("side", np.array([extent * 1.5, 0, 0])),
        ("top", np.array([0, extent * 1.5, 0]))
    ]
    
    for name, direction in angles:
        # Look at the object from different directions
        eye = center + direction
        renderer.setup_camera(60.0, center.astype(np.float32), eye.astype(np.float32), up)
        img = renderer.render_to_image()
        # Save to the PCD-specific directory
        view_path = os.path.join(pcd_output_dir, f"neutral_pose_{name}.png")
        o3d.io.write_image(view_path, img)
        print(f"{name.capitalize()} view saved to {view_path}")


def main():
    parser = argparse.ArgumentParser(description='Render point cloud in neutral pose')
    parser.add_argument('--pcd', type=str, required=True, help='Path to the point cloud file (.pcd)')
    parser.add_argument('--output', type=str, default='pcd_renders', help='Output directory for rendered images')
    parser.add_argument('--point-size', type=int, default=5, help='Size of points in the rendering')
    args = parser.parse_args()
    
    render_pcd_neutral_pose(args.pcd, args.output, args.point_size)


if __name__ == "__main__":
    main()