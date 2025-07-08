import os
import pandas as pd
from pathlib import Path
import re
import json

def load_camera_info(camera_info_files):
    """
    Load camera calibration information from JSON files.
    
    Args:
        camera_info_files (dict): Dictionary mapping camera_id to JSON file path
    
    Returns:
        dict: Dictionary mapping camera_id to camera parameters
    """
    camera_info = {}
    
    for camera_id, json_path in camera_info_files.items():
        if json_path and Path(json_path).exists():
            try:
                with open(json_path, 'r') as f:
                    camera_info[camera_id] = json.load(f)
                print(f"Loaded camera info for {camera_id}: {json_path}")
            except Exception as e:
                print(f"Error loading camera info for {camera_id}: {e}")
                camera_info[camera_id] = {}
        else:
            print(f"Camera info file not found for {camera_id}: {json_path}")
            camera_info[camera_id] = {}
    
    return camera_info

def load_scene_camera_positions(scene_dir, camera_types):
    """
    Load scene-specific camera position files for all camera types.
    
    Args:
        scene_dir (Path): Path to the scene directory
        camera_types (list): List of camera types to look for
    
    Returns:
        dict: Dictionary mapping camera_id to camera position data for each image
    """
    scene_camera_positions = {}
    
    for camera_type in camera_types:
        camera_file = scene_dir / f"scene_camera_{camera_type}.json"
        if camera_file.exists():
            try:
                with open(camera_file, 'r') as f:
                    scene_camera_positions[camera_type] = json.load(f)
                print(f"  Loaded scene camera positions for {camera_type}")
            except Exception as e:
                print(f"  Error loading scene camera positions for {camera_type}: {e}")
                scene_camera_positions[camera_type] = {}
        else:
            scene_camera_positions[camera_type] = {}
    
    return scene_camera_positions

def generate_dataset_csv(root_directory, camera_info_files=None, output_csv_path="test_dataset.csv"):
    """
    Generate a CSV file from a dataset directory structure with camera calibration info.
    
    Args:
        root_directory (str): Path to the root directory containing scene folders
        camera_info_files (dict): Dictionary mapping camera_id to JSON file path
        output_csv_path (str): Path where the CSV file will be saved
    
    Returns:
        pd.DataFrame: DataFrame containing the dataset information
    """
    
    # List to store all the data rows
    data_rows = []
    
    # Load camera calibration information
    camera_info = {}
    if camera_info_files:
        camera_info = load_camera_info(camera_info_files)
    
    # Convert to Path object for easier manipulation
    root_path = Path(root_directory)
    
    if not root_path.exists():
        raise ValueError(f"Root directory does not exist: {root_directory}")
    
    # Get all scene directories (folders with numeric names like 00000, 00001)
    scene_dirs = [d for d in root_path.iterdir() 
                  if d.is_dir() and re.match(r'^\d+$', d.name)]
    
    if not scene_dirs:
        print("No scene directories found (looking for folders with numeric names)")
        return pd.DataFrame()
    
    # Sort scene directories to ensure consistent ordering
    scene_dirs.sort(key=lambda x: int(x.name))
    
    print(f"Found {len(scene_dirs)} scene directories")
    
    # Camera types to look for
    camera_types = ['cam1', 'cam2', 'cam3', 'photoneo']
    
    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        print(f"Processing scene: {scene_id}")
        
        # Load scene-specific camera position files
        scene_camera_positions = load_scene_camera_positions(scene_dir, camera_types)
        
        # For each camera type, find matching RGB and depth folders
        for camera_type in camera_types:
            rgb_folder = scene_dir / f"rgb_{camera_type}"
            depth_folder = scene_dir / f"depth_{camera_type}"
            
            # Check if both RGB and depth folders exist
            if rgb_folder.exists() and depth_folder.exists():
                # Get all image files from RGB folder
                rgb_images = [f for f in rgb_folder.iterdir() 
                             if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']]
                
                # Get all image files from depth folder
                depth_images = [f for f in depth_folder.iterdir() 
                               if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.exr']]
                
                # Create dictionaries for quick lookup
                rgb_dict = {f.stem: f for f in rgb_images}
                depth_dict = {f.stem: f for f in depth_images}
                
                # Find matching images (same image_id)
                common_image_ids = set(rgb_dict.keys()) & set(depth_dict.keys())
                
                for image_id in sorted(common_image_ids):
                    rgb_path = rgb_dict[image_id].absolute()
                    depth_path = depth_dict[image_id].absolute()
                    
                    # Create base row data
                    row_data = {
                        'scene_id': scene_id,
                        'camera_id': camera_type,
                        'image_id': image_id,
                        'rgb_image_path': str(rgb_path),
                        'depth_image_path': str(depth_path)
                    }
                    
                    # Add camera calibration parameters if available
                    if camera_type in camera_info:
                        cam_params = camera_info[camera_type]
                        row_data.update({
                            'cx': cam_params.get('cx', None),
                            'cy': cam_params.get('cy', None),
                            'depth_scale': cam_params.get('depth_scale', None),
                            'fx': cam_params.get('fx', None),
                            'fy': cam_params.get('fy', None),
                            'height': cam_params.get('height', None),
                            'width': cam_params.get('width', None)
                        })
                    else:
                        # Add empty camera parameters if no info available
                        row_data.update({
                            'cx': None,
                            'cy': None,
                            'depth_scale': None,
                            'fx': None,
                            'fy': None,
                            'height': None,
                            'width': None
                        })
                    
                    # Add scene-specific camera position information
                    cam_pos_data = None
                    if camera_type in scene_camera_positions:
                        # Try to find matching camera position data
                        # First try exact image_id match
                        if image_id in scene_camera_positions[camera_type]:
                            cam_pos_data = scene_camera_positions[camera_type][image_id]
                        else:
                            # Try numeric matching (e.g., "00000" -> "0")
                            try:
                                numeric_id = str(int(image_id))
                                if numeric_id in scene_camera_positions[camera_type]:
                                    cam_pos_data = scene_camera_positions[camera_type][numeric_id]
                                else:
                                    # Try zero-padded versions if numeric_id didn't work
                                    for key in scene_camera_positions[camera_type].keys():
                                        try:
                                            if int(key) == int(image_id):
                                                cam_pos_data = scene_camera_positions[camera_type][key]
                                                break
                                        except ValueError:
                                            continue
                            except ValueError:
                                # image_id is not numeric, skip numeric matching
                                pass
                    
                    if cam_pos_data is not None:
                        # Add camera intrinsic matrix (K)
                        cam_K = cam_pos_data.get('cam_K', [None] * 9)
                        for i, k_val in enumerate(cam_K):
                            row_data[f'cam_K_{i}'] = k_val
                        
                        # Add rotation matrix (R_w2c)
                        cam_R = cam_pos_data.get('cam_R_w2c', [None] * 9)
                        for i, r_val in enumerate(cam_R):
                            row_data[f'cam_R_w2c_{i}'] = r_val
                        
                        # Add translation vector (t_w2c)
                        cam_t = cam_pos_data.get('cam_t_w2c', [None] * 3)
                        for i, t_val in enumerate(cam_t):
                            row_data[f'cam_t_w2c_{i}'] = t_val
                        
                        # Add depth scale from scene camera data
                        row_data['scene_depth_scale'] = cam_pos_data.get('depth_scale', None)
                    else:
                        # Add empty camera position data if not available
                        for i in range(9):
                            row_data[f'cam_K_{i}'] = None
                        for i in range(9):
                            row_data[f'cam_R_w2c_{i}'] = None
                        for i in range(3):
                            row_data[f'cam_t_w2c_{i}'] = None
                        row_data['scene_depth_scale'] = None
                    
                    data_rows.append(row_data)
                
                print(f"  {camera_type}: Found {len(common_image_ids)} matching image pairs")
            
            elif rgb_folder.exists():
                print(f"  {camera_type}: RGB folder exists but depth folder missing")
            elif depth_folder.exists():
                print(f"  {camera_type}: Depth folder exists but RGB folder missing")
            else:
                print(f"  {camera_type}: Both RGB and depth folders missing")
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    if not df.empty:
        # Save to CSV
        df.to_csv(output_csv_path, index=False)
        print(f"\nCSV file saved to: {output_csv_path}")
        print(f"Total rows: {len(df)}")
        print(f"Scenes: {df['scene_id'].nunique()}")
        print(f"Cameras: {df['camera_id'].nunique()}")
        print(f"Unique camera types: {df['camera_id'].unique().tolist()}")
        
        # Show which scenes have camera position info
        scenes_with_pos_info = []
        for scene_id in df['scene_id'].unique():
            scene_data = df[df['scene_id'] == scene_id]
            if not scene_data['cam_K_0'].isna().all():
                scenes_with_pos_info.append(scene_id)
        
        print(f"Scenes with camera position info: {len(scenes_with_pos_info)} out of {df['scene_id'].nunique()}")
    else:
        print("No matching RGB-depth image pairs found!")
    
    return df

def preview_dataset_structure(root_directory, max_scenes=3):
    """
    Preview the dataset structure to understand the directory layout.
    
    Args:
        root_directory (str): Path to the root directory
        max_scenes (int): Maximum number of scenes to preview
    """
    root_path = Path(root_directory)
    
    if not root_path.exists():
        print(f"Directory does not exist: {root_directory}")
        return
    
    print(f"Dataset structure preview for: {root_directory}")
    print("=" * 50)
    
    # Get scene directories
    scene_dirs = [d for d in root_path.iterdir() 
                  if d.is_dir() and re.match(r'^\d+$', d.name)]
    scene_dirs.sort(key=lambda x: int(x.name))
    
    for i, scene_dir in enumerate(scene_dirs[:max_scenes]):
        print(f"\nScene: {scene_dir.name}")
        
        # List camera folders
        camera_folders = [d for d in scene_dir.iterdir() if d.is_dir()]
        camera_folders.sort()
        
        for camera_folder in camera_folders:
            # Count images in this folder
            image_files = [f for f in camera_folder.iterdir() 
                          if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.exr']]
            print(f"  {camera_folder.name}: {len(image_files)} images")
            
            # Show first few image names as examples
            if image_files:
                example_names = [f.name for f in sorted(image_files)[:3]]
                print(f"    Examples: {', '.join(example_names)}")
        
        # Check for scene camera files
        camera_files = [f for f in scene_dir.iterdir() 
                       if f.is_file() and f.name.startswith('scene_camera_') and f.suffix == '.json']
        if camera_files:
            print(f"  Camera position files: {[f.name for f in camera_files]}")

def get_camera_info_files():
    """
    Get camera info JSON file paths from user input.
    
    Returns:
        dict: Dictionary mapping camera_id to JSON file path
    """
    camera_types = ['cam1', 'cam2', 'cam3', 'photoneo']
    camera_info_files = {}
    
    print("\nEnter camera info JSON file paths (press Enter to skip if not available):")
    
    for camera_type in camera_types:
        while True:
            file_path = input(f"{camera_type} JSON file path: ").strip()
            
            if not file_path:
                # User pressed Enter - skip this camera
                camera_info_files[camera_type] = None
                print(f"  Skipping {camera_type} camera info")
                break
            
            if Path(file_path).exists():
                camera_info_files[camera_type] = file_path
                print(f"  ✓ {camera_type} camera info file found")
                break
            else:
                print(f"  ✗ File not found: {file_path}")
                retry = input("  Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    camera_info_files[camera_type] = None
                    print(f"  Skipping {camera_type} camera info")
                    break
    
    return camera_info_files

if __name__ == "__main__":
    # Example usage
    root_directory = input("Enter the root directory path: ").strip()
    
    # Get camera info file paths
    camera_info_files = get_camera_info_files()
    
    # First, preview the structure
    print("\nPreviewing dataset structure...")
    preview_dataset_structure(root_directory)
    
    # Generate CSV
    print("\nGenerating CSV...")
    df = generate_dataset_csv(root_directory, camera_info_files)
    
    if not df.empty:
        print("\nFirst few rows of the generated dataset:")
        print(df.head())
        
        # Show some statistics
        print(f"\nDataset Statistics:")
        print(f"Total image pairs: {len(df)}")
        print(f"Number of scenes: {df['scene_id'].nunique()}")
        print(f"Camera types: {', '.join(df['camera_id'].unique())}")
        print(f"Images per camera type:")
        print(df['camera_id'].value_counts())
        
        # Show which cameras have calibration info
        print(f"\nCamera calibration info status:")
        for camera_id in df['camera_id'].unique():
            has_info = not df[df['camera_id'] == camera_id]['cx'].isna().all()
            status = "✓ Available" if has_info else "✗ Missing"
            print(f"  {camera_id}: {status}")
        
        # Show camera position info status
        print(f"\nCamera position info status:")
        for camera_id in df['camera_id'].unique():
            has_pos_info = not df[df['camera_id'] == camera_id]['cam_K_0'].isna().all()
            status = "✓ Available" if has_pos_info else "✗ Missing"
            print(f"  {camera_id}: {status}")