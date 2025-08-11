import json
import os
import numpy as np
import trimesh
import cv2
import csv
from pathlib import Path
from tqdm import tqdm
import glob

class TestDatasetProcessor:
    def __init__(self, dataset_root_path,  output_csv_path):
        self.dataset_root_path = dataset_root_path
        self.output_csv_path = output_csv_path
        
        # CSV headers
        self.headers = [
            'rgb_image_path', 'camera_type', 'scene_id', 'image_index',
            'image_width', 'image_height',
            # Camera intrinsics (K matrix)
            'k11', 'k12', 'k13', 'k21', 'k22', 'k23', 'k31', 'k32', 'k33',
            # Camera extrinsics (R_w2c matrix)
            'r_w2c_11', 'r_w2c_12', 'r_w2c_13', 'r_w2c_21', 'r_w2c_22', 'r_w2c_23', 'r_w2c_31', 'r_w2c_32', 'r_w2c_33',
            # Camera translation (t_w2c vector)
            't_w2c_x', 't_w2c_y', 't_w2c_z',
            # Depth scale
            'depth_scale'
        ]
        
        self.csv_rows = []
    
    def get_scene_camera_filename(self, cam_id):
        """Get the scene camera filename based on cam_id"""
        if isinstance(cam_id, str):
            return f"scene_camera_{cam_id}.json"
        else:
            return f"scene_camera_cam{cam_id}.json"
    
    def get_rgb_folder_name(self, cam_id):
        """Get the RGB folder name based on cam_id"""
        if isinstance(cam_id, str):
            return f"rgb_{cam_id}"
        else:
            return f"rgb_cam{cam_id}"
    
    def get_camera_type_string(self, cam_id):
        """Get the camera type string for CSV output"""
        if isinstance(cam_id, str):
            return cam_id
        else:
            return f"cam{cam_id}"
    
    def load_scene_camera(self, dataset_path, cam_id):
        """Load scene camera parameters"""
        scene_cam_file = os.path.join(dataset_path, self.get_scene_camera_filename(cam_id))
        if os.path.exists(scene_cam_file):
            with open(scene_cam_file, 'r') as f:
                return json.load(f)
        else:
            return None
    
    
    def process_dataset(self):
        """Process the entire dataset and generate CSV"""
        # Find all dataset folders
        dataset_folders = sorted([f for f in os.listdir(self.dataset_root_path) 
                                 if os.path.isdir(os.path.join(self.dataset_root_path, f))])
        
        print(f"Found {len(dataset_folders)} dataset folders to process")
        
        # Process each dataset folder
        for dataset_folder in tqdm(dataset_folders, desc="Processing dataset folders"):
            dataset_path = os.path.join(self.dataset_root_path, dataset_folder)
            self.process_dataset_folder(dataset_path, dataset_folder)
        
        # Write CSV file
        self.write_csv()
        
        return len(self.csv_rows)
    
    def process_dataset_folder(self, dataset_path, dataset_folder):
        """Process a single dataset folder"""
        print(f"\nProcessing dataset folder: {dataset_folder}")
        
        # Camera IDs to process
        camera_ids = [1, 2, 3, "photoneo"]
        
        # First, find all available image indices by checking one camera's scene_camera file
        available_image_indices = []
        for cam_id in camera_ids:
            scene_cameras = self.load_scene_camera(dataset_path, cam_id)
            if scene_cameras is not None:
                available_image_indices = sorted(scene_cameras.keys())
                break
        
        if not available_image_indices:
            print(f"  No valid scene camera files found in {dataset_folder}")
            return
        
        print(f"  Found image indices: {available_image_indices}")
        
        # Process each image index
        for image_index in tqdm(available_image_indices, desc=f"Processing images in {dataset_folder}", leave=False):
            # Process each camera for this image index
            for cam_id in camera_ids:
                # Load camera  data for this camera
                scene_cameras = self.load_scene_camera(dataset_path, cam_id)

                # Check if this image index exists for this camera
                if image_index not in scene_cameras:
                    print(f"    Skipping camera {cam_id} for image {image_index} - image index not found in scene_cameras")
                    continue
                
                
                # Process this specific image-camera combination
                self.process_image(dataset_path, dataset_folder, cam_id, image_index, 
                                scene_cameras)
    
    def process_image(self, dataset_path, dataset_folder, cam_id, image_index, 
                     scene_cameras):
        """Process a single image for a specific camera"""
        try:
            # Get image path
            rgb_folder = self.get_rgb_folder_name(cam_id)
            img_path = os.path.join(dataset_path, rgb_folder, f"{int(image_index):06d}.png")
            
            if not os.path.exists(img_path):
                print(f"      Warning: Image {img_path} not found")
                return
            
            # Read image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"      Warning: Could not read image {img_path}")
                return
            
            height, width = img.shape[:2]
            
            # Get camera parameters for this specific image
            cam_params = scene_cameras[image_index]
            cam_K = np.array(cam_params["cam_K"]).reshape(3, 3)
            
            # Extract camera extrinsics (R_w2c) and translation (t_w2c)
            # These might be stored with different key names, so we'll try common variations
            cam_R_w2c = None
            cam_t_w2c = None
            depth_scale = None
            
            # Try to get camera extrinsics (world to camera transformation)
            if "cam_R_w2c" in cam_params:
                cam_R_w2c = np.array(cam_params["cam_R_w2c"]).reshape(3, 3)
            elif "R_w2c" in cam_params:
                cam_R_w2c = np.array(cam_params["R_w2c"]).reshape(3, 3)
            elif "cam_R" in cam_params:
                cam_R_w2c = np.array(cam_params["cam_R"]).reshape(3, 3)
            else:
                # If not available, use identity matrix as default
                cam_R_w2c = np.eye(3)
            
            # Try to get camera translation (world to camera)
            if "cam_t_w2c" in cam_params:
                cam_t_w2c = np.array(cam_params["cam_t_w2c"])
            elif "t_w2c" in cam_params:
                cam_t_w2c = np.array(cam_params["t_w2c"])
            elif "cam_t" in cam_params:
                cam_t_w2c = np.array(cam_params["cam_t"])
            else:
                # If not available, use zero translation as default
                cam_t_w2c = np.zeros(3)
            
            # Try to get depth scale
            if "depth_scale" in cam_params:
                depth_scale = cam_params["depth_scale"]
            elif "scale" in cam_params:
                depth_scale = cam_params["scale"]
            else:
                # Default depth scale if not available
                depth_scale = 1.0
            
            
            row = [
                img_path,                           # image_path
                self.get_camera_type_string(cam_id), # camera_type
                dataset_folder,                     # scene_id (folder name)
                image_index,                        # image_index
                width, height,                      # image_width, image_height
                # Camera intrinsics (K matrix)
                cam_K[0, 0], cam_K[0, 1], cam_K[0, 2],  # k11, k12, k13
                cam_K[1, 0], cam_K[1, 1], cam_K[1, 2],  # k21, k22, k23
                cam_K[2, 0], cam_K[2, 1], cam_K[2, 2],  # k31, k32, k33
                # Camera extrinsics (R_w2c matrix)
                cam_R_w2c[0, 0], cam_R_w2c[0, 1], cam_R_w2c[0, 2],  # r_w2c_11, r_w2c_12, r_w2c_13
                cam_R_w2c[1, 0], cam_R_w2c[1, 1], cam_R_w2c[1, 2],  # r_w2c_21, r_w2c_22, r_w2c_23
                cam_R_w2c[2, 0], cam_R_w2c[2, 1], cam_R_w2c[2, 2],  # r_w2c_31, r_w2c_32, r_w2c_33
                # Camera translation (t_w2c vector)
                cam_t_w2c[0], cam_t_w2c[1], cam_t_w2c[2],  # t_w2c_x, t_w2c_y, t_w2c_z
                # Depth scale
                depth_scale                         # depth_scale
            ]
                
            self.csv_rows.append(row)
                
        except Exception as e:
            print(f"      Error processing image {image_index} for camera {cam_id}: {e}")
    
    def write_csv(self):
        """Write the CSV file"""
        print(f"\nWriting {len(self.csv_rows)} rows to {self.output_csv_path}")
        
        with open(self.output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.headers)
            writer.writerows(self.csv_rows)
        
        print("CSV file created successfully!")

def read_test_dataset(dataset_root_path,  output_csv_path):
    """
    Main function to read dataset and create CSV
    
    Args:
        dataset_root_path (str): Root folder containing multiple dataset folders
        output_csv_path (str): Output CSV file path
    
    Returns:
        dict: Result with 'success', 'message', and 'total_rows'
    """
    try:
        # Validate paths
        if not os.path.exists(dataset_root_path):
            return {
                'success': False,
                'message': f"Dataset root path '{dataset_root_path}' does not exist.",
                'total_rows': 0
            }
        
        
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Create processor and run
        processor = TestDatasetProcessor(
            dataset_root_path=dataset_root_path,
            output_csv_path=output_csv_path
        )
        
        total_rows = processor.process_dataset()
        
        return {
            'success': True,
            'message': f"Successfully processed dataset. Created {total_rows} entries.",
            'total_rows': total_rows
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f"Error processing dataset: {str(e)}",
            'total_rows': 0
        }