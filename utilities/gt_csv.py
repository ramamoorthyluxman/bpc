import json
import os
import numpy as np
import trimesh
import cv2
import csv
from pathlib import Path
from tqdm import tqdm
import glob

class DatasetProcessor:
    def __init__(self, dataset_root_path, models_path, camera_info_path, output_csv_path):
        self.dataset_root_path = dataset_root_path
        self.models_path = models_path
        self.camera_info_path = camera_info_path
        self.output_csv_path = output_csv_path
        
        # CSV headers
        self.headers = [
            'image_path', 'camera_type', 'scene_id', 'object_id',
            'r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33',
            'tx', 'ty', 'tz', 'polygon_mask', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
            'image_width', 'image_height'
        ]
        
        self.csv_rows = []
    
    def get_camera_filename(self, cam_id):
        """Get the camera filename based on cam_id"""
        if isinstance(cam_id, str):
            return f"camera_{cam_id}.json"
        else:
            return f"camera_cam{cam_id}.json"
    
    def get_scene_camera_filename(self, cam_id):
        """Get the scene camera filename based on cam_id"""
        if isinstance(cam_id, str):
            return f"scene_camera_{cam_id}.json"
        else:
            return f"scene_camera_cam{cam_id}.json"
    
    def get_scene_gt_filename(self, cam_id):
        """Get the scene ground truth filename based on cam_id"""
        if isinstance(cam_id, str):
            return f"scene_gt_{cam_id}.json"
        else:
            return f"scene_gt_cam{cam_id}.json"
    
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
    
    def load_camera_intrinsics(self, cam_id):
        """Load camera intrinsic parameters"""
        cam_file = os.path.join(self.camera_info_path, self.get_camera_filename(cam_id))
        if os.path.exists(cam_file):
            with open(cam_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: Camera intrinsics file {cam_file} not found")
            return None
    
    def load_scene_camera(self, dataset_path, cam_id):
        """Load scene camera parameters"""
        scene_cam_file = os.path.join(dataset_path, self.get_scene_camera_filename(cam_id))
        if os.path.exists(scene_cam_file):
            with open(scene_cam_file, 'r') as f:
                return json.load(f)
        else:
            return None
    
    def load_scene_gt(self, dataset_path, cam_id):
        """Load scene ground truth"""
        scene_gt_file = os.path.join(dataset_path, self.get_scene_gt_filename(cam_id))
        if os.path.exists(scene_gt_file):
            with open(scene_gt_file, 'r') as f:
                return json.load(f)
        else:
            return None
    
    def load_model(self, obj_id):
        """Load 3D model"""
        model_path = os.path.join(self.models_path, f"obj_{obj_id:06d}.ply")
        if os.path.exists(model_path):
            try:
                mesh = trimesh.load(model_path)
                return mesh
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
                return None
        else:
            print(f"Model not found: {model_path}")
            return None
    
    def create_polygon_from_mesh(self, mesh, cam_K, R, t, img_shape, simplify_factor=0.001):
        """Create polygon from 3D mesh projection"""
        try:
            # Get mesh faces and vertices
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
            
            # Transform vertices to camera coordinates
            vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
            camera_coords = np.dot(vertices_homogeneous, np.vstack([
                np.hstack([R, t.reshape(3, 1)]),
                [0, 0, 0, 1]
            ]).T)
            
            # Project to image plane
            projected_points = np.dot(camera_coords[:, :3], cam_K.T)
            projected_points = projected_points[:, :2] / projected_points[:, 2:3]
            
            # Create an empty mask
            mask = np.zeros(img_shape[:2], dtype=np.uint8)
            
            # Filter visible faces (those with all vertices in front of camera)
            visible_faces = []
            for face in faces:
                if all(camera_coords[v, 2] > 0 for v in face):
                    visible_faces.append(face)
            
            # Draw the visible faces on the mask
            for face in visible_faces:
                pts = projected_points[face].astype(np.int32)
                # Check if points are within image boundaries
                if np.all((pts[:, 0] >= 0) & (pts[:, 0] < img_shape[1]) & 
                          (pts[:, 1] >= 0) & (pts[:, 1] < img_shape[0])):
                    cv2.fillPoly(mask, [pts], 255)
            
            # Find contours on the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, None
            
            # Get the largest contour
            max_contour = max(contours, key=cv2.contourArea)
            
            # Simplify the contour
            epsilon = simplify_factor * cv2.arcLength(max_contour, True)
            approx_polygon = cv2.approxPolyDP(max_contour, epsilon, True)
            
            # Convert to the format needed for the annotation
            polygon = approx_polygon.reshape(-1, 2).tolist()
            
            # Make sure the polygon has at least 3 points
            if len(polygon) < 3:
                return None, None
            
            # Calculate bounding box
            x_coords = [p[0] for p in polygon]
            y_coords = [p[1] for p in polygon]
            bbox_x = min(x_coords)
            bbox_y = min(y_coords)
            bbox_w = max(x_coords) - bbox_x
            bbox_h = max(y_coords) - bbox_y
            
            return polygon, (bbox_x, bbox_y, bbox_w, bbox_h)
            
        except Exception as e:
            print(f"Error creating polygon: {e}")
            return None, None
    
    def polygon_to_string(self, polygon):
        """Convert polygon to string format for CSV"""
        if polygon is None:
            return ""
        # Keep nested structure: [[x1,y1],[x2,y2],[x3,y3],...]
        return json.dumps(polygon)
    
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
        
        # Process cameras 1, 2, 3, and photoneo
        camera_ids = [1, 2, 3, "photoneo"]
        
        for cam_id in camera_ids:
            # Check if camera files exist
            scene_gt = self.load_scene_gt(dataset_path, cam_id)
            scene_cameras = self.load_scene_camera(dataset_path, cam_id)
            
            if scene_gt is None or scene_cameras is None:
                print(f"  Skipping camera {cam_id} - missing required files")
                continue
            
            # Load camera intrinsics
            cam_intrinsics = self.load_camera_intrinsics(cam_id)
            if cam_intrinsics is None:
                print(f"  Skipping camera {cam_id} - missing camera intrinsics")
                continue
            
            print(f"  Processing camera {cam_id}...")
            
            # Process each scene
            scene_ids = list(scene_gt.keys())
            for scene_id in tqdm(scene_ids, desc=f"    Scenes for {self.get_camera_type_string(cam_id)}", leave=False):
                self.process_scene(dataset_path, dataset_folder, cam_id, scene_id, 
                                 scene_gt, scene_cameras, cam_intrinsics)
    
    def process_scene(self, dataset_path, dataset_folder, cam_id, scene_id, 
                      scene_gt, scene_cameras, cam_intrinsics):
        """Process a single scene"""
        try:
            # Get image path
            rgb_folder = self.get_rgb_folder_name(cam_id)
            img_path = os.path.join(dataset_path, rgb_folder, f"{int(scene_id):06d}.png")
            
            if not os.path.exists(img_path):
                print(f"    Warning: Image {img_path} not found")
                return
            
            # Read image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                print(f"    Warning: Could not read image {img_path}")
                return
            
            height, width = img.shape[:2]
            
            # Get camera parameters for this scene
            if scene_id not in scene_cameras:
                print(f"    Warning: Camera parameters not found for scene {scene_id}")
                return
            
            cam_params = scene_cameras[scene_id]
            cam_K = np.array(cam_params["cam_K"]).reshape(3, 3)
            
            # Process each object in the scene
            for obj_idx, obj_gt in enumerate(scene_gt[scene_id]):
                obj_id = obj_gt["obj_id"]
                
                # Get object pose
                R = np.array(obj_gt["cam_R_m2c"]).reshape(3, 3)
                t = np.array(obj_gt["cam_t_m2c"])
                
                # Load 3D model
                mesh = self.load_model(obj_id)
                
                # Generate polygon mask
                polygon = None
                bbox = (0, 0, 0, 0)
                
                if mesh is not None:
                    polygon, bbox = self.create_polygon_from_mesh(
                        mesh, cam_K, R, t, (height, width),
                        simplify_factor=0.0005
                    )
                    if bbox is None:
                        bbox = (0, 0, 0, 0)
                
                # Create CSV row
                row = [
                    img_path,                           # image_path
                    self.get_camera_type_string(cam_id), # camera_type
                    scene_id,                           # scene_id
                    obj_id,                             # object_id
                    R[0, 0], R[0, 1], R[0, 2],         # r11, r12, r13
                    R[1, 0], R[1, 1], R[1, 2],         # r21, r22, r23
                    R[2, 0], R[2, 1], R[2, 2],         # r31, r32, r33
                    t[0], t[1], t[2],                   # tx, ty, tz
                    self.polygon_to_string(polygon),    # polygon_mask
                    bbox[0], bbox[1], bbox[2], bbox[3], # bbox_x, bbox_y, bbox_w, bbox_h
                    width, height                       # image_width, image_height
                ]
                
                self.csv_rows.append(row)
                
        except Exception as e:
            print(f"    Error processing scene {scene_id}: {e}")
    
    def write_csv(self):
        """Write the CSV file"""
        print(f"\nWriting {len(self.csv_rows)} rows to {self.output_csv_path}")
        
        with open(self.output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.headers)
            writer.writerows(self.csv_rows)
        
        print("CSV file created successfully!")

def load_polygon_from_csv_string(polygon_string):
    """Helper function to load polygon from CSV string back to coordinate list"""
    if not polygon_string:
        return None
    
    try:
        # Polygon is already in nested format: [[x1,y1],[x2,y2],[x3,y3],...]
        polygon = json.loads(polygon_string)
        return polygon
    except:
        return None

def main():
    # Configuration
    dataset_root_path = "/home/rama/bpc_ws/bpc/ipd/val"  # Root folder containing multiple dataset folders
    models_path = "/home/rama/bpc_ws/bpc/ipd/models"           # 3D models folder
    camera_info_path = "/home/rama/bpc_ws/bpc/ipd"             # Path containing camera_cam*.json files
    output_csv_path = "master_dataset_with_polygons.csv"       # Output CSV file
    
    # Validate paths
    if not os.path.exists(dataset_root_path):
        print(f"Error: Dataset root path '{dataset_root_path}' does not exist.")
        return
    
    if not os.path.exists(models_path):
        print(f"Error: Models path '{models_path}' does not exist.")
        return
    
    if not os.path.exists(camera_info_path):
        print(f"Error: Camera info path '{camera_info_path}' does not exist.")
        return
    
    # Create processor and run
    try:
        processor = DatasetProcessor(
            dataset_root_path=dataset_root_path,
            models_path=models_path,
            camera_info_path=camera_info_path,
            output_csv_path=output_csv_path
        )
        
        total_rows = processor.process_dataset()
        
        print(f"\nSummary:")
        print(f"- Total entries created: {total_rows}")
        print(f"- Output file: {output_csv_path}")
        print(f"- Columns: {len(processor.headers)}")
        print(f"- Headers: {processor.headers}")
        
        # Show example of how to load polygon back
        print(f"\nTo load polygon from CSV string, use:")
        print(f"polygon = load_polygon_from_csv_string(row['polygon_mask'])")
        print(f"# Returns: [[x1, y1], [x2, y2], [x3, y3], ...]")
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()