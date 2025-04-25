import json
import os
import numpy as np
import trimesh
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm

# Define paths
dataset_path = "/home/rama/bpc_ws/bpc/ipd/val/000000"  # Your dataset folder
output_path = "/home/rama/bpc_ws/bpc/datasets/ipd_val/000000"  # Output folder
models_path = "/home/rama/bpc_ws/bpc/ipd/models"  # 3D models folder

# Create output directories
os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
os.makedirs(os.path.join(output_path, "annotations"), exist_ok=True)

# Load camera intrinsic parameters
def load_camera_intrinsics(cam_id):
    with open(os.path.join(f"/home/rama/bpc_ws/bpc/ipd/", f"camera_cam{cam_id}.json"), 'r') as f:
        cam_data = json.load(f)
    return cam_data

# Load scene camera parameters
def load_scene_camera(cam_id):
    with open(os.path.join(dataset_path, f"scene_camera_cam{cam_id}.json"), 'r') as f:
        scene_cam_data = json.load(f)
    return scene_cam_data

# Load scene ground truth
def load_scene_gt(cam_id):
    with open(os.path.join(dataset_path, f"scene_gt_cam{cam_id}.json"), 'r') as f:
        scene_gt_data = json.load(f)
    return scene_gt_data

# Load scene ground truth info
def load_scene_gt_info(cam_id):
    with open(os.path.join(dataset_path, f"scene_gt_info_cam{cam_id}.json"), 'r') as f:
        scene_gt_info_data = json.load(f)
    return scene_gt_info_data

# Load 3D model
def load_model(obj_id):
    model_path = os.path.join(models_path, f"obj_{obj_id:06d}.ply")
    mesh = trimesh.load(model_path)
    return mesh

# Project 3D model to 2D image
def project_model_to_2d(mesh, K, R, t):
    # Get mesh vertices
    vertices = np.array(mesh.vertices)
    
    # Convert to homogeneous coordinates
    homogeneous_vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    
    # Transform to camera coordinates: R*X + t
    camera_coords = np.dot(R, vertices.T).T + t
    
    # Project to image plane
    image_coords = np.dot(K, camera_coords.T).T
    
    # Convert to pixel coordinates
    pixel_coords = image_coords[:, :2] / image_coords[:, 2:3]
    
    return pixel_coords

# Create polygon from projected points
def create_polygon(pixel_coords, img_shape):
    # Filter points inside image boundaries
    valid_points = []
    for x, y in pixel_coords:
        if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
            valid_points.append([int(x), int(y)])
    
    if len(valid_points) < 3:
        return None
    
    # Compute convex hull
    hull = cv2.convexHull(np.array(valid_points))
    
    # Simplify polygon if needed
    epsilon = 0.001 * cv2.arcLength(hull, True)
    polygon = cv2.approxPolyDP(hull, epsilon, True)
    
    return polygon.reshape(-1, 2).tolist()

# Process all cameras
for cam_id in [1, 2, 3]:
    print(f"Processing camera {cam_id}...")
    
    # Load camera data
    cam_intrinsics = load_camera_intrinsics(cam_id)
    scene_cameras = load_scene_camera(cam_id)
    scene_gt = load_scene_gt(cam_id)
    scene_gt_info = load_scene_gt_info(cam_id)
    
    # Create the intrinsic matrix
    K = np.array([
        [cam_intrinsics["fx"], 0, cam_intrinsics["cx"]],
        [0, cam_intrinsics["fy"], cam_intrinsics["cy"]],
        [0, 0, 1]
    ])
    
    # Process each scene
    for scene_id in tqdm(scene_gt.keys()):
        # Get camera parameters for this scene
        cam_params = scene_cameras[scene_id]
        cam_K = np.array(cam_params["cam_K"]).reshape(3, 3)
        cam_R_w2c = np.array(cam_params["cam_R_w2c"]).reshape(3, 3)
        cam_t_w2c = np.array(cam_params["cam_t_w2c"])
        
        # Copy the RGB image to output folder
        src_img_path = os.path.join(dataset_path, f"rgb_cam{cam_id}", f"{int(scene_id):06d}.png")
        dst_img_path = os.path.join(output_path, "images", f"{int(scene_id):06d}_cam{cam_id}.jpg")
        
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"Warning: Image {src_img_path} not found")
            continue
        
        # Read the image to get dimensions
        img = cv2.imread(src_img_path)
        if img is None:
            print(f"Warning: Could not read image {src_img_path}")
            continue
        
        height, width = img.shape[:2]
        
        # Prepare annotation for this image
        annotation = {
            "image_path": f"images/{int(scene_id):06d}_cam{cam_id}.jpg",
            "height": height,
            "width": width,
            "masks": []
        }
        
        # Process each object in the scene
        for obj_idx, obj_gt in enumerate(scene_gt[scene_id]):
            obj_id = obj_gt["obj_id"]
            
            # Get object pose
            R = np.array(obj_gt["cam_R_m2c"]).reshape(3, 3)
            t = np.array(obj_gt["cam_t_m2c"])
            
            # Load 3D model
            try:
                mesh = load_model(obj_id)
            except FileNotFoundError:
                print(f"Warning: Model obj_{obj_id:06d}.ply not found")
                continue
            
            # Project 3D model to 2D
            pixel_coords = project_model_to_2d(mesh, cam_K, R, t)
            
            # Create polygon
            polygon = create_polygon(pixel_coords, (height, width))
            
            if polygon:
                # Add to annotation
                annotation["masks"].append({
                    "label": f"obj_{obj_id:06d}",
                    "points": polygon
                })
        
        # Save annotation
        with open(os.path.join(output_path, "annotations", f"{int(scene_id):06d}_cam{cam_id}.json"), 'w') as f:
            json.dump(annotation, f, indent=4)

print("Dataset creation completed!")