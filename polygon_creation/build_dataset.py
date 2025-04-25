import json
import os
import numpy as np
import trimesh
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm
from scipy.spatial import ConvexHull

# Define paths
dataset_path = "/home/rama/bpc_ws/bpc/ipd/val/000001"  # Your dataset folder
output_path = "/home/rama/bpc_ws/bpc/datasets/ipd_val/000001"  # Output folder
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

# Render model silhouette
def render_silhouette(mesh, cam_K, R, t, img_shape):
    # Create a scene with the mesh
    scene = trimesh.Scene()
    
    # Apply the transformation to the mesh
    transformed_mesh = mesh.copy()
    transformed_mesh.apply_transform(np.vstack([
        np.hstack([R, t.reshape(3, 1)]),
        [0, 0, 0, 1]
    ]))
    
    scene.add_geometry(transformed_mesh)
    
    # Configure the camera
    fx, fy = cam_K[0, 0], cam_K[1, 1]
    cx, cy = cam_K[0, 2], cam_K[1, 2]
    
    # Create a camera with a resolution matching the image
    camera = trimesh.scene.Camera(
        resolution=(img_shape[1], img_shape[0]),
        focal=(fx, fy),
        fov=None,
        center=(cx, cy)
    )
    
    # Set the camera to the scene
    scene.camera = camera
    
    # Render the silhouette
    silhouette = scene.save_image(resolution=(img_shape[1], img_shape[0]), visible=True)
    
    # Convert to binary mask
    silhouette_gray = cv2.cvtColor(silhouette, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(silhouette_gray, 1, 255, cv2.THRESH_BINARY)
    
    return mask

# Create detailed polygon from silhouette
def create_detailed_polygon(mask, simplify_factor=0.001):
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour
    max_contour = max(contours, key=cv2.contourArea)
    
    # Simplify the contour (adjust epsilon for more or less detail)
    epsilon = simplify_factor * cv2.arcLength(max_contour, True)
    approx_polygon = cv2.approxPolyDP(max_contour, epsilon, True)
    
    # Convert to the format needed for the annotation
    polygon = approx_polygon.reshape(-1, 2).tolist()
    
    # Make sure the polygon has at least 3 points
    if len(polygon) < 3:
        return None
        
    return polygon

# Alternative approach using a depth buffer for occlusion handling
def create_polygon_from_mesh(mesh, cam_K, R, t, img_shape, simplify_factor=0.001):
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
        if all(0 <= p[0] < img_shape[1] and 0 <= p[1] < img_shape[0] for p in pts):
            cv2.fillPoly(mask, [pts], 255)
    
    # Find contours on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour
    max_contour = max(contours, key=cv2.contourArea)
    
    # Simplify the contour
    epsilon = simplify_factor * cv2.arcLength(max_contour, True)
    approx_polygon = cv2.approxPolyDP(max_contour, epsilon, True)
    
    # Convert to the format needed for the annotation
    polygon = approx_polygon.reshape(-1, 2).tolist()
    
    # Make sure the polygon has at least 3 points
    if len(polygon) < 3:
        return None
        
    return polygon

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
            
            # Try to create a detailed polygon
            try:
                # Method 1: Try to render the silhouette
                # polygon = create_detailed_polygon(
                #     render_silhouette(mesh, cam_K, R, t, (height, width)),
                #     simplify_factor=0.0005  # Adjust for more detail
                # )
                
                # Method 2: Alternative approach using depth buffer
                polygon = create_polygon_from_mesh(
                    mesh, cam_K, R, t, (height, width),
                    simplify_factor=0.0005  # Adjust for more detail
                )
                
                if polygon:
                    # Add to annotation
                    annotation["masks"].append({
                        "label": f"obj_{obj_id:06d}",
                        "points": polygon
                    })
                else:
                    print(f"Warning: Could not create polygon for object {obj_id} in scene {scene_id}")
                    
            except Exception as e:
                print(f"Error processing object {obj_id} in scene {scene_id}: {e}")
        
        # Save annotation
        with open(os.path.join(output_path, "annotations", f"{int(scene_id):06d}_cam{cam_id}.json"), 'w') as f:
            json.dump(annotation, f, indent=4)

print("Dataset creation completed!")