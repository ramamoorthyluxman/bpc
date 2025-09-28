import os
import sys
from unittest import result
import yaml 
import cv2
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import json
from collections import defaultdict
import numpy as np
import csv
import ast
import torch
from functools import partial
import threading


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pcl_builder')))
from create_pointcloud_gpu_accelerated import build_pcl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SuperGluePretrainedNetwork')))
from superglue import SuperGlueMatcher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'maskRCNN')))
from inference import infer


from update_detection_json import add_3d_centers_to_json, find_valid_neighbor_pixel, transform_3d_point,  transform_pose_to_world
from pose_estimator import compute_rigid_transform
from pxl_to_point import pixel_to_point_normal



def process_single_mask_rcnn_cpu_only(args):
    """Process a single image with MaskRCNN - CPU only preprocessing"""
    cam_row, config, meta_data_path, i = args
    
    # Load images (CPU only)
    cam_row["rgb_img"] = cv2.imread(cam_row["rgb_image_path"])
    
    # Find depth image path
    rgb_dir = os.path.dirname(cam_row["rgb_image_path"])
    dataset_dir = os.path.dirname(rgb_dir)
    camera_type = cam_row["camera_type"]
    
    if camera_type == "photoneo":
        depth_dir = os.path.join(dataset_dir, "depth_photoneo")
    elif camera_type.startswith("cam"):
        depth_dir = os.path.join(dataset_dir, f"depth_{camera_type}")
    else:
        depth_dir = os.path.join(dataset_dir, f"depth_cam{camera_type}")
        if not os.path.exists(depth_dir):
            depth_dir = os.path.join(dataset_dir, "depth")
    
    image_filename = os.path.basename(cam_row["rgb_image_path"])
    depth_path = os.path.join(depth_dir, image_filename)
    cam_row["depth_image_path"] = depth_path
    cam_row["depth_img"] = cv2.imread(cam_row["depth_image_path"])
    
    # Prepare for GPU inference (return preprocessed data)
    maskrcnn_model_path = config['maskrcnn_model_path']
    maskrcnn_output_path = os.path.join(meta_data_path, f"{cam_row['scene_id']}_{cam_row['image_index']}_{i}")
    maskrcnn_category_txt_path = config['maskrcnn_category_txt_path']
    maskrcnn_confidence_threshold = config['maskrcnn_confidence_threshold']
    
    return i, cam_row, {
        'model_path': maskrcnn_model_path,
        'output_path': maskrcnn_output_path,
        'category_path': maskrcnn_category_txt_path,
        'confidence_threshold': maskrcnn_confidence_threshold,
        'visualize_and_save': config['maskrcnn_visualization_and_save']
    }


def process_maskrcnn_gpu_batch(preprocessed_data_list, max_batch_size=4):
    """Process MaskRCNN inference in batches on GPU - single threaded"""
    results = {}
    
    # Process in smaller batches to avoid GPU memory issues
    for batch_start in range(0, len(preprocessed_data_list), max_batch_size):
        batch = preprocessed_data_list[batch_start:batch_start + max_batch_size]
        
        for i, cam_row, inference_params in batch:
            try:
                # Run inference one at a time to avoid GPU conflicts
                detection_json = infer(
                    cam_row["rgb_image_path"], 
                    inference_params['model_path'],
                    inference_params['output_path'],  
                    inference_params['category_path'], 
                    inference_params['confidence_threshold'], 
                    visualize_and_save=inference_params['visualize_and_save']
                )
                
                cam_row["detection_json"] = detection_json
                
                # Add 3D centers (CPU task)
                cam_row["detection_json"] = add_3d_centers_to_json(
                    cam_row=cam_row,
                    json_data=cam_row["detection_json"],
                    depth_path=cam_row["depth_image_path"],
                    search_radius=5
                )
                
                results[i] = cam_row
                
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                # Create empty result to maintain order
                cam_row["detection_json"] = {"results": [{"masks": []}]}
                results[i] = cam_row
        
        # Force GPU memory cleanup between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


# Global matcher pool for threading
_matcher_pool = []
_matcher_lock = threading.Lock()
_pool_initialized = False


def initialize_matcher_pool(pool_size=3):
    """Initialize a pool of SuperGlue matchers for threading"""
    global _matcher_pool, _pool_initialized
    if not _pool_initialized:
        print(f"Initializing {pool_size} SuperGlue matchers...")
        for i in range(pool_size):
            try:
                # Set different CUDA device for each matcher if multiple GPUs available
                if torch.cuda.device_count() > 1:
                    device_id = i % torch.cuda.device_count()
                    with torch.cuda.device(device_id):
                        matcher = SuperGlueMatcher()
                else:
                    matcher = SuperGlueMatcher()
                
                # Ensure matcher is in eval mode and gradients are disabled
                if hasattr(matcher, 'superglue_model'):
                    matcher.superglue_model.eval()
                    for param in matcher.superglue_model.parameters():
                        param.requires_grad = False
                
                _matcher_pool.append(matcher)
                print(f"Matcher {i+1}/{pool_size} initialized")
            except Exception as e:
                print(f"Failed to initialize matcher {i}: {e}")
        _pool_initialized = True
        print(f"Matcher pool initialized with {len(_matcher_pool)} matchers")


def get_matcher():
    """Get a matcher from the pool with timeout and fallback"""
    global _matcher_pool, _matcher_lock
    start_time = time.time()
    
    while time.time() - start_time < 5.0:  # 5 second timeout
        with _matcher_lock:
            if _matcher_pool:
                return _matcher_pool.pop()
        time.sleep(0.01)
    
    # Fallback: create a fresh matcher if pool is empty
    print("Pool exhausted, creating fresh matcher...")
    try:
        fresh_matcher = SuperGlueMatcher()
        if hasattr(fresh_matcher, 'superglue_model'):
            fresh_matcher.superglue_model.eval()
            for param in fresh_matcher.superglue_model.parameters():
                param.requires_grad = False
        return fresh_matcher
    except Exception as e:
        print(f"Failed to create fresh matcher: {e}")
        raise RuntimeError("Cannot get SuperGlue matcher")


def return_matcher(matcher):
    """Return a matcher to the pool with cleanup"""
    global _matcher_pool, _matcher_lock
    
    try:
        # Clean up any remaining gradients or cached tensors
        if hasattr(matcher, 'superglue_model'):
            matcher.superglue_model.eval()
            for param in matcher.superglue_model.parameters():
                param.requires_grad = False
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with _matcher_lock:
            _matcher_pool.append(matcher)
            
    except Exception as e:
        print(f"Error returning matcher to pool: {e}")
        # Don't return faulty matcher to pool


def compare_masked_images_optimized(test_img, test_polygon_mask, camera_id, object_id, ref_dataset_grouped):
    """Optimized version with matcher pooling and proper gradient handling"""
    test_mask = np.zeros(test_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(test_mask, [np.array(test_polygon_mask, dtype=np.int32)], 255)
    masked_test = cv2.bitwise_and(test_img, test_img, mask=test_mask)
    
    # Fast lookup using pre-grouped dataset
    key = f"{camera_id}_{object_id}"
    matching_refs = ref_dataset_grouped.get(key, [])
    
    if not matching_refs:
        return None, None, None, None, None, None, None
    
    # Get matcher from pool
    matcher = get_matcher()
    
    try:
        # Try each reference image with proper gradient handling
        for row in matching_refs:
            ref_img = cv2.imread(row['image_path'])
            if ref_img is None:
                continue
                
            ref_polygon = ast.literal_eval(row['polygon_mask'])
            ref_mask = np.zeros(ref_img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(ref_mask, [np.array(ref_polygon, dtype=np.int32)], 255)
            masked_ref = cv2.bitwise_and(ref_img, ref_img, mask=ref_mask)
            
            # Use no_grad context to prevent gradient accumulation
            with torch.no_grad():
                num_matches, confidence, viz_image, h_mat, matched_points0, matched_points1 = matcher.superglue(masked_test, masked_ref)
            
            # Clear any remaining gradients and GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if num_matches is not None and num_matches > 4:
                return num_matches, confidence, viz_image, h_mat, row, matched_points0, matched_points1
    
    except Exception as e:
        print(f"SuperGlue error: {e}")
        # Force cleanup on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    finally:
        # Always return matcher to pool
        return_matcher(matcher)
    
    return None, None, None, None, None, None, None


def process_single_feature_matching_fast(args):
    """Fast feature matching with pooled matchers"""
    detection_info, scene_data, ref_dataset_grouped, meta_data_path = args
    
    detection, consolidated_idx, j = detection_info
    row_num = detection[4]
    mask_idx = detection[3]
    camera_id = detection[1]
    object_id = int(detection[0].split('_')[1])
    mask = scene_data[row_num]["detection_json"]["results"][0]["masks"][mask_idx]["points"]
    image_id = scene_data[row_num]["image_index"]
    scene_id = scene_data[row_num]["scene_id"]
    
    # Quick feature matching with pooled matcher and grouped dataset
    num_matches, confidence, viz_image, h_mat, ref_row, matched_points0, matched_points1 = compare_masked_images_optimized(
        test_img=scene_data[row_num]["rgb_img"],
        test_polygon_mask=mask,
        camera_id=camera_id,
        object_id=object_id,
        ref_dataset_grouped=ref_dataset_grouped
    )
    
    if num_matches is None:
        return None, consolidated_idx, j, row_num, mask_idx, camera_id, object_id, None, None, None, None, None
    
    # Fast 3D point processing
    K = np.array([float(scene_data[row_num][f'k{n//3+1}{n%3+1}']) for n in range(9)]).reshape(3, 3)
    depth_scale = float(scene_data[row_num]['depth_scale'])
    depth_map0 = scene_data[row_num]['depth_image_path']
    depth_map1 = ref_row['image_path'].replace('rgb', 'depth')
    
    matched_points0_3d = []
    matched_points1_3d = []
    
    # Vectorized processing for speed
    for n in range(len(matched_points0)):
        matched_point0_3d, _ = find_valid_neighbor_pixel(
            int(matched_points0[n][0]), int(matched_points0[n][1]), depth_map0, K, depth_scale, 8
        )
        matched_point1_3d, _ = find_valid_neighbor_pixel(
            int(matched_points1[n][0]), int(matched_points1[n][1]), depth_map1, K, depth_scale, 8
        )
        
        if matched_point0_3d is not None and matched_point1_3d is not None:
            matched_points0_3d.append(matched_point0_3d)
            matched_points1_3d.append(matched_point1_3d)
    
    if len(matched_points0_3d) > 3 and len(matched_points0_3d) == len(matched_points1_3d):
        save_match_img_path = os.path.join(meta_data_path, f"matches_scene_{scene_id}_image_{image_id}_cam_{camera_id}_obj_{object_id}_mask_{mask_idx}.png")
        
        os.makedirs(meta_data_path, exist_ok=True)
        cv2.putText(viz_image, f"Scene: {scene_id}, Image: {image_id}, Cam: {camera_id}, Obj: {object_id}, Mask: {mask_idx}", 
                   (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
        cv2.imwrite(save_match_img_path, viz_image)
        
        result = {
            'scene_cam_row_num': row_num,
            'camera_id': camera_id,
            'mask_idx': mask_idx,
            'object_idx': object_id,
            'ref_row': ref_row,
            'h_mat': h_mat,
            'confidence': confidence,
            'num_matches': num_matches,
            'matched_points0': matched_points0,
            'matched_points1': matched_points1,
            'matched_points0_3d': matched_points0_3d,
            'matched_points1_3d': matched_points1_3d,
            'viz_image': viz_image,
            'consolidated_idx': consolidated_idx,
            'detection_j': j
        }
        return result, consolidated_idx, j, row_num, mask_idx, camera_id, object_id, ref_row, h_mat, confidence, matched_points0, matched_points1
    
    return None, consolidated_idx, j, row_num, mask_idx, camera_id, object_id, ref_row, h_mat, confidence, matched_points0, matched_points1


class process_scene_data:
    def __init__(self, scene_data, max_workers=None):
        self.scene_data = scene_data
        self.config = self.load_config("config.yaml")
        self.meta_data_path = self.config["meta_data_folder"]
        self.mesh_dir = self.config["mesh_dir"]

        # Optimize worker count for GPU memory and CPU cores
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(scene_data), 8)  # Increased back for better CPU utilization
        self.max_workers = max_workers

        self.display_results = {}
        self.display_results["images"] = []
        self.display_results["polygons"] = []
        self.display_results["camera_ids"] = []
        self.display_results["detected_objects"] = []
        self.display_results["nb_maskrcnn_detections"] = 0
        self.display_results["results_summary"] = []
        self.display_results["feature_matching_images"] = []
        
        # Load and pre-process reference dataset
        ref_csv_path = self.config["ref_csv_path"]
        with open(ref_csv_path, 'r') as f:
            self.ref_dataset = list(csv.DictReader(f))
        
        # Pre-group reference dataset by camera_type and object_id for faster lookups
        self.ref_dataset_grouped = defaultdict(list)
        for row in self.ref_dataset:
            key = f"{row['camera_type']}_{row['object_id']}"
            self.ref_dataset_grouped[key].append(row)

        self.detections = []
        self.detections_cluster_distance_threshold = 60
        self.consolidated_detections = []
        self.detections_homographies = []

        # Initialize pose predictor once in main thread
        self.pose_predictor = None
        self._pose_predictor_lock = threading.Lock()

    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def mask_objects(self):
        """Hybrid approach: CPU preprocessing + Sequential GPU processing"""
        print("Starting MaskRCNN processing...")
        start_time = time.time()
        
        # Step 1: Parallel CPU preprocessing
        args_list = [(cam_row, self.config, self.meta_data_path, i) for i, cam_row in enumerate(self.scene_data)]
        
        print("Preprocessing images in parallel...")
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            preprocessed_results = list(executor.map(process_single_mask_rcnn_cpu_only, args_list))
        
        print("Running GPU inference sequentially...")
        # Step 2: Sequential GPU processing to avoid CUDA conflicts
        gpu_results = process_maskrcnn_gpu_batch(preprocessed_results, max_batch_size=2)
        
        # Step 3: Update scene_data and display_results
        for i in range(len(self.scene_data)):
            if i in gpu_results:
                self.scene_data[i] = gpu_results[i]
                self.display_results["images"].append(gpu_results[i]["rgb_img"])
                self.display_results["polygons"].append(gpu_results[i]["detection_json"]["results"][0]["masks"])
                self.display_results["camera_ids"].append(gpu_results[i]["camera_type"])
        
        print(f"MaskRCNN processing completed in {time.time() - start_time:.2f} seconds")
        return self.scene_data

    def consolidate_detections(self):
        """Optimized detection consolidation"""
        print("Consolidating detections...")
        start_time = time.time()
        
        for row_num, row in enumerate(self.scene_data):
            camera_id = row['camera_type']
            detection_data = row['detection_json']
            
            for mask_idx, mask in enumerate(detection_data['results'][0]['masks']):
                object_id = mask['label']
                location_world = mask['object_center_3d_world']
                self.detections.append([object_id, camera_id, location_world, mask_idx, row_num])

        # Vectorized clustering using NumPy
        by_object = defaultdict(list)
        for idx, detection in enumerate(self.detections):
            by_object[detection[0]].append(idx)
        
        for indices in by_object.values():
            used = set()
            locations = np.array([self.detections[idx][2] for idx in indices])
            
            for i, idx1 in enumerate(indices):
                if idx1 in used:
                    continue
                    
                cluster = [idx1]
                used.add(idx1)
                
                # Vectorized distance computation
                distances = np.linalg.norm(locations - locations[i], axis=1)
                nearby_indices = np.where(distances <= self.detections_cluster_distance_threshold)[0]
                
                for nearby_idx in nearby_indices:
                    actual_idx = indices[nearby_idx]
                    if actual_idx not in used:
                        cluster.append(actual_idx)
                        used.add(actual_idx)
                
                self.consolidated_detections.append(cluster)
        
        print(f"Detection consolidation completed in {time.time() - start_time:.2f} seconds")

    



    def do_feature_matchings(self):
        
        for i in range(0,len(self.consolidated_detections)):
            # print("Detections shape: ", len(self.consolidated_detections),len(self.consolidated_detections[i]))
            
            self.display_results["nb_maskrcnn_detections"] = self.consolidated_detections[i][0]

            for j in range(0, len(self.consolidated_detections[i])):

                detection = self.detections[self.consolidated_detections[i][j]]
                row_num = detection[4]
                mask_idx = detection[3]
                camera_id = detection[1]
                object_id = int(detection[0].split('_')[1])
                mask = self.scene_data[row_num]["detection_json"]["results"][0]["masks"][mask_idx]["points"]
                image_id = self.scene_data[row_num]["image_index"]
                scene_id = self.scene_data[row_num]["scene_id"]

                # print("scene_id: ", scene_id)
                # print("image_id: ", image_id)
                # print("camera_id: ", camera_id)
                # print("mask ID: ", mask_idx )
                # print("object_id: ", object_id)
                self.display_results["detected_objects"].append(object_id)
                
                num_matches, confidence, viz_image, h_mat, ref_row, matched_points0, matched_points1 = self.compare_masked_images(test_img = self.scene_data[row_num]["rgb_img"], 
                                                    test_polygon_mask = mask,
                                                    camera_id = camera_id, 
                                                    object_id = object_id)
                
                if num_matches is not None: # refer self.compare_masked_images - it explains the None here

                    K = np.array([float(self.scene_data[row_num][f'k{n//3+1}{n%3+1}']) for n in range(9)]).reshape(3, 3)
                    # Extract depth scale
                    depth_scale = float(self.scene_data[row_num]['depth_scale'])

                    depth_map0 = self.scene_data[row_num]['depth_image_path']
            
                    depth_map1 = ref_row['image_path'].replace('rgb', 'depth')
            
                    matched_points0_3d = []
                    matched_points1_3d = []


                    for n in range(0, len(matched_points0)): 
                        
                        matched_point0_3d, used_pixel = find_valid_neighbor_pixel(
                            int(matched_points0[n][0]), int(matched_points0[n][1]), depth_map0, K, depth_scale, 8
                        )
                        matched_point1_3d, used_pixel = find_valid_neighbor_pixel(
                            int(matched_points1[n][0]), int(matched_points1[n][1]), depth_map1, K, depth_scale, 8
                        )
                        
                        if matched_point0_3d is not None and matched_point1_3d is not None:
                            # matched_point0_3d = transform_3d_point(matched_point0_3d, self.scene_data[row_num])
                            # matched_point1_3d = transform_3d_point(matched_point1_3d, self.scene_data[row_num])
                            
                            matched_points0_3d.append(matched_point0_3d)
                            matched_points1_3d.append(matched_point1_3d)    


                    if len(matched_points0_3d)>3 and len(matched_points0_3d) == len(matched_points1_3d):

                        save_match_img_path = os.path.join(self.meta_data_path, "matches_scene_" + str(scene_id)  + "_image_" + str(image_id) + "_cam_" + camera_id + "_obj_" + str(object_id) + "_mask_" + str(mask_idx) + ".png")
                        
                        os.makedirs(self.meta_data_path, exist_ok=True)
                        cv2.putText(viz_image, f"Scene: {scene_id}, Image: {image_id}, Cam: {camera_id}, Obj: {object_id}, Mask: {mask_idx}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
                        cv2.imwrite(save_match_img_path, viz_image)

                        result = {'scene_cam_row_num': row_num,
                                'camera_id': camera_id,
                                'mask_idx': mask_idx,
                                'object_idx': object_id,
                                'ref_row': ref_row,
                                'h_mat': h_mat,
                                'confidence': confidence,
                                # 'num_matches': num_matches,
                                'num_matches': 0,
                                'matched_points0': matched_points0,
                                'matched_points1': matched_points1,
                                'matched_points0_3d' : matched_points0_3d,
                                'matched_points1_3d' : matched_points1_3d,
                                'viz_image': viz_image}
                        self.detections_homographies.append(result)
                        self.display_results["feature_matching_images"].append(viz_image)
                        if self.compute_6d_poses(result):
                            break


                if j == len(self.consolidated_detections[i])-1 and len(self.detections_homographies) < i+1:
                    result = {
                            'scene_cam_row_num': row_num,
                            'camera_id': camera_id,
                            'mask_idx': mask_idx,
                            'object_idx': object_id,
                            'ref_row': ref_row,
                            'h_mat': h_mat,
                            'confidence': confidence,
                            'num_matches': 0,
                            'matched_points0': matched_points0,
                            'matched_points1': matched_points1,
                            'matched_points0_3d' : None,
                            'matched_points1_3d' : None
                            }
                        
                    self.detections_homographies.append(result)
                    self.display_results["feature_matching_images"].append(viz_image)
                    self.compute_6d_poses(result)    
                

    def get_rotation_matrix(self, normal_vector, angle_degrees):
        """Get rotation matrix from normal vector and angle."""
        # Normalize normal vector
        n = normal_vector / np.linalg.norm(normal_vector)
        
        # Convert angle to radians
        theta = np.radians(angle_degrees)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        # Rodrigues' rotation formula
        K = np.array([[0, -n[2], n[1]], 
                    [n[2], 0, -n[0]], 
                    [-n[1], n[0], 0]])
        
        R = np.eye(3) + sin_t * K + (1 - cos_t) * np.dot(K, K)
        return R

    def compute_6d_poses(self, detection_result):
        result = detection_result
        
        test_center_3d_world = self.scene_data[result['scene_cam_row_num']]['detection_json']['results'][0]['masks'][result['mask_idx']]['object_center_3d_world']

        row_num = result['scene_cam_row_num']   

        R_test_cam = None
        T_test_cam = None
        rmse = 0.5
        camera_id = self.scene_data[result['scene_cam_row_num']]['camera_type']
        object_id = result['object_idx']
        
        if result['num_matches'] == 0:
            mesh_path = self.mesh_dir + '/obj_' + str(f'{result["object_idx"]: 06d}') + '.ply'
            image_size = (int(self.scene_data[row_num]['image_width']), int(self.scene_data[row_num]['image_height']))  
            polygon_mask_coordinates = self.scene_data[result['scene_cam_row_num']]['detection_json']['results'][0]['masks'][result['mask_idx']]['points']
            image = self.scene_data[result['scene_cam_row_num']]["rgb_img"]
            height, width = image.shape[:2]
            mask_image = np.zeros((height, width), dtype=np.uint8)
            points = np.array(polygon_mask_coordinates, dtype=np.int32)
            cv2.fillPoly(mask_image, [points], 255)

            m = cv2.moments(mask_image)
            cx = m['m10'] / m['m00']
            cy = m['m01'] / m['m00']
            angle = 0.5 * np.arctan2(2 * m['mu11'], m['mu20'] - m['mu02']) * 180 / np.pi

            k_matrix = np.array([float(self.scene_data[row_num][f'k{n//3+1}{n%3+1}']) for n in range(9)]).reshape(3, 3)
            depth_scale = float(self.scene_data[row_num]['depth_scale'])

            normal_vector = pixel_to_point_normal(int(cx), int(cy), self.scene_data[row_num]['depth_image_path'], k_matrix = k_matrix, depth_scale=depth_scale)

            R_test_cam = self.get_rotation_matrix(normal_vector, angle)
            T_test_cam = self.scene_data[result['scene_cam_row_num']]['detection_json']['results'][0]['masks'][result['mask_idx']]['object_center_3d_local']


        else:        

            ref_row = result['ref_row']
            R_ref_cam = np.array([[float(ref_row['r11']), float(ref_row['r12']), float(ref_row['r13'])], 
                                    [float(ref_row['r21']), float(ref_row['r22']), float(ref_row['r23'])], 
                                    [float(ref_row['r31']), float(ref_row['r32']), float(ref_row['r33'])]])
            T_ref_cam = np.array([float(ref_row['tx']), float(ref_row['ty']), float(ref_row['tz'])])
            
            matched_points0_3d = result['matched_points0_3d']
            matched_points1_3d = result['matched_points1_3d']
            
            matched_points0_wrt_world = []
            for pt in matched_points0_3d:
                matched_points0_wrt_world.append(transform_3d_point(pt, self.scene_data[result['scene_cam_row_num']]))
            
            R_test_ref, T_test_ref, rmse = compute_rigid_transform(matched_points1_3d, matched_points0_3d)
            R_test_cam = R_test_ref @ R_ref_cam
            T_test_cam = R_test_ref @ T_ref_cam + T_test_ref

            camera_id = ref_row['camera_type']
            object_id = ref_row['object_id']
            
        R_test_world, T_test_world = transform_pose_to_world(R_test_cam, T_test_cam, self.scene_data[result['scene_cam_row_num']])
        
        distance = np.linalg.norm(np.array(test_center_3d_world) - np.array(T_test_world))
        
        if (
            distance > 120
            and not np.isnan(test_center_3d_world).any()  # no NaNs
            and not np.allclose(test_center_3d_world, [0, 0, 0])  # not (0,0,0)
        ):
            return False
        
        result['R'] = R_test_world
        result['T'] = T_test_world
        result['rmse'] = rmse
        
        
    
        summary_result = {
            "camera ID": camera_id,
            "object ID": object_id,
            "R_test_cam": R_test_cam,
            "T_test_cam": T_test_cam,
            "R_test_world": result['R'],
            "T_test_world": result['T'],
            "rmse": result['rmse']
        }
        
        self.display_results["results_summary"].append(summary_result)
        
        return True

    def compute_6d_poses_backup(self):
        """Sequential 6D pose computation"""
        print("Starting 6D pose computation...")
        start_time = time.time()
        
        if not self.detections_homographies:
            return
        
        
        for result in self.detections_homographies:
            test_center_3d_world = self.scene_data[result['scene_cam_row_num']]['detection_json']['results'][0]['masks'][result['mask_idx']]['object_center_3d_world']
            
            if result['num_matches'] == 0:
                result['T'] = test_center_3d_world
                result['rmse'] = 0.5
                
                image = self.scene_data[result['scene_cam_row_num']]["rgb_img"]
                polygon_mask = self.scene_data[result['scene_cam_row_num']]['detection_json']['results'][0]['masks'][result['mask_idx']]['points']
                
                R_test_cam = result['R']
                T_test_cam = self.scene_data[result['scene_cam_row_num']]['detection_json']['results'][0]['masks'][result['mask_idx']]['object_center_3d_local']

                camera_id = result['camera_id']
                object_id = result['object_idx']
            else:
                ref_row = result['ref_row']
                R_ref_cam = np.array([[float(ref_row['r11']), float(ref_row['r12']), float(ref_row['r13'])], 
                                     [float(ref_row['r21']), float(ref_row['r22']), float(ref_row['r23'])], 
                                     [float(ref_row['r31']), float(ref_row['r32']), float(ref_row['r33'])]])
                T_ref_cam = np.array([float(ref_row['tx']), float(ref_row['ty']), float(ref_row['tz'])])
                
                matched_points0_3d = result['matched_points0_3d']
                matched_points1_3d = result['matched_points1_3d']
                
                matched_points0_wrt_world = []
                for pt in matched_points0_3d:
                    matched_points0_wrt_world.append(transform_3d_point(pt, self.scene_data[result['scene_cam_row_num']]))
                
                R_test_ref, T_test_ref, rmse = compute_rigid_transform(matched_points1_3d, matched_points0_3d)
                R_test_cam = R_test_ref @ R_ref_cam
                T_test_cam = R_test_ref @ T_ref_cam + T_test_ref
                
                R_test_world, T_test_world = transform_pose_to_world(R_test_cam, T_test_cam, self.scene_data[result['scene_cam_row_num']])
                
                distance = np.linalg.norm(np.array(test_center_3d_world) - np.array(T_test_world))
                
                if (
                    distance > 70
                    and not np.isnan(test_center_3d_world).any()  # no NaNs
                    and not np.allclose(test_center_3d_world, [0, 0, 0])  # not (0,0,0)
                ):
                    T_test_cam = self.scene_data[result['scene_cam_row_num']]['detection_json']['results'][0]['masks'][result['mask_idx']]['object_center_3d_local']
                    R_test_world, T_test_world = transform_pose_to_world(R_test_cam, T_test_cam, self.scene_data[result['scene_cam_row_num']])
                
                result['R'] = R_test_world
                result['T'] = T_test_world
                result['rmse'] = rmse
                
                camera_id = ref_row['camera_type']
                object_id = ref_row['object_id']
            
            summary_result = {
                "camera ID": camera_id,
                "object ID": object_id,
                "R_test_cam": R_test_cam,
                "T_test_cam": T_test_cam,
                "R_test_world": result['R'],
                "T_test_world": result['T'],
                "rmse": result['rmse']
            }
            
            self.display_results["results_summary"].append(summary_result)
        
        print(f"6D pose computation completed in {time.time() - start_time:.2f} seconds")

    def process_all(self):
        """Process entire pipeline with careful GPU management"""
        total_start = time.time()
        
        self.mask_objects()
        self.consolidate_detections()
        self.do_feature_matchings()
        self.compute_6d_poses()
        
        total_time = time.time() - total_start
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        
        return self.display_results

    def compare_masked_images(self, test_img, test_polygon_mask, camera_id, object_id):
        """Legacy method for backward compatibility"""
        # Initialize pool if not done
        if not _pool_initialized:
            initialize_matcher_pool(pool_size=1)
        return compare_masked_images_optimized(test_img, test_polygon_mask, camera_id, object_id, self.ref_dataset_grouped)