import os
import sys
import yaml 
import cv2
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
from collections import defaultdict
import numpy as np
import csv
import ast


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pcl_builder')))
from create_pointcloud_gpu_accelerated import build_pcl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SuperGluePretrainedNetwork')))
from superglue import SuperGlueMatcher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'maskRCNN')))
from inference import infer

from update_detection_json import add_3d_centers_to_json, find_valid_neighbor_pixel, transform_3d_point,  transform_pose_to_world
from homogrpahy_to_pose import pose_from_homography_complete
from pose_estimator import compute_rigid_transform

class process_scene_data:
    
    def __init__(self, scene_data, max_workers=None):
        self.scene_data = scene_data
        self.config = self.load_config("config.yaml")
        self.meta_data_path = self.config["meta_data_folder"]

        # Auto-detect optimal number of workers
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(scene_data), 8)  # Cap at 8 for GPU memory
        self.max_workers = max_workers

        ref_csv_path = self.config["ref_csv_path"]
        with open(ref_csv_path, 'r') as f:
            self.ref_dataset = list(csv.DictReader(f))

        self.detections = []
        self.detections_cluster_distance_threshold = 50
        self.consolidated_detections = []
        self.detections_homographies = [] 


        # self.do_feature_matchings()

        

    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
        
    def mask_objects(self):
        for i, cam_row in enumerate(self.scene_data):
            cam_row["rgb_img"] = cv2.imread(cam_row["rgb_image_path"])
            cam_row["depth_img"] = cv2.imread(cam_row["depth_image_path"])
            cam_row["cam_params_file"] = self.config[cam_row["camera_id"]]
            maskrcnn_model_path = self.config['maskrcnn_model_path']
            maskrcnn_output_path = os.path.join(self.meta_data_path, f"{cam_row['scene_id']}_{cam_row['image_id']}_{i}")
            maskrcnn_category_txt_path = self.config['maskrcnn_category_txt_path']
            maskrcnn_confidence_threshold = self.config['maskrcnn_confidence_threshold']

            detection_json = infer(cam_row["rgb_image_path"], maskrcnn_model_path, maskrcnn_output_path,  maskrcnn_category_txt_path, maskrcnn_confidence_threshold, visualize_and_save=self.config['maskrcnn_visualization_and_save'])

            cam_row["detection_json"] = detection_json

            cam_row["detection_json"] = add_3d_centers_to_json(
                cam_row=cam_row,
                json_data=cam_row["detection_json"],
                depth_path=cam_row["depth_image_path"],
                search_radius=5
            )

        return self.scene_data


    def consolidate_detections(self):

        """
        Parse CSV with camera detections and return object data.
        
        Returns: List of [object_id, camera_id, location_world, row_number]
        """
        
        
        for row_num, row in enumerate(self.scene_data):
            camera_id = row['camera_id']
            detection_data = row['detection_json']
            
            # Extract objects from masks
            for mask_idx, mask in enumerate(detection_data['results'][0]['masks']):
                object_id = mask['label']
                location_world = mask['object_center_3d_world']
                self.detections.append([object_id, camera_id, location_world, mask_idx, row_num])

        print("detections: ", self.detections)

        # Group detections by same object in same location from different cameras.
    
        # Group by object_id first
        by_object = defaultdict(list)
        for idx, detection in enumerate(self.detections):
            by_object[detection[0]].append(idx)
        
        # For each object, cluster by location
        for indices in by_object.values():
            used = set()
            
            for i, idx1 in enumerate(indices):
                if idx1 in used:
                    continue
                    
                # Start new cluster with this detection
                cluster = [idx1]
                used.add(idx1)
                loc1 = self.detections[idx1][2]
                
                # Find nearby detections
                for j, idx2 in enumerate(indices):
                    if idx2 in used:
                        continue
                        
                    loc2 = self.detections[idx2][2]
                    distance = np.linalg.norm(loc1 - loc2)
                    
                    if distance <= self.detections_cluster_distance_threshold:
                        cluster.append(idx2)
                        used.add(idx2)
                
                self.consolidated_detections.append(cluster)




    def do_feature_matchings(self):
        
        for i in range(0,len(self.consolidated_detections)):
            print("Detections shape: ", len(self.consolidated_detections),len(self.consolidated_detections[i]))
            for j in range(0, len(self.consolidated_detections[i])):

                detection = self.detections[self.consolidated_detections[i][j]]
                row_num = detection[4]
                mask_idx = detection[3]
                camera_id = detection[1]
                object_id = int(detection[0].split('_')[1])
                mask = self.scene_data[row_num]["detection_json"]["results"][0]["masks"][mask_idx]["points"]
                image_id = self.scene_data[row_num]["image_id"]
                scene_id = self.scene_data[row_num]["scene_id"]

                # print("scene_id: ", scene_id)
                # print("image_id: ", image_id)
                # print("camera_id: ", camera_id)
                # print("mask ID: ", mask_idx )
                # print("object_id: ", object_id)
                
                num_matches, confidence, viz_image, h_mat, ref_row, matched_points0, matched_points1 = self.compare_masked_images(test_img = self.scene_data[row_num]["rgb_img"], 
                                                    test_polygon_mask = mask,
                                                    camera_id = camera_id, 
                                                    object_id = object_id)
                
                if num_matches is not None:

                    K = np.array([float(self.scene_data[row_num][f'cam_K_{i}']) for i in range(9)]).reshape(3, 3)
                    # Extract depth scale
                    depth_scale = float(self.scene_data[row_num]['scene_depth_scale'])

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
                        cv2.imwrite(save_match_img_path, viz_image)
                        result = {'scene_cam_row_num': row_num,
                                'mask_idx': mask_idx,
                                'object_idx': object_id,
                                'ref_row': ref_row,
                                'h_mat': h_mat,
                                'confidence': confidence,
                                'num_matches': num_matches,
                                'matched_points0': matched_points0,
                                'matched_points1': matched_points1,
                                'matched_points0_3d' : matched_points0_3d,
                                'matched_points1_3d' : matched_points1_3d}
                        self.detections_homographies.append(result)
                        
                        
                        break
                    
                    elif j == len(self.consolidated_detections[i])-1:
                        result = {'scene_cam_row_num': row_num,
                                'mask_idx': mask_idx,
                                'object_idx': object_id,
                                'ref_row': ref_row,
                                'h_mat': h_mat,
                                'confidence': confidence,
                                'num_matches': 0,
                                'matched_points0': matched_points0,
                                'matched_points1': matched_points1,
                                'matched_points0_3d' : matched_points0_3d,
                                'matched_points1_3d' : matched_points1_3d}

                        self.detections_homographies.append(result)
                    
    def compare_masked_images(self, test_img, test_polygon_mask, camera_id, object_id):
        
        # Load and mask test image
        test_mask = np.zeros(test_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(test_mask, [np.array(test_polygon_mask, dtype=np.int32)], 255)
        masked_test = cv2.bitwise_and(test_img, test_img, mask=test_mask)
        
        for row in self.ref_dataset:
            # Check if camera_type and object_id match
            if row['camera_type'] == camera_id and int(row['object_id']) == object_id:
                # Load reference image
                ref_img = cv2.imread(row['image_path'])
                
                # Parse and apply reference polygon mask
                ref_polygon = ast.literal_eval(row['polygon_mask'])
                ref_mask = np.zeros(ref_img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(ref_mask, [np.array(ref_polygon, dtype=np.int32)], 255)
                masked_ref = cv2.bitwise_and(ref_img, ref_img, mask=ref_mask)
                
                # Compare using superglue
                # Initialize the matcher
                matcher = SuperGlueMatcher()
                # Use it
                
                num_matches, confidence, viz_image, h_mat, matched_points0, matched_points1 = matcher.superglue(masked_test, masked_ref)

                if num_matches is not None:

                    num_matches, confidence, viz_image, h_mat, matched_points0, matched_points1 = matcher.superglue(masked_test, masked_ref)
                   

                    # if num_matches is not None:
                    if num_matches is not None:
                        if num_matches>4:
                            return num_matches, confidence, viz_image, h_mat, row, matched_points0, matched_points1

                    
        
        return None, None, None, None, None, None, None
    

    
    def compute_6d_poses(self):
        if self.detections_homographies is None:
            return 
        
        for result in self.detections_homographies:

            ref_row = result['ref_row']

            # this is the pose of the reference object wrt camera
            R_ref_cam = [[ref_row['r11'], ref_row['r12'], ref_row['r13']], 
                           [ref_row['r21'], ref_row['r22'], ref_row['r23']], 
                           [ref_row['r31'], ref_row['r32'], ref_row['r33']] ]

            T_ref_cam = [ref_row['tx'], ref_row['ty'], ref_row['tz']]

            R_ref_cam = np.array([[float(val) for val in row] for row in R_ref_cam])
            T_ref_cam = np.array([float(val) for val in T_ref_cam]) 
            
            matched_points0_3d = result['matched_points0_3d']
            matched_points1_3d = result['matched_points1_3d']

            print("Camera id", ref_row['camera_type'])

            print("matched_points0_3d (test) wrt camera: ", matched_points0_3d)
            print("matched_points1_3d (ref) wrt camera: ", matched_points1_3d)

            test_center_3d_world = self.scene_data[result['scene_cam_row_num']]['detection_json']['results'][0]['masks'][result['mask_idx']]['object_center_3d_world']
            print("test center 3D world: ", test_center_3d_world)

            if result['num_matches'] == 0:
                result['T'] = test_center_3d_world
                result['R'] = R_ref_cam
                result['rmse'] = 0.5

            else:

                matched_points0_wrt_world = []
                for pt in matched_points0_3d:
                    matched_points0_wrt_world.append(transform_3d_point(pt, self.scene_data[result['scene_cam_row_num']]))

                print('matched_points0_3d (test) wrt world', matched_points0_wrt_world)

                # this is the transformation between the test object and the reference object
                R_test_ref, T_test_ref, rmse = compute_rigid_transform(matched_points1_3d, matched_points0_3d)

                print('Rigid transformation R_test_ref: ', R_test_ref)
                print('Rigid transformation T_test_ref: ', T_test_ref)

                

                

                R_test_cam = R_test_ref @ R_ref_cam
                T_test_cam = R_test_ref @ T_ref_cam + T_test_ref

                print("T_test_cam: ", T_test_cam)

                # Now tranform this pose to world (photoneo)

                R_test_world, T_test_world = transform_pose_to_world(R_test_cam, T_test_cam, self.scene_data[result['scene_cam_row_num']])

                print("T_test_world: ", T_test_world)

                result['R'] = R_test_world
                result['T'] = T_test_world
                result['rmse'] = rmse 

            

            


    
    


    