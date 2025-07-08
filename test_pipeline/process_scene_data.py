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

from update_detection_json import add_3d_centers_to_json

class process_scene_data:
    
    def __init__(self, scene_data, max_workers=None):
        self.scene_data = scene_data
        self.config = self.load_config("config.yaml")
        self.meta_data_path = self.config["meta_data_folder"]

        # Auto-detect optimal number of workers
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(scene_data), 8)  # Cap at 8 for GPU memory
        self.max_workers = max_workers

        # self.mask_objects()

        self.detections = []
        self.detections_cluster_distance_threshold = 50
        self.consolidated_detections = []
        # self.consolidate_detections()

        self.min_nb_of_matches = 10
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
            for j in range(0, len(self.consolidated_detections[i])):

                detection = self.detections[self.consolidated_detections[i][j]]
                row_num = detection[4]
                mask_idx = detection[3]
                camera_id = detection[1]
                object_id = int(detection[0].split('_')[1])
                mask = self.scene_data[row_num]["detection_json"]["results"][0]["masks"][mask_idx]["points"]
                image_id = self.scene_data[row_num]["image_id"]
                scene_id = self.scene_data[row_num]["scene_id"]

                print("scene_id: ", scene_id)
                print("image_id: ", image_id)
                print("camera_id: ", camera_id)
                print("mask ID: ", mask_idx )
                print("object_id: ", object_id)
                
                num_matches, confidence, viz_image, ref_row = self.compare_masked_images(test_img = self.scene_data[row_num]["rgb_img"], 
                                                    test_polygon_mask = mask,
                                                    camera_id = camera_id, 
                                                    object_id = object_id)
                
                if num_matches is not None:
                    save_match_img_path = os.path.join(self.meta_data_path, "matches_scene_" + str(scene_id)  + "_image_" + str(image_id) + "_cam_" + camera_id + "_obj_" + str(object_id) + "_mask_" + str(mask_idx) + ".png")
                    cv2.imwrite(save_match_img_path, viz_image)

                    break

                    
                    



    def compare_masked_images(self, test_img, test_polygon_mask, camera_id, object_id):
        
        # Load and mask test image
        test_mask = np.zeros(test_img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(test_mask, [np.array(test_polygon_mask, dtype=np.int32)], 255)
        masked_test = cv2.bitwise_and(test_img, test_img, mask=test_mask)


        csv_path = self.config["ref_csv_path"]
        
        # Read CSV and process matching rows
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
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
                    
                    cv2.imwrite('/home/rama/bpc_ws/bpc/test_pipeline/tmp/ref.png', masked_ref)
                    cv2.imwrite('/home/rama/bpc_ws/bpc/test_pipeline/tmp/test.png', masked_test)

                    num_matches, confidence, viz_image = matcher.superglue(masked_test, masked_ref)
                   

                    if num_matches is not None:
                        print("Number of matches: ", num_matches)
                        print(f"Average confidence: ", confidence)
                        return num_matches, confidence, viz_image, row

                    
        
        return None, None, None, None
                

        



    
    


    