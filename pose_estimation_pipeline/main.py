# config.py
import yaml
import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'maskRCNN')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pcl_builder')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'roi_extraction')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pointcloud_registration')))

from inference import infer
from create_pointcloud import build_pcl
from roi_extraction import extract_and_save_labeled_point_clouds
from pointcloud_processor import PointCloudProcessor


## Load configs

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
config = load_config()

## Object detection

maskrcnn_image_path = config['maskrcnn_image_path']
maskrcnn_model_path = config['maskrcnn_model_path']
maskrcnn_output_path = config['maskrcnn_output_path']
maskrcnn_category_txt_path = config['maskrcnn_category_txt_path']
maskrcnn_confidence_threshold = config['maskrcnn_confidence_threshold']


infer(maskrcnn_image_path, maskrcnn_model_path, maskrcnn_output_path, maskrcnn_category_txt_path, maskrcnn_confidence_threshold)

## Build PCL

pcl_builder_rgb_image_path = config['pcl_builder_rgb_image_path']
pcl_builder_depth_image_path = config['pcl_builder_depth_image_path']
pcl_builder_camera_parameters_path = config['pcl_builder_camera_parameters_path']
pcl_builder_output_file_path = config['pcl_builder_output_file_path']

build_pcl(pcl_builder_rgb_image_path, pcl_builder_depth_image_path, pcl_builder_camera_parameters_path, pcl_builder_output_file_path)

## ROI extraction 

roi_extraction_annotation_json_path = config["roi_extraction_annotation_json_path"]
roi_extraction_output_path = config["roi_extraction_output_path"]

extract_and_save_labeled_point_clouds(pcl_builder_output_file_path, roi_extraction_annotation_json_path, roi_extraction_output_path)

## Register

register_test_roi_path = config["register_test_roi_path"]
register_reference_model_path = config["register_reference_model_path"]
register_output_path = config["register_output_path"]

register_pcl = PointCloudProcessor()
register_pcl.run_full_pipeline(source_file=register_test_roi_path, reference_file=register_reference_model_path, scene_file=pcl_builder_output_file_path, output_file=register_output_path)

