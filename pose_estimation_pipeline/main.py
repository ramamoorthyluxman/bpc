# pose_estimation_pipeline.py - Object Pose Estimation Pipeline Class
import yaml
import os
import sys
import torch
import time
from datetime import datetime
import concurrent.futures
import multiprocessing
from typing import List, Tuple, Dict, Any
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'maskRCNN')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pcl_builder')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'roi_extraction')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pointcloud_registration')))

from inference import infer
from create_pointcloud_gpu_accelerated import build_pcl
from roi_extraction import extract_labeled_point_clouds
from pointcloud_processor import PointCloudProcessor


class pose_estimation_pipeline:
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the pose estimation pipeline
        
        Args:
            config_path (str): Path to the configuration YAML file
        """
        self.config = self.load_config(config_path)
        self.scene_id = "000000"
        self.camera_id = "rgb_cam1"
        self.img_id = "000000"
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def print_step_timing(self, step_name, start_time=None, end_time=None):
        """Print timing information for pipeline steps"""
        if start_time is None:
            # Starting a step
            current_time = datetime.now()
            print(f"\n{'='*60}")
            print(f"STARTING: {step_name}")
            print(f"Start Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")
            return time.time()
        else:
            # Ending a step
            duration = end_time - start_time
            current_time = datetime.now()
            print(f"\n{'='*60}")
            print(f"COMPLETED: {step_name}")
            print(f"End Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            print(f"{'='*60}")

    def register_single_object(self, args):
        """
        Register a single object - designed for parallel execution
        
        Args:
            args: Tuple containing (object_data, object_info, scene_pcl, config, object_index)
        
        Returns:
            Dict containing registration results for this object
        """
        object_pcl, object_info, scene_pcl, config, object_index = args
        
        try:
            # Create a separate processor instance for this thread/process
            register_pcl = PointCloudProcessor()
            
            # Get reference model path for this object
            reference_model_path = os.path.join(
                config["register_reference_models_path"], 
                object_info['label'] + ".ply"
            )
            
            # Create output path for this specific object
            output_file = os.path.join(
                config["register_output_path"],
                f"object_{object_index}_{object_info['label']}_registration.ply"
            )
            
            print(f"[Object {object_index}] Starting registration for: {object_info['label']}")
            print(f"[Object {object_index}] Reference model: {reference_model_path}")
            
            # Check if reference model exists
            if not os.path.exists(reference_model_path):
                print(f"[Object {object_index}] WARNING: Reference model not found: {reference_model_path}")
                return {
                    'object_index': object_index,
                    'label': object_info['label'],
                    'success': False,
                    'error': 'Reference model not found',
                    'transformation': None
                }
            
            # Run registration
            transformation = register_pcl.run_full_pipeline(
                source_pcl=object_pcl,
                reference_file_path=reference_model_path,
                scene_pcl=scene_pcl,
                output_file=output_file,
                visualize_and_save_results=config["registration_visualization_and_save"]
            )
            
            print(f"[Object {object_index}] Registration completed for: {object_info['label']}")
            
            return {
                'object_index': object_index,
                'label': object_info['label'],
                'success': True,
                'transformation': transformation,
                'object_info': object_info,
                'output_file': output_file
            }
            
        except Exception as e:
            print(f"[Object {object_index}] Registration failed for: {object_info['label']}")
            print(f"[Object {object_index}] Error: {str(e)}")
            return {
                'object_index': object_index,
                'label': object_info['label'],
                'success': False,
                'error': str(e),
                'transformation': None
            }

    def register_all_objects_parallel(self, extracted_clouds_list, scene_pcl, max_workers=None):
        """
        Register all detected objects in parallel
        
        Args:
            extracted_clouds_list: List of (point_cloud, object_info) tuples
            scene_pcl: Original scene point cloud
            max_workers: Maximum number of parallel workers (None for auto-detect)
        
        Returns:
            List of registration results for all objects
        """
        if not extracted_clouds_list:
            print("No objects detected for registration")
            return []
        
        print(f"Preparing to register {len(extracted_clouds_list)} objects in parallel")
        
        # Prepare arguments for parallel processing
        registration_args = [
            (object_pcl, object_info, scene_pcl, self.config, idx)
            for idx, (object_pcl, object_info) in enumerate(extracted_clouds_list)
        ]
        
        # Determine optimal number of workers
        if max_workers is None:
            # Use number of CPU cores, but cap it to avoid overwhelming the system
            max_workers = min(len(extracted_clouds_list), multiprocessing.cpu_count())
        
        print(f"Using {max_workers} parallel workers for registration")
        
        results = []
        
        # Use ThreadPoolExecutor for I/O bound operations or ProcessPoolExecutor for CPU bound
        # ThreadPoolExecutor is often better for deep learning operations that may use shared GPU memory
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all registration tasks
            future_to_index = {
                executor.submit(self.register_single_object, args): idx
                for idx, args in enumerate(registration_args)
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_index):
                result = future.result()
                results.append(result)
                
                if result['success']:
                    print(f"✓ Object {result['object_index']} ({result['label']}) registered successfully")
                else:
                    print(f"✗ Object {result['object_index']} ({result['label']}) registration failed: {result['error']}")
        
        # Sort results by object index to maintain order
        results.sort(key=lambda x: x['object_index'])
        
        return results

    def print_registration_summary(self, registration_results):
        """Print a summary of all registration results"""
        successful = [r for r in registration_results if r['success']]
        failed = [r for r in registration_results if not r['success']]
        
        print(f"\n{'='*60}")
        print(f"REGISTRATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Objects: {len(registration_results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            print(f"\n✓ SUCCESSFUL REGISTRATIONS:")
            for result in successful:
                print(f"  - Object {result['object_index']}: {result['label']}")
                if result['transformation'] is not None:
                    print(f"    Transformation: {result['transformation']}")
        
        if failed:
            print(f"\n✗ FAILED REGISTRATIONS:")
            for result in failed:
                print(f"  - Object {result['object_index']}: {result['label']} - {result['error']}")
        
        print(f"{'='*60}")

    def pose_estimate_pipeline(self, rgb_image, depth_image, camera_parameters_file_path):
        """
        Main pipeline execution method
        
        Args:
            rgb_image (str): Path to RGB image
            depth_image (str): Path to depth image
            camera_parameters_file_path (str): Path to camera parameters file
            
        Returns:
            List of registration results for all detected objects
        """
        # Pipeline start
        pipeline_start = time.time()
        print(f"\n🚀 STARTING COMPLETE PIPELINE")
        print(f"Pipeline Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

        ## Step 1: Object detection
        step_start = self.print_step_timing("Object Detection (MaskRCNN Inference)")

        maskrcnn_model_path = self.config['maskrcnn_model_path']
        maskrcnn_output_path = self.config['maskrcnn_output_path']
        maskrcnn_category_txt_path = self.config['maskrcnn_category_txt_path']
        maskrcnn_confidence_threshold = self.config['maskrcnn_confidence_threshold']

        detection_json = infer(rgb_image, maskrcnn_model_path, maskrcnn_output_path, 
              maskrcnn_category_txt_path, maskrcnn_confidence_threshold, visualize_and_save=self.config['maskrcnn_visualization_and_save'])

        step_end = time.time()
        self.print_step_timing("Object Detection (MaskRCNN Inference)", step_start, step_end)

        ## Step 2: Build PCL
        step_start = self.print_step_timing("Point Cloud Generation")

        pcl_builder_output_file_path = "/home/rama/bpc_ws/bpc/pcl_builder/output/" + self.scene_id + "_" + self.camera_id + "_" + self.img_id + ".png"

        scene_pcl, width, height = build_pcl(rgb_image, depth_image, 
                  camera_parameters_file_path, pcl_builder_output_file_path, save_pcl=self.config['save_pcl'])

        print("scene pcl size: ", scene_pcl)

        step_end = time.time()
        self.print_step_timing("Point Cloud Generation", step_start, step_end)

        ## Step 3: ROI extraction
        step_start = self.print_step_timing("ROI Extraction")

        roi_extraction_output_path = self.config["roi_extraction_output_path"]

        extracted_clouds_list = extract_labeled_point_clouds(scene_pcl, detection_json["results"][0], roi_extraction_output_path, save_rois=self.config['save_rois'])

        step_end = time.time()
        self.print_step_timing("ROI Extraction", step_start, step_end)

        print(f"Total objects detected and extracted: {len(extracted_clouds_list)}")

        ## Step 4: Parallel Registration of ALL objects
        step_start = self.print_step_timing("Parallel Point Cloud Registration")

        # Register all objects in parallel
        registration_results = self.register_all_objects_parallel(
            extracted_clouds_list=extracted_clouds_list,
            scene_pcl=scene_pcl,
            max_workers=None  # Auto-detect optimal number of workers
        )

        step_end = time.time()
        self.print_step_timing("Parallel Point Cloud Registration", step_start, step_end)

        # Print detailed results
        self.print_registration_summary(registration_results)

        # Save all transformations to a file for later use
        transformations_file = os.path.join(self.config["register_output_path"], "all_transformations.yaml")
        transformations_data = {
            'scene_id': self.scene_id,
            'camera_id': self.camera_id,
            'img_id': self.img_id,
            'timestamp': datetime.now().isoformat(),
            'objects': []
        }

        for result in registration_results:
            if result['success']:
                transformations_data['objects'].append({
                    'object_index': result['object_index'],
                    'label': result['label'],
                    'transformation': result['transformation'].tolist() if hasattr(result['transformation'], 'tolist') else result['transformation'],
                    'output_file': result['output_file']
                })

        with open(transformations_file, 'w') as f:
            yaml.dump(transformations_data, f, default_flow_style=False)

        print(f"\nTransformations saved to: {transformations_file}")

        # Pipeline completion
        pipeline_end = time.time()
        total_duration = pipeline_end - pipeline_start

        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"Pipeline End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Pipeline Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        print(f"Objects Successfully Registered: {len([r for r in registration_results if r['success']])}/{len(registration_results)}")
        print(f"{'='*80}")
        
        return registration_results


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = pose_estimation_pipeline("config.yaml")
    
    # Run pipeline with your images
    rgb_image_path = "/home/rama/bpc_ws/bpc/datasets/ipd_val/000000/images/000000_cam1.jpg"
    depth_image_path = "/home/rama/bpc_ws/bpc/ipd/val/000000/depth_cam1/000000.png"
    camera_params_path = "/home/rama/bpc_ws/bpc/ipd/camera_cam1.json"
    
    results = pipeline.pose_estimate_pipeline(rgb_image_path, depth_image_path, camera_params_path)