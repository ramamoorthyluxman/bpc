import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pose_estimation_pipeline')))

from pose_estimation_pipeline import pose_estimation_pipeline

import os
import time
import csv


# Initialize pipeline
pipeline = pose_estimation_pipeline("/home/rama/bpc_ws/bpc/pose_estimation_pipeline/config.yaml")

dataset_path = "/home/rama/bpc_ws/bpc/datasets/dummy"
camera_params_path = "/home/rama/bpc_ws/bpc/ipd/camera_photoneo.json"  # Update as needed
output_csv = "pose_estimation_results.csv"

# Create CSV file
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])
    
    for scene_folder in sorted(os.listdir(dataset_path)):
        scene_path = os.path.join(dataset_path, scene_folder)
        if os.path.isdir(scene_path):
            rgb_folder = os.path.join(scene_path, "rgb_photoneo")
            depth_folder = os.path.join(scene_path, "depth_photoneo")
            
            if os.path.exists(rgb_folder) and os.path.exists(depth_folder):
                for image_file in sorted(os.listdir(rgb_folder)):
                    if image_file.endswith('.png'):
                        image_id = image_file.replace('.png', '')
                        
                        rgb_image_path = os.path.join(rgb_folder, image_file)
                        depth_image_path = os.path.join(depth_folder, image_file)
                        
                        # Check if depth image exists
                        if os.path.exists(depth_image_path):
                            print(f"Processing scene: {scene_folder}, image: {image_id}")
                            
                            try:
                                # Run pipeline and measure time
                                start_time = time.time()
                                results = pipeline.pose_estimate_pipeline(rgb_image_path, depth_image_path, camera_params_path)
                                execution_time = time.time() - start_time
                                
                                # Parse results and write to CSV
                                for result in results:
                                    if result.get('success', False):  # Only process successful detections
                                        obj_id = result['label']
                                        score = result['confidence_score']
                                        transformation = result['transformation']
                                        
                                        R = transformation[0]  # Rotation matrix as string
                                        t = transformation[1]  # Translation vector as string
                                        
                                        writer.writerow([scene_folder, image_id, obj_id, score, R, t, execution_time])
                            
                            except Exception as e:
                                print(f"Error processing {scene_folder}/{image_id}: {e}")
                                continue

print(f"Results saved to {output_csv}")