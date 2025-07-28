import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from process_scene_data import process_scene_data
import sys
import time
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utilities')))
from generate_output_csv import CSVWriter

class SceneImageGroupProcessor:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.csv_writer = CSVWriter('results.csv')
    
    def iterate_by_scene_image_groups(self):
        """
        Iterate through CSV grouped by (scene_id, image_id) combinations.
        For each group, process all camera rows together.
        """
        
        # Dictionary to group rows by (scene_id, image_id)
        grouped_data = defaultdict(list)

        k = None
        
        try:
            # First pass: Read and group the data
            with open(self.csv_file_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    scene_id = row['scene_id']
                    image_id = row['image_id']
                    key = (scene_id, image_id)
                    grouped_data[key].append(row)
            
            # Second pass: Iterate through each group
            print("Iterating through scene_id, image_id combinations:")
            print("=" * 60)
            
            for group_num, ((scene_id, image_id), camera_rows) in enumerate(grouped_data.items(), start=1):
                start_time = time.time()
                print(f"\nGroup {group_num}: Scene {scene_id}, Image {image_id}")
                print(f"Number of cameras: {len(camera_rows)}")
                print("-" * 40)
                
                scene_info = process_scene_data(scene_data=camera_rows)
                scene_info.mask_objects()
                scene_info.consolidate_detections()

                scene_info.do_feature_matchings()
                scene_info.compute_6d_poses()

                end_time = time.time()

                for result in scene_info.detections_homographies:
                    scene_id = scene_id
                    im_id = image_id
                    obj_id = result['object_idx']
                    score = result['confidence']
                    R = result['R']
                    t = result['T']
                    time_taken = end_time-start_time
                    # Add row to CSV and save
                    self.csv_writer.add_row(scene_id, im_id, obj_id, score, R, t, time_taken)

                del scene_info
                gc.collect()

                

                
        except FileNotFoundError:
            print(f"Error: Could not find the file '{self.csv_file_path}'")
        except KeyError as e:
            print(f"Error: Column {e} not found in the CSV file")
        except Exception as e:
            print(f"Error reading CSV file: {e}")


if __name__ == "__main__":
    csv_file_path = "/home/rama/bpc_ws/bpc/utilities/test_dataset.csv"
    
    processor = SceneImageGroupProcessor(csv_file_path)
    processor.iterate_by_scene_image_groups()