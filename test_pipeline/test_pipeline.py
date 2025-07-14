import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from process_scene_data import process_scene_data


class SceneImageGroupProcessor:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
    
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
                print(f"\nGroup {group_num}: Scene {scene_id}, Image {image_id}")
                print(f"Number of cameras: {len(camera_rows)}")
                print("-" * 40)
                
                scene_info = process_scene_data(scene_data=camera_rows)
                scene_info.mask_objects()
                scene_info.consolidate_detections()

                scene_info.do_feature_matchings()
                scene_info.compute_6d_poses()

                

                
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