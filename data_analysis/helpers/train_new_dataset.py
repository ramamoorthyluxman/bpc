import numpy as np
import os
import csv
import json
import shutil
from pathlib import Path
import ast
import yaml
from collections import defaultdict
import sys
import subprocess
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'maskRCNN')))
from train import Trainer


class TrainNewDataset:
    def __init__(self, dataset_csv_path, use_depth=False, progress_callback=None, stop_flag=None):
        self.dataset_csv = self.load_csv(dataset_csv_path)
        self.use_depth = use_depth
        self.progress_callback = progress_callback
        self.stop_flag = stop_flag
        self.config = self.load_config('config.yaml')
        self.output_dir = os.path.join(self.config.get('meta_data_folder'), 'maskrcnn_training_data')
        self.prepare_maskrcnn_dataset(output_dir=self.output_dir)
        

    def load_config(self, config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def load_csv(self, csv_path):
        """Load CSV as a list of dictionaries"""
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _report_progress(self, message):
        """Report progress to callback if available"""
        if self.progress_callback:
            self.progress_callback(message)
        print(message)

    def train(self):
        """Train Mask R-CNN model using the prepared dataset"""
        self._report_progress("Starting Mask R-CNN training...")
        
        # Determine which dataset folder to use
        if self.use_depth:
            data_path = os.path.join(self.output_dir, "depth")
            self._report_progress(f"Training with depth images from: {data_path}")
        else:
            data_path = os.path.join(self.output_dir, "rgb")
            self._report_progress(f"Training with RGB images from: {data_path}")
        
        # Check if data path exists
        if not os.path.exists(data_path):
            error_msg = f"Error: Data path does not exist: {data_path}"
            self._report_progress(error_msg)
            return None
        
        # Create models directory within maskrcnn_training_data
        models_dir = os.path.join(self.output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        self._report_progress(f"Models will be saved to: {models_dir}")
        
        # Create arguments for the train script
        class Args:
            def __init__(self):
                self.data_path = data_path
                self.output_dir = models_dir
                self.seed = 42
                self.batch_size = 2
                self.epochs = 1000  # You can make this configurable
                self.workers = 4
                self.lr = 0.005
                self.momentum = 0.9
                self.weight_decay = 0.0005
                self.lr_step_size = 5
                self.lr_gamma = 0.1
                self.print_freq = 10
                self.save_freq = 5
                self.resume = ''
        
        args = Args()
        
        # Log training parameters
        self._report_progress("\nTraining parameters:")
        self._report_progress(f"  Data path: {args.data_path}")
        self._report_progress(f"  Output dir: {args.output_dir}")
        self._report_progress(f"  Batch size: {args.batch_size}")
        self._report_progress(f"  Epochs: {args.epochs}")
        self._report_progress(f"  Learning rate: {args.lr}")
        self._report_progress(f"  Save frequency: {args.save_freq} epochs")
        
        try:
            # Call the training function with progress callback and stop flag
            self._report_progress("\nStarting training process...")
            trainer = Trainer(args, progress_callback=self.progress_callback, stop_flag=self.stop_flag)
            trainer.train()
            
            # After training, return the path to the trained model
            final_model_path = os.path.join(models_dir, f"model_epoch_{args.epochs-1}.pth")
            if os.path.exists(final_model_path):
                self._report_progress(f"\nTraining completed successfully!")
                self._report_progress(f"Final model saved at: {final_model_path}")
                return final_model_path
            else:
                # Look for any saved model
                model_files = [f for f in os.listdir(models_dir) if f.startswith("model_epoch_")]
                if model_files:
                    latest_model = sorted(model_files)[-1]
                    latest_path = os.path.join(models_dir, latest_model)
                    self._report_progress(f"\nTraining completed. Latest model: {latest_path}")
                    return latest_path
                else:
                    self._report_progress("\nWarning: No model files found after training")
                    return None
                    
        except Exception as e:
            error_msg = f"\nError during training: {e}"
            self._report_progress(error_msg)
            import traceback
            traceback.print_exc()
            return None
    
    def prepare_maskrcnn_dataset(self, output_dir="maskrcnn_training_data"):
        """Prepare Mask R-CNN training data from CSV dataset."""
        
        dataset_csv = self.dataset_csv
        output_path = Path(output_dir)
        
        # Create RGB and depth directories with nested structure
        rgb_dir = output_path / "rgb"
        depth_dir = output_path / "depth"
        
        rgb_images_dir = rgb_dir / "images"
        rgb_annotations_dir = rgb_dir / "annotations"
        depth_images_dir = depth_dir / "images"
        depth_annotations_dir = depth_dir / "annotations"
        
        # Create all directories
        rgb_images_dir.mkdir(parents=True, exist_ok=True)
        rgb_annotations_dir.mkdir(parents=True, exist_ok=True)
        if self.use_depth:
            depth_images_dir.mkdir(parents=True, exist_ok=True)
            depth_annotations_dir.mkdir(parents=True, exist_ok=True)
        
        
        # Group annotations by scene_id, camera_type, and image path
        image_annotations = defaultdict(list)
        
        # Iterate over list of dictionaries
        for idx, row in enumerate(dataset_csv):
            # Handle column name variations (mage_path vs image_path)
            if 'mage_path' in row:
                image_path = row['mage_path']
            elif 'image_path' in row:
                image_path = row['image_path']
            else:
                print(f"Warning: No image path found in row {idx}")
                continue
            
            # Get scene_id and camera_type for grouping
            scene_id = row.get('scene_id', 'unknown_scene')
            camera_type = row.get('camera_type', row.get('camera_id', 'unknown_cam'))
            
            # Create unique key for grouping
            # Using tuple of (scene_id, camera_type, image_path) as key
            group_key = (scene_id, camera_type, image_path)
            
            # Parse polygon mask (it's stored as string representation of list)
            try:
                polygon_points = ast.literal_eval(row['polygon_mask'])
            except (ValueError, SyntaxError):
                print(f"Warning: Could not parse polygon_mask for row {idx}")
                continue
            
            # Create annotation data for this object
            annotation_data = {
                'label': str(row["object_id"]),  # Default label, modify as needed based on your data
                'points': polygon_points,
                'object_id': int(row.get('object_id', idx)),
                'bbox': {
                    'x': int(row['bbox_x']),
                    'y': int(row['bbox_y']),
                    'width': int(row['bbox_w']),
                    'height': int(row['bbox_h'])
                }
            }
            
            # Add to the image's annotations with the group key
            image_annotations[group_key].append({
                'annotation': annotation_data,
                'image_width': int(row['image_width']),
                'image_height': int(row['image_height']),
                'scene_id': scene_id,
                'camera_type': camera_type,
                'original_image_path': image_path
            })
        
        # Process each unique image group
        processed_rgb_images = set()
        processed_depth_images = set()
        
        for group_key, annotations in image_annotations.items():
            scene_id, camera_type, image_path = group_key
            
            # Get original RGB image path
            original_rgb_path = Path(image_path)
            
            # Skip if RGB image doesn't exist
            if not original_rgb_path.exists():
                print(f"Warning: RGB image not found: {image_path}")
                continue
            
            # Create unique image name with scene_id, camera_type, and original name
            original_name_stem = original_rgb_path.stem
            original_extension = original_rgb_path.suffix
            
            # New filename format: scene_id_camera_type_originalname
            new_image_name = f"{scene_id}_{camera_type}_{original_name_stem}{original_extension}"
            
            # Process RGB image
            rgb_symlink_path = rgb_images_dir / new_image_name
            
            # Remove existing symlink if it exists
            if rgb_symlink_path.exists() or rgb_symlink_path.is_symlink():
                rgb_symlink_path.unlink()
            
            # Create new RGB symbolic link
            try:
                rgb_symlink_path.symlink_to(original_rgb_path.absolute())
                processed_rgb_images.add(group_key)
            except Exception as e:
                print(f"Error creating RGB symlink for {image_path}: {e}")
                continue
            
            # Prepare RGB JSON annotation
            rgb_json_data = {
                "image_path": f"images/{new_image_name}",
                "original_path": str(original_rgb_path.absolute()),
                "scene_id": scene_id,
                "camera_type": camera_type,
                "height": annotations[0]['image_height'],
                "width": annotations[0]['image_width'],
                "masks": []
            }
            
            # Add all masks for this image
            for anno_data in annotations:
                mask_entry = {
                    "mask_path": None,  # No separate mask files in this case
                    "label": anno_data['annotation']['label'],
                    "points": anno_data['annotation']['points'],
                    "bbox": anno_data['annotation']['bbox'],
                    "object_id": anno_data['annotation']['object_id']
                }
                rgb_json_data["masks"].append(mask_entry)
            
            # Save RGB JSON annotation file
            json_filename = f"{scene_id}_{camera_type}_{original_name_stem}.json"
            rgb_json_path = rgb_annotations_dir / json_filename
            
            with open(rgb_json_path, 'w') as f:
                json.dump(rgb_json_data, f, indent=2)
            
            # Process depth image if use_depth is True
            if self.use_depth:
                # Create depth image path by replacing 'rgb' with 'depth'
                depth_image_path = str(image_path).replace('/rgb_', '/depth_').replace('\\rgb_', '\\depth_')
                original_depth_path = Path(depth_image_path)
                
                # Check if depth image exists
                if not original_depth_path.exists():
                    print(f"Warning: Depth image not found: {depth_image_path}")
                else:
                    # Create depth symlink
                    depth_symlink_path = depth_images_dir / new_image_name
                    
                    # Remove existing symlink if it exists
                    if depth_symlink_path.exists() or depth_symlink_path.is_symlink():
                        depth_symlink_path.unlink()
                    
                    # Create new depth symbolic link
                    try:
                        depth_symlink_path.symlink_to(original_depth_path.absolute())
                        processed_depth_images.add(group_key)
                    except Exception as e:
                        print(f"Error creating depth symlink for {depth_image_path}: {e}")
                        continue
                    
                    # Prepare depth JSON annotation (same as RGB but with depth paths)
                    depth_json_data = {
                        "image_path": f"images/{new_image_name}",
                        "original_path": str(original_depth_path.absolute()),
                        "scene_id": scene_id,
                        "camera_type": camera_type,
                        "height": annotations[0]['image_height'],
                        "width": annotations[0]['image_width'],
                        "masks": []
                    }
                    
                    # Copy masks from RGB
                    depth_json_data["masks"] = rgb_json_data["masks"].copy()
                    
                    # Save depth JSON annotation file
                    depth_json_path = depth_annotations_dir / json_filename
                    
                    with open(depth_json_path, 'w') as f:
                        json.dump(depth_json_data, f, indent=2)
        
        print(f"Processing complete!")
        print(f"Processed {len(processed_rgb_images)} unique RGB image groups")
        print(f"Created RGB symbolic links in: {rgb_images_dir}")
        print(f"Created RGB annotations in: {rgb_annotations_dir}")
        
        if self.use_depth:
            print(f"Processed {len(processed_depth_images)} unique depth image groups")
            print(f"Created depth symbolic links in: {depth_images_dir}")
            print(f"Created depth annotations in: {depth_annotations_dir}")
        
        return output_path

    def verify_symlinks(self, output_dir=None):
        """
        Verify that all symbolic links are valid and point to existing files.
        
        Args:
            output_dir: The training data directory to verify
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        output_path = Path(output_dir)
        
        # Check RGB directories
        rgb_images_dir = output_path / "rgb" / "images"
        
        if not rgb_images_dir.exists():
            print(f"RGB images directory not found: {rgb_images_dir}")
        else:
            print("RGB Images:")
            rgb_valid, rgb_broken = self._verify_directory_symlinks(rgb_images_dir)
            print(f"  Valid links: {rgb_valid}")
            print(f"  Broken links: {rgb_broken}")
        
        # Check depth directories if use_depth is True
        if self.use_depth:
            depth_images_dir = output_path / "depth" / "images"
            
            if not depth_images_dir.exists():
                print(f"Depth images directory not found: {depth_images_dir}")
            else:
                print("\nDepth Images:")
                depth_valid, depth_broken = self._verify_directory_symlinks(depth_images_dir)
                print(f"  Valid links: {depth_valid}")
                print(f"  Broken links: {depth_broken}")
    
    def _verify_directory_symlinks(self, directory):
        """Helper method to verify symlinks in a directory"""
        valid_links = 0
        broken_links = 0
        
        for symlink in directory.iterdir():
            if symlink.is_symlink():
                if symlink.exists():
                    valid_links += 1
                else:
                    broken_links += 1
                    print(f"  Broken symlink: {symlink} -> {symlink.resolve()}")
            else:
                print(f"  Not a symlink: {symlink}")
        
        return valid_links, broken_links