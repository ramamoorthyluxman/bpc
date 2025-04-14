#!/usr/bin/env python3
"""
Custom Dataset Loader for segmentation dataset with the following structure:
- images/ folder containing image files
- annotations/ folder containing JSON files with mask information
- masks/ folder containing mask PNG files

Each annotation JSON contains:
- image_path: relative path to the image
- height, width: dimensions of the image
- masks: list of mask objects with label, points, and mask_path
"""

import os
import json
import numpy as np
from PIL import Image
import cv2

class CustomSegmentationDataset:
    """
    Dataset for instance segmentation with polygons and mask images.
    """
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with images, annotations, and masks folders
        """
        self.root_dir = root_dir
        
        # Get all annotation files
        self.annotations_dir = os.path.join(root_dir, 'annotations')
        self.annotation_files = sorted([f for f in os.listdir(self.annotations_dir) 
                                      if f.endswith('.json')])
        
        # Collect all unique class labels
        self.classes = set()
        for ann_file in self.annotation_files:
            with open(os.path.join(self.annotations_dir, ann_file), 'r') as f:
                ann_data = json.load(f)
                for mask in ann_data['masks']:
                    self.classes.add(mask['label'])
        
        self.classes = sorted(list(self.classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Loaded dataset with {len(self.annotation_files)} samples")
        print(f"Found {len(self.classes)} classes: {self.classes}")
    
    def __len__(self):
        return len(self.annotation_files)
    
    def get_item(self, idx):
        """
        Returns a dictionary containing:
            - image: The RGB image
            - masks: List of binary masks for each instance
            - labels: List of class names for each instance
            - polygons: List of polygon points for each instance
            - image_id: Original filename
        """
        # Load annotation
        ann_file = self.annotation_files[idx]
        with open(os.path.join(self.annotations_dir, ann_file), 'r') as f:
            ann_data = json.load(f)
        
        # Load image
        img_path = os.path.join(self.root_dir, ann_data['image_path'])
        image = np.array(Image.open(img_path).convert('RGB'))
        height, width = image.shape[:2]
        
        # Initialize empty lists
        masks = []
        labels = []
        polygons = []
        
        # Process each mask
        for mask_info in ann_data['masks']:
            # Get label
            label = mask_info['label']
            labels.append(label)
            
            # Get polygon points
            points = mask_info['points']
            polygons.append(points)
            
            # Load mask image if it exists, otherwise create from polygon
            mask_path = os.path.join(self.root_dir, mask_info['mask_path'])
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert('L'))
                mask = (mask > 0).astype(np.uint8)
            else:
                # Create mask from polygon points
                mask = np.zeros((height, width), dtype=np.uint8)
                pts = np.array(points, dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 1)
            
            masks.append(mask)
        
        return {
            'image': image,
            'masks': masks,
            'labels': labels,
            'polygons': polygons,
            'image_id': os.path.splitext(ann_file)[0]
        }

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    
    # Change this to your dataset path
    dataset = CustomSegmentationDataset(root_dir='/home/rama/bpc_ws/bpc/sam/data/sam_dataset')
    
    if len(dataset) > 0:
        # Load a sample
        sample = dataset.get_item(0)
        
        # Show image with annotations
        plt.figure(figsize=(10, 8))
        plt.imshow(sample['image'])
        
        # Draw polygons on image
        for i, (points, label) in enumerate(zip(sample['polygons'], sample['labels'])):
            if points:
                poly = np.array(points)
                color = plt.cm.tab10(i % 10)
                
                # Draw polygon
                p = Polygon(poly, fill=False, edgecolor=color, linewidth=2)
                plt.gca().add_patch(p)
                
                # Add label
                plt.text(poly[0][0], poly[0][1], label, 
                        fontsize=8, color='white', 
                        bbox=dict(facecolor=color, alpha=0.8))
        
        plt.title(f"Sample: {sample['image_id']}")
        plt.axis('off')
        plt.show()
        
        # Show masks
        num_masks = len(sample['masks'])
        if num_masks > 0:
            fig, axes = plt.subplots(1, min(num_masks, 5), figsize=(15, 3))
            if num_masks == 1:
                axes = [axes]  # Make it iterable
            
            for i, (mask, label) in enumerate(zip(sample['masks'], sample['labels'])):
                if i >= 5:  # Show at most 5 masks
                    break
                axes[i].imshow(mask, cmap='gray')
                axes[i].set_title(f"Mask: {label}")
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
    else:
        print("Dataset is empty!")