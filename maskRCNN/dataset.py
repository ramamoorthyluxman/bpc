import os
import json
import numpy as np
import cv2
from torch.utils.data import Dataset
from pycocotools import mask as maskUtils

class MechanicalPartsDataset(Dataset):
    """Dataset class for mechanical parts like brackets, bolts, nuts, etc."""
    
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Args:
            root_dir (string): Directory with images and annotations folders
            transform (callable, optional): Optional transform to be applied on a sample
            is_train (bool): Flag to indicate training or validation set
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        # Get the image and annotation paths
        self.images_dir = os.path.join(root_dir, 'images')
        self.annotations_dir = os.path.join(root_dir, 'annotations')
        
        # Get all image filenames
        self.image_files = [f for f in os.listdir(self.images_dir) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Create a dictionary to map class names to IDs
        self.class_dict = self._get_class_dict()
        
    def _get_class_dict(self):
        """Create a dictionary that maps class names to class IDs."""
        class_dict = {'background': 0}
        class_id = 1  # Start from 1 as 0 is for background
        
        # Scan all annotation files to find unique classes
        for img_file in self.image_files:
            json_file = os.path.splitext(img_file)[0] + '.json'
            json_path = os.path.join(self.annotations_dir, json_file)
            
            if not os.path.exists(json_path):
                continue
                
            with open(json_path, 'r') as f:
                annotation = json.load(f)
            
            for mask in annotation.get('masks', []):
                label = mask.get('label')
                if label and label not in class_dict:
                    class_dict[label] = class_id
                    class_id += 1
        
        return class_dict
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a sample with image and annotations."""
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Load annotation
        json_file = os.path.splitext(self.image_files[idx])[0] + '.json'
        json_path = os.path.join(self.annotations_dir, json_file)
        
        # Initialize empty lists for masks, class_ids, and boxes
        masks = []
        class_ids = []
        boxes = []
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                annotation = json.load(f)
            
            # Assuming annotation has 'masks' key as shown in the example
            for mask_info in annotation.get('masks', []):
                label = mask_info.get('label')
                points = mask_info.get('points', [])
                
                if not points or label not in self.class_dict:
                    continue
                
                # Create binary mask from polygon points
                points = np.array(points, dtype=np.int32)
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask, [points], 1)
                
                # Get bounding box
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x1, y1 = np.min(x_indices), np.min(y_indices)
                    x2, y2 = np.max(x_indices), np.max(y_indices)
                    boxes.append([x1, y1, x2, y2])
                    masks.append(mask)
                    class_ids.append(self.class_dict[label])
        
        # Convert lists to numpy arrays
        if masks:
            masks = np.stack(masks, axis=0).astype(np.bool_)
            boxes = np.array(boxes, dtype=np.int32)
            class_ids = np.array(class_ids, dtype=np.int32)
        else:
            # No annotations found
            masks = np.zeros((0, height, width), dtype=np.bool_)
            boxes = np.zeros((0, 4), dtype=np.int32)
            class_ids = np.zeros(0, dtype=np.int32)
        
        # Create a sample dictionary
        sample = {
            'image': image,
            'masks': masks,
            'boxes': boxes,
            'class_ids': class_ids,
            'image_id': idx,
            'image_path': img_path,
            'height': height,
            'width': width
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

    def get_class_names(self):
        """Returns a list of class names ordered by class ID."""
        classes = ['background'] + [k for k, v in sorted(self.class_dict.items(), key=lambda item: item[1]) 
                                  if v > 0]  # Filter out 'background'
        return classes