import os
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from pycocotools import mask as coco_mask


class MechanicalPartsDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Custom dataset for mechanical parts with masks
        
        Args:
            root_dir (str): Root directory containing 'images' and 'annotations' folders
            split (str): 'train' or 'val' split
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Paths
        self.images_dir = os.path.join(root_dir, 'images')
        self.annotations_dir = os.path.join(root_dir, 'annotations')
        
        # Get list of all annotation files
        self.annotation_files = [f for f in os.listdir(self.annotations_dir) if f.endswith('.json')]
        
        # Load all annotations
        self.annotations = []
        for ann_file in self.annotation_files:
            with open(os.path.join(self.annotations_dir, ann_file), 'r') as f:
                self.annotations.append(json.load(f))
        
        # Extract unique category names from annotations
        self.categories = ['background']  # Always include background as first class (id 0)
        category_set = set()
        
        # Process all annotations to extract unique categories
        for ann in self.annotations:
            for mask_ann in ann.get('masks', []):
                category = mask_ann.get('label')
                if category and category not in category_set and category != 'background':
                    category_set.add(category)
        
        # Add sorted categories to the list (after background)
        self.categories.extend(sorted(category_set))
        self.cat_to_id = {cat: i for i, cat in enumerate(self.categories)}
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Get dataset item with image and target annotations"""
        ann = self.annotations[idx]
        
        # Load image
        img_path = os.path.join(self.root_dir, ann['image_path'])
        image = Image.open(img_path).convert('RGB')
        
        # Image dimensions
        width, height = image.size
        if width != ann['width'] or height != ann['height']:
            # Resize if necessary to match annotations
            image = image.resize((ann['width'], ann['height']))
        
        # Create target dictionary
        target = {}
        target["boxes"] = []
        target["labels"] = []
        target["masks"] = []
        target["image_id"] = torch.tensor([idx])
        target["area"] = []
        target["iscrowd"] = []
        
        # Process each mask
        for mask_ann in ann['masks']:
            # Get category id
            label = mask_ann.get('label', 'bracket')  # Default to bracket if not specified
            if label not in self.cat_to_id:
                # Skip unknown categories
                continue
                
            cat_id = self.cat_to_id[label]
            
            # Create binary mask from polygon points
            points = mask_ann['points']
            if len(points) < 3:  # Skip invalid polygons
                continue
                
            # Create mask image
            mask_img = Image.new('L', (ann['width'], ann['height']), 0)
            draw = ImageDraw.Draw(mask_img)
            flat_points = [item for sublist in points for item in sublist]
            draw.polygon(flat_points, fill=1)
            mask = np.array(mask_img)
            
            # Calculate bounding box
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue  # Skip empty masks
                
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # Skip invalid boxes
            if x_max <= x_min or y_max <= y_min:
                continue
                
            # Calculate area
            area = float((x_max - x_min) * (y_max - y_min))
            
            # Add to target
            target["boxes"].append([x_min, y_min, x_max, y_max])
            target["labels"].append(cat_id)
            target["masks"].append(mask)
            target["area"].append(area)
            target["iscrowd"].append(0)  # Assuming all instances are not crowds
        
        # Convert lists to tensors
        if len(target["boxes"]) == 0:
            # Create empty target with correct shapes if no valid masks
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0), dtype=torch.int64)
            target["masks"] = torch.zeros((0, ann['height'], ann['width']), dtype=torch.uint8)
            target["area"] = torch.zeros((0), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0), dtype=torch.int64)
        else:
            target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
            target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
            target["masks"] = torch.as_tensor(np.stack(target["masks"]), dtype=torch.uint8)
            target["area"] = torch.as_tensor(target["area"], dtype=torch.float32)
            target["iscrowd"] = torch.as_tensor(target["iscrowd"], dtype=torch.int64)
        
        # Apply transforms if specified
        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            # Default transform to tensor
            image = T.ToTensor()(image)
            # No need to modify target here
        
        return image, target
    
    def export_categories(self, output_path='categories.txt'):
        """
        Export category names to a text file for inference
        
        Args:
            output_path (str): Path to save the categories file
        
        Returns:
            None
        """
        # Skip the background class (id 0)
        categories = self.categories[1:]
        
        try:
            with open(output_path, 'w') as f:
                for category in categories:
                    f.write(f"{category}\n")
            
            print(f"Exported {len(categories)} categories to {output_path}")
        except Exception as e:
            print(f"Error exporting categories: {e}")


class Compose:
    """Custom Compose for transformations that handles both image and target"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class CustomTransform:
    """Base class for custom transforms that process both image and target"""
    def __call__(self, image, target):
        return image, target


class ToTensor(CustomTransform):
    """Convert PIL image to tensor"""
    def __call__(self, image, target):
        image = T.ToTensor()(image)
        return image, target


class RandomHorizontalFlip(CustomTransform):
    """Horizontally flip image and target with probability p"""
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            # Check if image is PIL or Tensor and flip accordingly
            if isinstance(image, torch.Tensor):
                # For tensor (C, H, W), flip the last dimension (width)
                image = torch.flip(image, [2])
            else:
                # For PIL image
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Flip boxes
            if "boxes" in target and len(target["boxes"]) > 0:
                # Get image width (handle both PIL and tensor)
                if isinstance(image, torch.Tensor):
                    width = image.shape[2]
                else:
                    width = image.width
                
                boxes = target["boxes"].clone()  # Clone to avoid modifying original
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target["boxes"] = boxes
                
            # Flip masks
            if "masks" in target and len(target["masks"]) > 0:
                masks = target["masks"]
                target["masks"] = torch.flip(masks, [2])  # Flip along width dimension
                
        return image, target


def get_transform(train):
    """
    Get the transforms to apply to the dataset
    
    Args:
        train (bool): Whether this is for training or not
    """
    transforms = []
    # Converts PIL image to tensor and scales values to [0, 1]
    transforms.append(ToTensor())
    
    if train:
        # Add training-specific augmentations here
        transforms.append(RandomHorizontalFlip(0.5))
        # Optional: add additional augmentations specific to mechanical parts
    
    return Compose(transforms)


def collate_fn(batch):
    """
    Custom collate function for data loader
    
    Args:
        batch: batch of samples from dataset
    """
    return tuple(zip(*batch))




# Example usage
if __name__ == "__main__":
    # Test the dataset class
    import sys
    
    # Use current directory as the default dataset path
    dataset_path = "."  # Assuming the script is run from the directory containing images and annotations
    
    # Allow command line override
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Create dataset with transforms
    dataset = MechanicalPartsDataset(dataset_path, transform=get_transform(train=True))
    
    print(f"Dataset loaded with {len(dataset)} samples")

    # Export categories to file
    print("Exporting the categories file.")
    dataset.export_categories('categories.txt')
    
    if len(dataset) > 0:
        # Get a sample
        image, target = dataset[0]
        
        # Print info about the sample
        print(f"Image shape: {image.shape}")
        print(f"Number of objects: {len(target['boxes'])}")
        print(f"Object labels: {target['labels']}")
        print(f"Bounding boxes: {target['boxes']}")
        
        # Create DataLoader
        from torch.utils.data import DataLoader
        data_loader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        # Test batch loading
        print("\nTesting DataLoader...")
        for batch_idx, (images, targets) in enumerate(data_loader):
            print(f"Batch {batch_idx}: {len(images)} images, {len(targets)} targets")
            if batch_idx >= 1:  # Just check the first 2 batches
                break
    else:
        print("No samples found in the dataset. Please check your dataset path and structure.")