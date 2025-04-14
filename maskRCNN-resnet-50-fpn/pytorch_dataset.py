#!/usr/bin/env python3
"""
PyTorch Dataset for custom segmentation data format.

This script creates a PyTorch Dataset class that can be used with DataLoader
for training segmentation models.

Usage example:
    from pytorch_dataset import SegmentationDataset
    
    train_dataset = SegmentationDataset(dataset_dir='path/to/dataset', 
                                        transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import cv2

class SegmentationDataset(Dataset):
    """PyTorch dataset for custom segmentation data."""
    
    def __init__(self, dataset_dir, transform=None, target_transform=None):
        """
        Args:
            dataset_dir (str): Path to dataset directory
            transform (callable, optional): Transform to apply to images
            target_transform (callable, optional): Transform to apply to masks
        """
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Get annotation files
        annotations_dir = os.path.join(dataset_dir, 'annotations')
        self.annotation_files = sorted([
            f for f in os.listdir(annotations_dir) 
            if f.endswith('.json')
        ])
        
        # Get class mapping
        self.classes = set()
        for ann_file in self.annotation_files:
            with open(os.path.join(annotations_dir, ann_file), 'r') as f:
                annotation = json.load(f)
                for mask in annotation.get('masks', []):
                    self.classes.add(mask.get('label', 'unknown'))
        
        self.classes = sorted(list(self.classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Dataset initialized with {len(self.annotation_files)} samples")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.annotation_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            dict: Contains 'image', 'masks', 'labels', 'image_id'
        """
        annotations_dir = os.path.join(self.dataset_dir, 'annotations')
        ann_file = self.annotation_files[idx]
        
        # Load annotation
        with open(os.path.join(annotations_dir, ann_file), 'r') as f:
            annotation = json.load(f)
        
        # Load image
        image_path = os.path.join(self.dataset_dir, annotation['image_path'])
        image = Image.open(image_path).convert('RGB')
        
        # Get image dimensions
        width, height = image.size
        
        # Initialize masks and labels
        instance_masks = []
        class_labels = []
        
        # Process each mask
        for mask_info in annotation.get('masks', []):
            # Get class label
            label = mask_info.get('label', 'unknown')
            label_idx = self.class_to_idx[label]
            class_labels.append(label_idx)
            
            # Get mask from file or create from polygon
            mask_path = os.path.join(self.dataset_dir, mask_info.get('mask_path', ''))
            
            if os.path.exists(mask_path):
                # Load mask from file
                mask = np.array(Image.open(mask_path).convert('L'))
                mask = (mask > 0).astype(np.uint8)
            else:
                # Create mask from polygon points
                points = mask_info.get('points', [])
                mask = np.zeros((height, width), dtype=np.uint8)
                
                if points and len(points) >= 3:
                    pts = np.array(points, dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 1)
            
            instance_masks.append(mask)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform to tensor
            image = T.ToTensor()(image)
        
        # Convert masks to tensor
        if instance_masks:
            masks_tensor = torch.as_tensor(np.stack(instance_masks), dtype=torch.uint8)
            
            if self.target_transform:
                masks_tensor = self.target_transform(masks_tensor)
        else:
            # Empty tensor if no masks
            masks_tensor = torch.zeros((0, height, width), dtype=torch.uint8)
        
        # Convert class labels to tensor
        labels_tensor = torch.as_tensor(class_labels, dtype=torch.int64)
        
        return {
            'image': image,
            'masks': masks_tensor,
            'labels': labels_tensor,
            'image_id': os.path.splitext(ann_file)[0]
        }

def get_transforms(train=True, target_size=(512, 512)):
    """
    Get default transforms for training or evaluation.
    
    Args:
        train (bool): Whether to use training or validation transforms
        target_size (tuple): Target size for resizing images
        
    Returns:
        transforms: Composition of transforms
    """
    if train:
        return T.Compose([
            T.Resize(target_size),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_data_loaders(dataset_dir, batch_size=8, val_split=0.2, num_workers=4):
    """
    Create training and validation data loaders.
    
    Args:
        dataset_dir (str): Path to dataset directory
        batch_size (int): Batch size for training
        val_split (float): Fraction of data to use for validation
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets with different transforms
    full_dataset = SegmentationDataset(
        dataset_dir=dataset_dir,
        transform=get_transforms(train=True)
    )
    
    # Split into train and validation sets
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # For validation set, we want to use a different transform
    val_dataset.dataset = SegmentationDataset(
        dataset_dir=dataset_dir,
        transform=get_transforms(train=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    This handles batches with varying numbers of instances per image.
    """
    return tuple(batch)

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PyTorch Dataset')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory')
    args = parser.parse_args()
    
    # Create dataset
    dataset = SegmentationDataset(
        dataset_dir=args.dataset_dir,
        transform=T.Compose([
            T.Resize((384, 384)),
            T.ToTensor()
        ])
    )
    
    if len(dataset) > 0:
        # Display a sample
        idx = 0
        sample = dataset[idx]
        
        image = sample['image']
        masks = sample['masks']
        labels = sample['labels']
        
        # Convert tensors to numpy for visualization
        image_np = image.permute(1, 2, 0).numpy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Show image
        axes[0].imshow(image_np)
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        # Show masks
        combined_mask = np.zeros((*masks.shape[1:], 3))
        colors = plt.cm.tab10(np.linspace(0, 1, len(dataset.classes)))
        
        for i, (mask, label_idx) in enumerate(zip(masks, labels)):
            color = colors[label_idx.item()][:3]
            for c in range(3):
                combined_mask[:, :, c] += mask.numpy() * color[c] * 0.7
        
        # Normalize combined mask
        if combined_mask.max() > 0:
            combined_mask /= combined_mask.max()
        
        axes[1].imshow(combined_mask)
        axes[1].set_title('Masks')
        axes[1].axis('off')
        
        plt.suptitle(f"Sample {idx}: {sample['image_id']}")
        plt.tight_layout()
        plt.show()
    else:
        print("Dataset is empty!")