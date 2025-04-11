"""
Dataset class for SAM fine-tuning with fixed 1024x1024 image size.
"""

import os
import json
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional

class SAMDataset(Dataset):
    def __init__(
        self, 
        data_dir: str,
        split: str = "train",
        split_ratio: float = 0.8,
        transform = None,
        num_points: int = 1,
        point_selection: str = "center",  # "center", "random", or "bbox"
        image_size: int = 1024,
        seed: int = 42
    ):
        """
        SAM dataset for fine-tuning.
        
        Args:
            data_dir: Path to processed dataset directory
            split: "train" or "val"
            split_ratio: Ratio of train/val split
            transform: Optional transform to apply to images
            num_points: Number of points to use per mask
            point_selection: How to select points from masks ("center", "random", or "bbox")
            image_size: Size of the square images for SAM (must be 1024 for vit_h)
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.num_points = num_points
        self.point_selection = point_selection
        self.image_size = image_size
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Load dataset info
        with open(os.path.join(data_dir, "dataset_info.json"), 'r') as f:
            self.dataset_info = json.load(f)
        
        # Get all annotation files
        self.annotations_dir = os.path.join(data_dir, "annotations")
        annotation_files = [f for f in os.listdir(self.annotations_dir) if f.endswith(".json")]
        
        # Split dataset
        random.shuffle(annotation_files)
        split_idx = int(len(annotation_files) * split_ratio)
        
        if split == "train":
            self.annotation_files = annotation_files[:split_idx]
        elif split == "val":
            self.annotation_files = annotation_files[split_idx:]
        else:
            raise ValueError(f"Invalid split: {split}, must be 'train' or 'val'")
        
        print(f"Loaded {len(self.annotation_files)} samples for {split}")
    
    def __len__(self) -> int:
        return len(self.annotation_files)
    
    def preprocess_image(self, image: np.ndarray, target_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image to exact target size (square).
        Returns the processed image and the transformation matrix.
        """
        # Get original dimensions
        h, w = image.shape[:2]
        
        # Calculate scale to fit within target_size
        scale = min(target_size / h, target_size / w)
        
        # Calculate new dimensions
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create square image with padding
        square_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        
        # Place resized image in center
        square_img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # Create transformation matrix for points
        transform_matrix = np.array([
            [scale, 0, pad_w],
            [0, scale, pad_h]
        ])
        
        return square_img, transform_matrix
    
    def transform_point(self, point: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Transform point using the transformation matrix."""
        # Convert to homogeneous coordinates
        point_h = np.array([point[0], point[1], 1])
        
        # Apply transformation
        transformed = transform_matrix @ point_h
        
        return transformed[:2]
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset."""
        # Load annotation
        annotation_file = self.annotation_files[idx]
        with open(os.path.join(self.annotations_dir, annotation_file), 'r') as f:
            annotation = json.load(f)
        
        # Load image
        image_path = os.path.join(self.data_dir, annotation["image_path"])
        original_image = np.array(Image.open(image_path).convert("RGB"))
        
        # Image dimensions
        h, w = annotation["height"], annotation["width"]
        original_size = (h, w)
        
        # Preprocess image to exact square size (1024x1024)
        processed_image, transform_matrix = self.preprocess_image(original_image, self.image_size)
        
        # Convert to tensor (and normalize to [0, 1])
        image_tensor = torch.from_numpy(processed_image.transpose(2, 0, 1)).float() / 255.0
        
        # Load masks and prepare data
        masks = []
        points = []
        labels = []
        
        for mask_info in annotation["masks"]:
            # Load mask
            mask_path = os.path.join(self.data_dir, mask_info["mask_path"])
            original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            original_mask = original_mask > 0  # Convert to binary
            
            # Get points based on selection method
            selected_points = self._select_points(original_mask, mask_info["points"], h, w)
            
            # Transform points using the transformation matrix
            transformed_points = np.array([self.transform_point(p, transform_matrix) for p in selected_points])
            
            # Transform mask to match the processed image
            mask_processed = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            
            # Get original mask dimensions
            mask_h, mask_w = original_mask.shape[:2]
            
            # Calculate new dimensions
            scale = min(self.image_size / h, self.image_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize mask
            mask_resized = cv2.resize(
                original_mask.astype(np.uint8), 
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Calculate padding
            pad_h = (self.image_size - new_h) // 2
            pad_w = (self.image_size - new_w) // 2
            
            # Place resized mask in center
            mask_processed[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = mask_resized
            
            masks.append(mask_processed)
            points.append(transformed_points)
            labels.append(mask_info["label"])
        
        # Convert lists to arrays/tensors
        masks = np.array(masks, dtype=np.float32)
        
        # Create sample dictionary
        sample = {
            "image": image_tensor,
            "masks": torch.from_numpy(masks).float(),
            "points": [torch.from_numpy(p).float() for p in points],
            "labels": labels,
            "image_size": original_size,
            "file_name": annotation_file
        }
        
        # Apply additional transforms if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _select_points(self, mask: np.ndarray, polygon_points: List, h: int, w: int) -> np.ndarray:
        """
        Select points to use as prompt for SAM.
        
        Args:
            mask: Binary mask array
            polygon_points: Original polygon points from annotation
            h, w: Image height and width
            
        Returns:
            ndarray: Selected points as [N, 2] array (x, y)
        """
        if self.point_selection == "center":
            # Use the center of mass of the mask
            indices = np.where(mask > 0)
            if len(indices[0]) == 0:  # Empty mask
                return np.zeros((self.num_points, 2))
                
            y_center = np.mean(indices[0])
            x_center = np.mean(indices[1])
            points = np.array([[x_center, y_center]])
            
        elif self.point_selection == "random":
            # Select random points inside the mask
            indices = np.where(mask > 0)
            if len(indices[0]) == 0:  # Empty mask
                return np.zeros((self.num_points, 2))
                
            random_idx = np.random.choice(len(indices[0]), size=min(self.num_points, len(indices[0])), replace=False)
            points = np.array([[indices[1][i], indices[0][i]] for i in random_idx])
            
        elif self.point_selection == "bbox":
            # Use corners of the bounding box
            indices = np.where(mask > 0)
            if len(indices[0]) == 0:  # Empty mask
                return np.zeros((self.num_points, 2))
                
            y_min, y_max = np.min(indices[0]), np.max(indices[0])
            x_min, x_max = np.min(indices[1]), np.max(indices[1])
            
            bbox_points = [
                [x_min, y_min],  # Top-left
                [x_max, y_min],  # Top-right
                [x_max, y_max],  # Bottom-right
                [x_min, y_max]   # Bottom-left
            ]
            points = np.array(bbox_points[:self.num_points])
            
        else:
            raise ValueError(f"Invalid point selection method: {self.point_selection}")
        
        # Ensure we have exactly num_points points (pad with zeros if necessary)
        if len(points) < self.num_points:
            padding = np.zeros((self.num_points - len(points), 2))
            points = np.vstack([points, padding])
        
        return points[:self.num_points]


class SAMTransform:
    """
    Transform for SAM dataset.
    Ensures all tensors are in the correct format.
    """
    def __call__(self, sample: Dict) -> Dict:
        # Already handled in __getitem__
        return sample


def visualize_sample(sample: Dict, figsize: Tuple[int, int] = (15, 10)):
    """
    Visualize a sample from the dataset.
    
    Args:
        sample: Sample dictionary from dataset
        figsize: Figure size for visualization
    """
    image = sample["image"]
    masks = sample["masks"]
    points = sample["points"]
    labels = sample["labels"]
    
    # Convert tensors to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    
    n_masks = len(masks)
    fig, axes = plt.subplots(n_masks, 3, figsize=figsize)
    
    # Handle single mask case
    if n_masks == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_masks):
        mask = masks[i]
        
        # Original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Image with Points ({labels[i]})")
        axes[i, 0].axis('off')
        
        # Plot points
        if isinstance(points[i], torch.Tensor):
            pts = points[i].cpu().numpy()
        else:
            pts = points[i]
            
        for p in pts:
            if np.all(p == 0):  # Skip zero points
                continue
            axes[i, 0].scatter(p[0], p[1], c='red', s=40, marker='*')
        
        # Mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f"Mask ({labels[i]})")
        axes[i, 1].axis('off')
        
        # Overlay
        axes[i, 2].imshow(image)
        axes[i, 2].imshow(mask, alpha=0.5, cmap='jet')
        axes[i, 2].set_title(f"Overlay ({labels[i]})")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def create_dataloaders(
    data_dir: str,
    batch_size: int = 1,
    num_workers: int = 4,
    split_ratio: float = 0.8,
    num_points: int = 1,
    point_selection: str = "center",
    image_size: int = 1024,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        data_dir: Path to processed dataset directory
        batch_size: Batch size
        num_workers: Number of workers for dataloaders
        split_ratio: Ratio of train/val split
        num_points: Number of points to use per mask
        point_selection: How to select points from masks
        image_size: Size of square images for SAM (must be 1024 for vit_h)
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    # Create transforms
    transform = SAMTransform()
    
    # Create datasets
    train_dataset = SAMDataset(
        data_dir=data_dir,
        split="train",
        split_ratio=split_ratio,
        transform=transform,
        num_points=num_points,
        point_selection=point_selection,
        image_size=image_size,
        seed=seed
    )
    
    val_dataset = SAMDataset(
        data_dir=data_dir,
        split="val",
        split_ratio=split_ratio,
        transform=transform,
        num_points=num_points,
        point_selection=point_selection,
        image_size=image_size,
        seed=seed
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SAM dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to processed dataset directory")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to visualize")
    parser.add_argument("--point_selection", type=str, default="center", choices=["center", "random", "bbox"],
                        help="Point selection method")
    parser.add_argument("--num_points", type=int, default=1, help="Number of points per mask")
    parser.add_argument("--image_size", type=int, default=1024, help="Image size for SAM (must be 1024 for vit_h)")
    
    args = parser.parse_args()
    
    # Create dataset
    dataset = SAMDataset(
        data_dir=args.data_dir,
        split="train",
        transform=SAMTransform(),
        num_points=args.num_points,
        point_selection=args.point_selection,
        image_size=args.image_size
    )
    
    # Visualize samples
    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Masks shape: {sample['masks'].shape}")
        print(f"  Number of points: {len(sample['points'])}")
        print(f"  Labels: {sample['labels']}")
        print(f"  Image size: {sample['image_size']}")
        
        visualize_sample(sample)