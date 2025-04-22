import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import random
from tqdm import tqdm

class SAMDataset(Dataset):
    """
    Dataset class for loading custom dataset with polygon masks for SAM fine-tuning.
    """
    def __init__(
        self,
        data_dir: str,
        transform=None,
        image_size: Tuple[int, int] = (1024, 1024),
        num_points: int = 1,
        point_selection: str = "center",
        create_masks: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to dataset directory containing 'images' and 'annotations' folders
            transform: Transform to apply to images and masks
            image_size: Size to resize images to
            num_points: Number of points to use per mask
            point_selection: Method for selecting points ("center", "random", "bbox")
            create_masks: Whether to generate mask images from polygons if they don't exist
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_size = image_size
        self.num_points = num_points
        self.point_selection = point_selection
        self.create_masks = create_masks
        
        # Get annotation files
        self.annotations_dir = os.path.join(data_dir, "annotations")
        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "masks")
        
        if not os.path.exists(self.annotations_dir):
            raise ValueError(f"Annotations directory not found: {self.annotations_dir}")
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory not found: {self.images_dir}")
        
        # Create masks directory if it doesn't exist and we're generating masks
        if self.create_masks and not os.path.exists(self.masks_dir):
            os.makedirs(self.masks_dir)
        
        # Load all annotations
        self.annotation_files = [f for f in os.listdir(self.annotations_dir) if f.endswith(".json")]
        self.data = []
        
        # Process annotations
        for ann_file in tqdm(self.annotation_files, desc="Loading annotations"):
            ann_path = os.path.join(self.annotations_dir, ann_file)
            try:
                with open(ann_path, "r") as f:
                    annotation = json.load(f)
                
                # Get image path - handle both image_path and imagePath keys
                image_path = annotation.get("image_path") or annotation.get("imagePath")
                if not image_path:
                    print(f"Warning: No image path found in {ann_file}, skipping")
                    continue
                
                # Make sure the path is relative to the data directory
                if "images/" in image_path:
                    image_filename = image_path.split("images/")[-1]
                else:
                    image_filename = os.path.basename(image_path)
                
                # Create full image path
                full_image_path = os.path.join(self.images_dir, image_filename)
                if not os.path.exists(full_image_path):
                    print(f"Warning: Image not found: {full_image_path}, skipping")
                    continue
                
                # Store data
                self.data.append({
                    "annotation_file": ann_file,
                    "image_path": full_image_path,
                    "height": annotation.get("height"),
                    "width": annotation.get("width"),
                    "masks": annotation.get("masks", [])
                })
                
                # Generate mask images if needed
                if self.create_masks:
                    for i, mask_info in enumerate(annotation.get("masks", [])):
                        # Skip if mask_path is not provided
                        mask_path = mask_info.get("mask_path")
                        if not mask_path:
                            # Generate mask path
                            mask_filename = f"{os.path.splitext(image_filename)[0]}_{i}.png"
                            mask_path = os.path.join("masks", mask_filename)
                            mask_info["mask_path"] = mask_path
                            
                            # Update the annotation file with the new mask path
                            annotation["masks"][i]["mask_path"] = mask_path
                            with open(ann_path, "w") as f:
                                json.dump(annotation, f, indent=2)
                        
                        # Create full mask path
                        full_mask_path = os.path.join(self.data_dir, mask_path)
                        
                        # Generate mask image if it doesn't exist
                        if not os.path.exists(full_mask_path) and "points" in mask_info:
                            points = mask_info["points"]
                            if len(points) >= 3:  # Need at least 3 points for a polygon
                                self._generate_mask(
                                    points, 
                                    annotation.get("height"), 
                                    annotation.get("width"), 
                                    full_mask_path
                                )
            except Exception as e:
                print(f"Error processing {ann_file}: {e}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get item from dataset.
        
        Args:
            idx: Index of item
            
        Returns:
            dict: Dictionary containing image, masks, points, labels, etc.
        """
        item = self.data[idx]
        
        # Load image
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            print(f"Error loading image {item['image_path']}: {e}")
            # Return a random valid item
            return self.__getitem__(random.randint(0, len(self.data) - 1))
        
        # Get original image dimensions
        if item["height"] and item["width"]:
            orig_height, orig_width = item["height"], item["width"]
        else:
            orig_width, orig_height = image.size
        
        # Initialize masks tensor and points list
        masks = []
        points_list = []
        labels = []
        
        # Process each mask
        for mask_idx, mask_info in enumerate(item["masks"]):
            # Get mask path
            mask_path = mask_info.get("mask_path")
            if not mask_path:
                continue
            
            full_mask_path = os.path.join(self.data_dir, mask_path)
            
            # Load or generate mask
            if os.path.exists(full_mask_path):
                try:
                    mask = Image.open(full_mask_path).convert("L")
                except Exception as e:
                    print(f"Error loading mask {full_mask_path}: {e}")
                    continue
            elif "points" in mask_info and len(mask_info["points"]) >= 3:
                # Generate mask from points if it doesn't exist
                points = mask_info["points"]
                mask_np = self._create_mask_from_points(points, orig_height, orig_width)
                mask = Image.fromarray(mask_np.astype(np.uint8) * 255)
                
                # Save mask if create_masks is enabled
                if self.create_masks:
                    os.makedirs(os.path.dirname(full_mask_path), exist_ok=True)
                    mask.save(full_mask_path)
            else:
                continue
            
            # Get label
            label = mask_info.get("label", "object")
            
            # Select points based on the method
            if "points" in mask_info and len(mask_info["points"]) >= 3:
                selected_points = self._select_points(
                    mask_info["points"],
                    self.point_selection,
                    self.num_points,
                    mask_np if "mask_np" in locals() else None
                )
            else:
                # If no valid points, select center point or use default
                mask_np = np.array(mask)
                if np.any(mask_np > 0):
                    # Find center of mass
                    y_indices, x_indices = np.where(mask_np > 0)
                    center_x = int(np.mean(x_indices))
                    center_y = int(np.mean(y_indices))
                    selected_points = [[center_x, center_y]]
                else:
                    # Default to center of image
                    selected_points = [[orig_width // 2, orig_height // 2]]
            
            # Add to lists
            masks.append(np.array(mask))
            points_list.append(selected_points)
            labels.append(label)
        
        # Handle case with no valid masks
        if not masks:
            # Create a dummy mask and point
            empty_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
            masks = [empty_mask]
            points_list = [[[orig_width // 2, orig_height // 2]]]
            labels = ["background"]
        
        # Resize image
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        
        # Resize masks
        resized_masks = []
        for mask in masks:
            mask_pil = Image.fromarray(mask)
            resized_mask = mask_pil.resize(self.image_size, Image.Resampling.NEAREST)
            resized_masks.append(np.array(resized_mask))
        
        # Resize points
        resized_points = []
        for points in points_list:
            scale_x = self.image_size[0] / orig_width
            scale_y = self.image_size[1] / orig_height
            scaled_points = [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in points]
            resized_points.append(scaled_points)
        
        # Convert to tensors
        image_tensor = transforms.ToTensor()(image)
        mask_tensors = torch.as_tensor(np.stack(resized_masks), dtype=torch.float32) / 255.0
        point_tensors = [torch.tensor(points, dtype=torch.float32) for points in resized_points]
        
        # Apply transforms if provided
        if self.transform is not None:
            image_tensor, mask_tensors, point_tensors = self.transform(
                image_tensor, mask_tensors, point_tensors
            )
        
        return {
            "image": image_tensor,
            "masks": mask_tensors,
            "points": point_tensors[0] if point_tensors else torch.zeros((1, 2)),
            "labels": labels,
            "image_path": item["image_path"],
            "image_size": (orig_height, orig_width)
        }
    
    def _generate_mask(self, points, height, width, mask_path):
        """
        Generate mask image from polygon points.
        
        Args:
            points: List of [x, y] points
            height: Image height
            width: Image width
            mask_path: Path to save mask
        """
        # Create mask
        mask = self._create_mask_from_points(points, height, width)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        
        # Save mask
        cv2.imwrite(mask_path, mask * 255)
    
    def _create_mask_from_points(self, points, height, width):
        """
        Create binary mask from polygon points.
        
        Args:
            points: List of [x, y] points
            height: Image height
            width: Image width
            
        Returns:
            mask: Binary mask
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Convert points to the format expected by fillPoly
        if len(points) < 3:
            return mask
            
        pts = np.array(points, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Fill polygon with white (1)
        cv2.fillPoly(mask, [pts], 1)
        
        return mask
    
    def _select_points(self, polygon_points, method, num_points, mask=None):
        """
        Select points from polygon for prompting.
        
        Args:
            polygon_points: List of polygon points
            method: Method for selecting points ('center', 'random', 'bbox')
            num_points: Number of points to select
            mask: Binary mask image
            
        Returns:
            points: Selected points
        """
        if method == "center":
            # Calculate centroid
            points_array = np.array(polygon_points)
            center_x = int(np.mean(points_array[:, 0]))
            center_y = int(np.mean(points_array[:, 1]))
            
            # Use centroid as the point
            return [[center_x, center_y]]
            
        elif method == "random":
            # Get a random subset of points
            if len(polygon_points) <= num_points:
                return polygon_points
            else:
                return random.sample(polygon_points, num_points)
                
        elif method == "bbox":
            # Calculate bounding box
            points_array = np.array(polygon_points)
            x_min, y_min = np.min(points_array, axis=0)
            x_max, y_max = np.max(points_array, axis=0)
            
            # Use center of bounding box
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)
            
            return [[center_x, center_y]]
            
        else:
            # Default to a single point at the center
            points_array = np.array(polygon_points)
            center_x = int(np.mean(points_array[:, 0]))
            center_y = int(np.mean(points_array[:, 1]))
            
            return [[center_x, center_y]]
            
    def visualize_sample(self, idx):
        """
        Visualize a sample from the dataset.
        
        Args:
            idx: Index of sample to visualize
        """
        sample = self[idx]
        
        # Get image and mask
        image = sample["image"]
        masks = sample["masks"]
        points = sample["points"]
        
        # Convert to numpy for visualization
        image_np = image.permute(1, 2, 0).numpy()
        
        # Iterate through masks and points
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Image with points
        axes[0].imshow(image_np)
        axes[0].set_title("Image with Points")
        
        for point in points:
            axes[0].scatter(point[0], point[1], c='red', s=40, marker='*')
            
        axes[0].axis("off")
        
        # Image with mask overlay
        axes[1].imshow(image_np)
        
        for i in range(masks.shape[0]):
            mask_np = masks[i].numpy()
            axes[1].imshow(mask_np, alpha=0.5, cmap="jet")
            
        axes[1].set_title("Image with Mask")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.show()


class SAMTransform:
    """
    Transforms for SAM dataset.
    """
    def __init__(self, train=True):
        """
        Initialize transforms.
        
        Args:
            train: Whether this is for training or validation
        """
        self.train = train
    
    def __call__(self, image, masks, points):
        """
        Apply transforms to image, masks, and points.
        
        Args:
            image: Image tensor [C, H, W]
            masks: Mask tensor [N, H, W]
            points: List of points tensors each [M, 2]
            
        Returns:
            image: Transformed image
            masks: Transformed masks
            points: Transformed points
        """
        # Convert tensors to numpy for easier manipulation
        image_np = image.permute(1, 2, 0).numpy()  # [H, W, C]
        masks_np = masks.numpy()  # [N, H, W]
        
        # Apply transforms for training
        if self.train:
            # Random horizontal flip
            if random.random() < 0.5:
                image_np = np.fliplr(image_np).copy()
                masks_np = np.fliplr(masks_np).copy()
                
                # Flip points
                width = image_np.shape[1]
                for i in range(len(points)):
                    points[i][:, 0] = width - points[i][:, 0]
            
            # Random vertical flip
            if random.random() < 0.5:
                image_np = np.flipud(image_np).copy()
                masks_np = np.flipud(masks_np).copy()
                
                # Flip points
                height = image_np.shape[0]
                for i in range(len(points)):
                    points[i][:, 1] = height - points[i][:, 1]
        
        # Convert back to tensors
        image = torch.from_numpy(image_np).permute(2, 0, 1)  # [C, H, W]
        masks = torch.from_numpy(masks_np).float()  # [N, H, W]
        
        # Normalize image
        image = image / 255.0
        
        return image, masks, points


def collate_fn(batch):
    """
    Collate function for SAM dataset.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        dict: Dictionary containing batched tensors
    """
    # Get individual items
    images = []
    masks = []
    points = []
    labels = []
    image_paths = []
    image_sizes = []
    
    for item in batch:
        images.append(item["image"])
        
        # Ensure mask has proper dimensions (C, H, W)
        mask = item["masks"]
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension
        masks.append(mask)
        
        points.append(item["points"])
        labels.append(item["labels"])
        image_paths.append(item["image_path"])
        image_sizes.append(item["image_size"])
    
    # Stack images - all should have the same size
    images = torch.stack(images)
    
    # Return dictionary
    return {
        "image": images,
        "masks": masks,  # List of tensors of varying sizes
        "points": points,  # List of tensors
        "labels": labels,  # List of strings
        "image_path": image_paths,  # List of strings
        "image_size": image_sizes  # List of tuples (h, w)
    }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 1,
    num_workers: int = 4,
    split_ratio: float = 0.8,
    num_points: int = 1,
    point_selection: str = "center",
    seed: int = 42
):
    """
    Create DataLoaders for training and validation.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size
        num_workers: Number of workers for data loading
        split_ratio: Train/validation split ratio
        num_points: Number of points to use per mask
        point_selection: Method for selecting points
        seed: Random seed
        
    Returns:
        train_loader, val_loader: DataLoaders for training and validation
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Create dataset
    full_dataset = SAMDataset(
        data_dir=data_dir,
        transform=None,  # Will be applied later
        image_size=(1024, 1024),
        num_points=num_points,
        point_selection=point_selection,
        create_masks=True
    )
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * split_ratio)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Loaded {len(train_dataset)} samples for train")
    print(f"Loaded {len(val_dataset)} samples for val")
    
    # Apply transforms
    train_dataset.dataset.transform = SAMTransform(train=True)
    val_dataset.dataset.transform = SAMTransform(train=False)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    return train_loader, val_loader