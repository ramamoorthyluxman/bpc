"""
SAM model preparation for fine-tuning.
Loads the pre-trained SAM model and modifies it for fine-tuning.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import argparse

class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Predicted logits, shape [B, 1, H, W]
            target (torch.Tensor): Target mask, shape [B, 1, H, W]
        """
        # Resize target to match prediction size if they differ
        if pred.shape != target.shape:
            target = F.interpolate(
                target, 
                size=pred.shape[2:],
                mode='nearest'
            )
        
        # Apply sigmoid to prediction
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Calculate Dice score
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        # Calculate Dice loss
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal loss for segmentation.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Predicted logits, shape [B, 1, H, W]
            target (torch.Tensor): Target mask, shape [B, 1, H, W]
        """
        # Resize target to match prediction size if they differ
        if pred.shape != target.shape:
            target = F.interpolate(
                target, 
                size=pred.shape[2:],
                mode='nearest'
            )
        
        # Apply sigmoid to prediction
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Calculate focal weights
        pt = torch.where(target_flat == 1, pred_flat, 1 - pred_flat)
        alpha_factor = torch.where(target_flat == 1, self.alpha, 1 - self.alpha)
        
        # Calculate focal loss
        focal_weight = alpha_factor * (1 - pt).pow(self.gamma)
        
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy(
            pred_flat, target_flat, reduction="none"
        )
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()


class SAMLoss(nn.Module):
    """
    Combined loss for SAM fine-tuning.
    """
    def __init__(self, dice_weight: float = 1.0, focal_weight: float = 1.0):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Predicted logits, shape [B, 1, H, W]
            target (torch.Tensor): Target mask, shape [B, 1, H, W]
        """
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        return self.dice_weight * dice + self.focal_weight * focal


class SAMForFineTuning(nn.Module):
    """
    SAM model modified for fine-tuning on custom datasets.
    """
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_h",
        device: str = "cuda",
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = False
    ):
        """
        Initialize SAM model for fine-tuning.
        
        Args:
            checkpoint_path: Path to SAM checkpoint
            model_type: SAM model type ("vit_h", "vit_l", "vit_b")
            device: Device to use
            freeze_image_encoder: Whether to freeze the image encoder
            freeze_prompt_encoder: Whether to freeze the prompt encoder
        """
        super().__init__()
        
        # Load SAM model
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device)
        
        # Set up which parts to freeze
        self.freeze_image_encoder = freeze_image_encoder
        self.freeze_prompt_encoder = freeze_prompt_encoder
        
        # Freeze image encoder if specified
        if freeze_image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
        
        # Freeze prompt encoder if specified
        if freeze_prompt_encoder:
            for param in self.sam.prompt_encoder.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        images: torch.Tensor,
        points: List[torch.Tensor],
        original_image_sizes: List[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass for fine-tuning.
        
        Args:
            images: Batch of images, shape [B, 3, H, W]
            points: List of points tensors, each with shape [N, 2]
            original_image_sizes: List of original image sizes (H, W)
            
        Returns:
            mask_predictions: Predicted masks, shape [B, 1, H, W]
            metrics: Additional metrics for analysis
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Get image embeddings
        with torch.set_grad_enabled(not self.freeze_image_encoder):
            image_embeddings = self.sam.image_encoder(images)
        
        # Initialize output for masks and metrics
        mask_predictions = []
        metrics = {"iou": [], "stability_score": []}
        
        # Process each image in the batch
        for b in range(batch_size):
            # Get points for this image
            point_coords = points[b]
            
            # Skip if no points or all zeros
            if point_coords.size(0) == 0 or torch.all(point_coords == 0):
                # Create empty mask prediction
                h, w = images.shape[2], images.shape[3]  # Use input image size
                empty_mask = torch.zeros((1, 1, h, w), device=device)
                mask_predictions.append(empty_mask)
                metrics["iou"].append(0.0)
                metrics["stability_score"].append(0.0)
                continue
            
            # Filter out zero points (padding) - reshaping to handle different dimensions
            point_coords = point_coords.reshape(-1, 2)
            valid_mask = ~torch.all(point_coords == 0, dim=1)
            valid_points = point_coords[valid_mask]
            
            if len(valid_points) == 0:
                # Create empty mask prediction if no valid points
                h, w = images.shape[2], images.shape[3]  # Use input image size
                empty_mask = torch.zeros((1, 1, h, w), device=device)
                mask_predictions.append(empty_mask)
                metrics["iou"].append(0.0)
                metrics["stability_score"].append(0.0)
                continue
            
            # Prepare point coordinates and labels
            point_coords = valid_points.unsqueeze(0)  # [1, N, 2]
            point_labels = torch.ones(1, point_coords.shape[1], device=device)  # [1, N]
            
            # Get prompt embeddings
            with torch.set_grad_enabled(not self.freeze_prompt_encoder):
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None,
                    masks=None
                )
            
            # Get low-res mask prediction
            low_res_masks, iou_predictions = self.sam.mask_decoder(
                image_embeddings=image_embeddings[b:b+1],
                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            
            # Use input image size instead of original size
            input_size = images.shape[-2:]
            
            # Upsample masks to input image size
            masks = self.sam.postprocess_masks(
                low_res_masks,
                input_size=input_size,
                original_size=input_size
            )
            
            mask_predictions.append(masks)
            metrics["iou"].append(iou_predictions.item())
            metrics["stability_score"].append(0.0)  # Not computed during training
        
        # Concatenate masks
        if mask_predictions:
            mask_predictions = torch.cat(mask_predictions, dim=0)
        else:
            # Create empty prediction if no valid masks
            h, w = images.shape[2], images.shape[3]  # Use input image size
            mask_predictions = torch.zeros((batch_size, 1, h, w), device=device)
        
        return mask_predictions, metrics


def create_sam_model(
    checkpoint_path: str,
    model_type: str = "vit_h",
    device: str = "cuda",
    freeze_image_encoder: bool = True,
    freeze_prompt_encoder: bool = False
) -> SAMForFineTuning:
    """
    Create SAM model for fine-tuning.
    
    Args:
        checkpoint_path: Path to SAM checkpoint
        model_type: SAM model type ("vit_h", "vit_l", "vit_b")
        device: Device to use
        freeze_image_encoder: Whether to freeze the image encoder
        freeze_prompt_encoder: Whether to freeze the prompt encoder
        
    Returns:
        sam_model: SAM model for fine-tuning
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}")
    
    # Create SAM model
    sam_model = SAMForFineTuning(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device,
        freeze_image_encoder=freeze_image_encoder,
        freeze_prompt_encoder=freeze_prompt_encoder
    )
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in sam_model.parameters())
    
    print(f"Created SAM model with {trainable_params:,} trainable parameters out of {total_params:,} total parameters")
    print(f"Freeze image encoder: {freeze_image_encoder}")
    print(f"Freeze prompt encoder: {freeze_prompt_encoder}")
    
    return sam_model


if __name__ == "__main__":
    # Test model creation
    parser = argparse.ArgumentParser(description="Test SAM model creation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM checkpoint")
    parser.add_argument("--model_type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"],
                        help="SAM model type")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    
    args = parser.parse_args()
    
    # Create SAM model
    sam_model = create_sam_model(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device=args.device
    )
    
    # Test with random input
    device = torch.device(args.device)
    images = torch.randn(2, 3, 1024, 1024).to(device)
    points = [
        torch.tensor([[500, 500]], dtype=torch.float).to(device),
        torch.tensor([[400, 400]], dtype=torch.float).to(device)
    ]
    original_image_sizes = [(512, 512), (512, 512)]
    
    # Forward pass
    with torch.no_grad():
        masks, metrics = sam_model(images, points, original_image_sizes)
    
    print(f"Output mask shape: {masks.shape}")
    print(f"IoU predictions: {metrics['iou']}")
    
    # Test SAM loss
    print("Testing SAM loss...")
    criterion = SAMLoss()
    target = torch.randint(0, 2, (2, 1, 512, 512)).float().to(device)
    loss = criterion(masks, target)
    print(f"Loss value: {loss.item()}")