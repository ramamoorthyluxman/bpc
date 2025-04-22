"""
Training script for fine-tuning SAM on custom datasets.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Import local modules
from model import SAMForFineTuning, SAMLoss, create_sam_model
from dataset import SAMDataset, SAMTransform, create_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM model on custom dataset")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Path to processed dataset directory")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for checkpoints and logs")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to SAM checkpoint")
    parser.add_argument("--model_type", type=str, default="vit_h", 
                        choices=["vit_h", "vit_l", "vit_b"],
                        help="SAM model type")
    parser.add_argument("--freeze_image_encoder", action="store_true", 
                        help="Freeze image encoder")
    parser.add_argument("--freeze_prompt_encoder", action="store_true", 
                        help="Freeze prompt encoder")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of workers for data loading")
    parser.add_argument("--num_epochs", type=int, default=10, 
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, 
                        help="Weight decay")
    parser.add_argument("--eval_interval", type=int, default=1, 
                        help="Interval to evaluate model on validation set")
    parser.add_argument("--save_interval", type=int, default=1, 
                        help="Interval to save model checkpoint")
    
    # Data processing parameters
    parser.add_argument("--num_points", type=int, default=1, 
                        help="Number of points to use per mask")
    parser.add_argument("--point_selection", type=str, default="center", 
                        choices=["center", "random", "bbox"],
                        help="Point selection method")
    parser.add_argument("--split_ratio", type=float, default=0.8, 
                        help="Train/validation split ratio")
    
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def visualize_predictions(images, masks_gt, masks_pred, points, output_dir, epoch, batch_idx=0, num_samples=5):
    """Visualize predictions during training."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Convert tensors to numpy
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, C]
    
    if isinstance(masks_pred, torch.Tensor):
        masks_pred = masks_pred.detach().cpu().numpy()  # [B, 1, H, W]
    
    # Limit number of samples to visualize
    num_samples = min(num_samples, images.shape[0])
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    # Handle case when num_samples == 1
    if num_samples == 1:
        axes = np.array([axes])  # Ensure axes is 2D even with one sample
    
    for i in range(num_samples):
        # Get image and masks
        image = images[i]
        
        # Normalize image for visualization if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        # Get ground truth mask for this sample
        mask_gt = masks_gt[i][0].cpu().numpy() if i < len(masks_gt) else np.zeros_like(masks_pred[i][0])
        
        # Get predicted mask
        mask_pred = masks_pred[i][0]  # [H, W]
        
        # Convert mask_pred to binary using threshold of 0
        mask_pred_binary = (mask_pred > 0).astype(np.float32)
        
        # Image with points
        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Image with Points")
        axes[i, 0].axis("off")
        
        # Plot points if available
        if i < len(points):
            pts = points[i].cpu().numpy()
            axes[i, 0].scatter(pts[:, 0], pts[:, 1], c='red', s=40, marker='*')
        
        # Ground truth mask
        axes[i, 1].imshow(image)
        axes[i, 1].imshow(mask_gt, alpha=0.5, cmap="jet")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        
        # Predicted mask
        axes[i, 2].imshow(image)
        axes[i, 2].imshow(mask_pred_binary, alpha=0.5, cmap="jet")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_dir, "visualizations", f"epoch_{epoch}_batch_{batch_idx}.png")
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def calculate_iou(pred, target):
    """Calculate IoU for binary masks."""
    pred = pred > 0
    target = target > 0
    
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, output_dir):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    total_iou = 0
    count = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Get data
            images = batch["image"].to(device)
            points = [p.to(device) for p in batch["points"]]
            image_sizes = batch["image_size"]
            
            # Get first mask for each sample (assuming single instance per image)
            masks = []
            for m in batch["masks"]:
                # Ensure the mask has proper dimensions for the model
                if m.dim() == 3 and m.size(0) > 0:  # [N, H, W]
                    mask = m[0:1].to(device)  # Take first mask [1, H, W]
                elif m.dim() == 2:  # [H, W]
                    mask = m.unsqueeze(0).to(device)  # Add batch dimension [1, H, W] 
                else:
                    # Create a dummy mask filled with zeros
                    mask = torch.zeros(1, images.shape[2], images.shape[3], device=device)
                    
                masks.append(mask.unsqueeze(1))  # Add channel dimension [1, 1, H, W]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred_masks, metrics = model(images, points, image_sizes)
            
            # Calculate loss and backpropagate
            batch_loss = 0
            batch_iou = 0
            
            for i in range(images.size(0)):
                # Get ground truth mask
                mask_target = masks[i] if i < len(masks) else torch.zeros_like(pred_masks[i:i+1])
                
                # Get corresponding prediction
                mask_pred = pred_masks[i:i+1]  # [1, 1, H, W]
                
                # Ensure valid dimensions
                if mask_target.dim() != 4:
                    print(f"Warning: Target mask has unexpected shape: {mask_target.shape}")
                    mask_target = mask_target.view(1, 1, mask_target.size(0), mask_target.size(1))
                
                # Calculate loss
                try:
                    loss = criterion(mask_pred, mask_target)
                    batch_loss += loss
                    
                    # Calculate IoU
                    iou = calculate_iou(
                        mask_pred.detach().cpu().numpy() > 0, 
                        mask_target.cpu().numpy() > 0
                    )
                    batch_iou += iou
                except Exception as e:
                    print(f"Error calculating loss: {e}")
                    print(f"Pred shape: {mask_pred.shape}, Target shape: {mask_target.shape}")
                    # Skip this sample but continue training
                    continue
            
            # Average loss for the batch
            if images.size(0) > 0:
                batch_loss /= images.size(0)
                batch_iou /= images.size(0)
                
                # Backpropagate
                batch_loss.backward()
                optimizer.step()
                
                # Update metrics
                total_loss += batch_loss.item() * images.size(0)
                total_iou += batch_iou * images.size(0)
                count += images.size(0)
            
            # Update progress bar
            pbar.set_postfix(loss=batch_loss.item(), iou=batch_iou)
            
            # Visualize predictions occasionally
            if batch_idx % 50 == 0:
                visualize_predictions(
                    images, masks, pred_masks, points, 
                    output_dir, epoch, batch_idx
                )
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            # Continue with next batch
            continue
    
    # Calculate average metrics
    avg_loss = total_loss / count if count > 0 else 0
    avg_iou = total_iou / count if count > 0 else 0
    
    return avg_loss, avg_iou

def validate(model, dataloader, criterion, device, epoch, output_dir):
    """Validate model on validation set."""
    model.eval()
    total_loss = 0
    total_iou = 0
    count = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            try:
                # Get data
                images = batch["image"].to(device)
                points = [p.to(device) for p in batch["points"]]
                image_sizes = batch["image_size"]
                
                # Get first mask for each sample (assuming single instance per image)
                masks = []
                for m in batch["masks"]:
                    # Ensure the mask has proper dimensions for the model
                    if m.dim() == 3 and m.size(0) > 0:  # [N, H, W]
                        mask = m[0:1].to(device)  # Take first mask [1, H, W]
                    elif m.dim() == 2:  # [H, W]
                        mask = m.unsqueeze(0).to(device)  # Add batch dimension [1, H, W] 
                    else:
                        # Create a dummy mask filled with zeros
                        mask = torch.zeros(1, images.shape[2], images.shape[3], device=device)
                        
                    masks.append(mask.unsqueeze(1))  # Add channel dimension [1, 1, H, W]
                
                # Forward pass
                pred_masks, metrics = model(images, points, image_sizes)
                
                # Calculate loss and metrics
                batch_loss = 0
                batch_iou = 0
                
                for i in range(images.size(0)):
                    # Get ground truth mask
                    mask_target = masks[i] if i < len(masks) else torch.zeros_like(pred_masks[i:i+1])
                    
                    # Get corresponding prediction
                    mask_pred = pred_masks[i:i+1]  # [1, 1, H, W]
                    
                    # Ensure valid dimensions
                    if mask_target.dim() != 4:
                        print(f"Warning: Target mask has unexpected shape: {mask_target.shape}")
                        mask_target = mask_target.view(1, 1, mask_target.size(0), mask_target.size(1))
                    
                    # Calculate loss
                    try:
                        loss = criterion(mask_pred, mask_target)
                        batch_loss += loss
                        
                        # Calculate IoU
                        iou = calculate_iou(
                            mask_pred.cpu().numpy() > 0, 
                            mask_target.cpu().numpy() > 0
                        )
                        batch_iou += iou
                    except Exception as e:
                        print(f"Error calculating validation loss: {e}")
                        print(f"Pred shape: {mask_pred.shape}, Target shape: {mask_target.shape}")
                        # Skip this sample but continue validation
                        continue
                
                # Average loss for the batch
                if images.size(0) > 0:
                    batch_loss /= images.size(0)
                    batch_iou /= images.size(0)
                    
                    # Update metrics
                    total_loss += batch_loss.item() * images.size(0)
                    total_iou += batch_iou * images.size(0)
                    count += images.size(0)
                
                # Update progress bar
                pbar.set_postfix(loss=batch_loss.item(), iou=batch_iou)
                
                # Visualize predictions occasionally
                if batch_idx % 20 == 0:
                    visualize_predictions(
                        images, masks, pred_masks, points, 
                        output_dir, epoch, batch_idx
                    )
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                # Continue with next batch
                continue
    
    # Calculate average metrics
    avg_loss = total_loss / count if count > 0 else 0
    avg_iou = total_iou / count if count > 0 else 0
    
    return avg_loss, avg_iou

def train(args):
    """Main training function."""
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_ratio=args.split_ratio,
        num_points=args.num_points,
        point_selection=args.point_selection,
        seed=args.seed
    )
    
    # Create model
    device = torch.device(args.device)
    model = create_sam_model(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        freeze_image_encoder=args.freeze_image_encoder,
        freeze_prompt_encoder=args.freeze_prompt_encoder
    )
    model.to(device)
    
    # Create criterion, optimizer and scheduler
    criterion = SAMLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=True
    )
    
    # Initialize training variables
    start_epoch = 0
    best_loss = float("inf")
    best_iou = 0
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint.get("epoch", 0)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_loss = checkpoint.get("best_loss", float("inf"))
            best_iou = checkpoint.get("best_iou", 0)
            train_losses = checkpoint.get("train_losses", [])
            val_losses = checkpoint.get("val_losses", [])
            train_ious = checkpoint.get("train_ious", [])
            val_ious = checkpoint.get("val_ious", [])
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at: {args.resume}")
    
    # Training loop
    print(f"Starting training from epoch {start_epoch + 1} to {args.num_epochs}")
    for epoch in range(start_epoch, args.num_epochs):
        # Train for one epoch
        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1, output_dir
        )
        
        # Validate
        val_loss, val_iou = validate(
            model, val_loader, criterion, device, epoch + 1, output_dir
        )
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Check if this is the best model
        if val_iou > best_iou:
            best_iou = val_iou
            # Save best model
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "best_iou": best_iou,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_ious": train_ious,
                "val_ious": val_ious,
            }, os.path.join(output_dir, "best_model.pth"))
            print(f"Saved best model with IoU: {best_iou:.4f}")
        
        # Save metrics
        train_losses.append(train_loss)
        train_ious.append(train_iou)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        
        # Print metrics
        print(f"Epoch {epoch + 1}/{args.num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        
        # Save checkpoint if needed
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "best_iou": best_iou,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_ious": train_ious,
                "val_ious": val_ious,
            }, os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth"))
        
        # Plot and save metrics
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_ious, label="Train IoU")
        plt.plot(val_ious, label="Val IoU")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metrics.png"))
        plt.close()
    
    # Save final model
    torch.save({
        "epoch": args.num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
        "best_iou": best_iou,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_ious": train_ious,
        "val_ious": val_ious,
    }, os.path.join(output_dir, "final_model.pth"))
    
    print("Training completed!")
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    args = parse_args()
    train(args)