import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from pycocotools import mask as mask_utils

class CustomSAMDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.transform = transform
        
        # Get all annotation files
        self.annotation_files = list(self.annotations_dir.glob('*.json'))
        print(f"Found {len(self.annotation_files)} annotation files")
        
        # Extract label set from annotations
        self.labels = set()
        for ann_file in self.annotation_files:
            with open(ann_file, 'r') as f:
                data = json.load(f)
                for shape in data.get('shapes', []):
                    self.labels.add(shape.get('label', ''))
        
        self.labels = sorted(list(self.labels))
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        print(f"Found {len(self.labels)} unique labels: {self.labels}")

    def __len__(self):
        return len(self.annotation_files)
    
    def __getitem__(self, idx):
        # Load annotation
        ann_file = self.annotation_files[idx]
        with open(ann_file, 'r') as f:
            annotation_data = json.load(f)
        
        # Get image path from annotation
        image_path = self.images_dir / annotation_data.get('imagePath', '')
        
        # If image_path doesn't exist, try to find it by matching the annotation filename
        if not image_path.exists():
            base_name = ann_file.stem
            possible_image_paths = [
                self.images_dir / f"{base_name}.jpg",
                self.images_dir / f"{base_name}.jpeg",
                self.images_dir / f"{base_name}.png"
            ]
            
            for path in possible_image_paths:
                if path.exists():
                    image_path = path
                    break
        
        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]
        
        # If imageHeight/Width are not in the annotation, use actual image dimensions
        image_height = annotation_data.get('imageHeight', h)
        image_width = annotation_data.get('imageWidth', w)
        
        # Create masks for each shape
        masks = []
        labels = []
        
        for shape in annotation_data.get('shapes', []):
            label = shape.get('label', '')
            points = shape.get('points', [])
            shape_type = shape.get('shape_type', 'polygon')
            
            if shape_type == 'polygon' and len(points) > 2:
                # Convert points to numpy array
                points = np.array(points, dtype=np.int32)
                
                # Create mask for this polygon
                mask = np.zeros((image_height, image_width), dtype=np.uint8)
                cv2.fillPoly(mask, [points], 1)
                
                masks.append(mask)
                labels.append(self.label_to_id[label])
        
        # Combine masks into a tensor
        if masks:
            masks = np.stack(masks, axis=0)
            labels = np.array(labels)
        else:
            # Create empty masks/labels if no annotations
            masks = np.zeros((0, image_height, image_width), dtype=np.uint8)
            labels = np.array([])
        
        # Apply transformations if any
        if self.transform:
            transformed = self.transform(image=image, masks=masks)
            image = transformed['image']
            masks = transformed['masks']
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        masks_tensor = torch.from_numpy(masks).float()
        labels_tensor = torch.from_numpy(labels).long()
        
        return {
            'image': image_tensor,
            'masks': masks_tensor,
            'labels': labels_tensor,
            'image_path': str(image_path)
        }


class SAMFineTuner(nn.Module):
    def __init__(self, model_type, checkpoint, freeze_image_encoder=True):
        super().__init__()
        # Initialize the SAM model
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        
        # Freeze the image encoder if specified
        if freeze_image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
        
        # We'll add a custom prediction head for class labels
        # This will be a simple MLP that takes the mask embeddings and predicts class labels
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # SAM's mask embeddings are 256-dim
            nn.ReLU(),
            nn.Linear(128, len(dataset.labels))
        )
    
    def forward(self, images, point_coords=None, point_labels=None, masks=None):
        # Get image embeddings
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(images)
        
        # Get mask predictions from the mask decoder
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=masks
        )
        
        mask_predictions, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        
        # Get mask embeddings
        mask_embedding_features = self.sam.mask_decoder.output_hypernetworks_mlps(sparse_embeddings)
        
        # Use the mask embeddings for classification
        class_predictions = self.classifier(mask_embedding_features)
        
        return mask_predictions, iou_predictions, class_predictions


def dice_loss(predictions, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        predictions: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    predictions = predictions.sigmoid()
    predictions = predictions.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (predictions * targets).sum(1)
    denominator = predictions.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


def train_sam(
    dataset_path,
    model_type="vit_b",
    checkpoint_path="sam_vit_b_01ec64.pth",
    output_dir="./fine_tuned_sam",
    batch_size=1,
    num_epochs=10,
    learning_rate=1e-5,
    freeze_image_encoder=True
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup dataset and dataloader
    images_dir = os.path.join(dataset_path, "images")
    annotations_dir = os.path.join(dataset_path, "annotations")
    
    # Initialize the dataset
    dataset = CustomSAMDataset(images_dir, annotations_dir)
    
    # Create a validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize the model
    model = SAMFineTuner(model_type, checkpoint_path, freeze_image_encoder)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define optimizer and loss functions
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    
    # Training loop
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mask_loss = 0.0
        train_cls_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in progress_bar:
            images = batch['image'].to(device)
            masks = batch['masks'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            # For simplicity, we're using the center of mass of each mask as a prompt point
            batch_size, num_masks, h, w = masks.shape
            point_coords = []
            point_labels = []
            
            for b in range(batch_size):
                for m in range(num_masks):
                    mask = masks[b, m].cpu().numpy()
                    if mask.sum() > 0:
                        # Find the center of mass of the mask
                        y_indices, x_indices = np.where(mask > 0)
                        x_center = x_indices.mean()
                        y_center = y_indices.mean()
                        point_coords.append([x_center, y_center])
                        point_labels.append(1)  # 1 for foreground
            
            if not point_coords:
                # Skip this batch if no masks
                continue
            
            point_coords = torch.tensor(point_coords, dtype=torch.float, device=device)
            point_labels = torch.tensor(point_labels, dtype=torch.int, device=device)
            
            # Reshape for batch processing
            point_coords = point_coords.unsqueeze(0)
            point_labels = point_labels.unsqueeze(0)
            
            # Forward pass
            mask_preds, iou_preds, cls_preds = model(images, point_coords, point_labels)
            
            # Compute losses
            mask_loss = dice_loss(mask_preds, masks)
            cls_loss = F.cross_entropy(cls_preds, labels)
            loss = mask_loss + cls_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running losses
            train_loss += loss.item()
            train_mask_loss += mask_loss.item()
            train_cls_loss += cls_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{train_loss / (progress_bar.n + 1):.4f}",
                'mask_loss': f"{train_mask_loss / (progress_bar.n + 1):.4f}",
                'cls_loss': f"{train_cls_loss / (progress_bar.n + 1):.4f}"
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mask_loss = 0.0
        val_cls_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in progress_bar:
                images = batch['image'].to(device)
                masks = batch['masks'].to(device)
                labels = batch['labels'].to(device)
                
                # Same point extraction as training
                batch_size, num_masks, h, w = masks.shape
                point_coords = []
                point_labels = []
                
                for b in range(batch_size):
                    for m in range(num_masks):
                        mask = masks[b, m].cpu().numpy()
                        if mask.sum() > 0:
                            y_indices, x_indices = np.where(mask > 0)
                            x_center = x_indices.mean()
                            y_center = y_indices.mean()
                            point_coords.append([x_center, y_center])
                            point_labels.append(1)
                
                if not point_coords:
                    continue
                
                point_coords = torch.tensor(point_coords, dtype=torch.float, device=device)
                point_labels = torch.tensor(point_labels, dtype=torch.int, device=device)
                
                point_coords = point_coords.unsqueeze(0)
                point_labels = point_labels.unsqueeze(0)
                
                # Forward pass
                mask_preds, iou_preds, cls_preds = model(images, point_coords, point_labels)
                
                # Compute losses
                mask_loss = dice_loss(mask_preds, masks)
                cls_loss = F.cross_entropy(cls_preds, labels)
                loss = mask_loss + cls_loss
                
                # Update running losses
                val_loss += loss.item()
                val_mask_loss += mask_loss.item()
                val_cls_loss += cls_loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'val_loss': f"{val_loss / (progress_bar.n + 1):.4f}",
                    'val_mask_loss': f"{val_mask_loss / (progress_bar.n + 1):.4f}",
                    'val_cls_loss': f"{val_cls_loss / (progress_bar.n + 1):.4f}"
                })
        
        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'labels': dataset.labels,
                'label_to_id': dataset.label_to_id
            }, os.path.join(output_dir, "best_model.pth"))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save checkpoint for this epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'labels': dataset.labels,
            'label_to_id': dataset.label_to_id
        }, os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth"))
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'labels': dataset.labels,
        'label_to_id': dataset.label_to_id
    }, os.path.join(output_dir, "final_model.pth"))
    
    print("Training complete!")
    return model


def visualize_predictions(model, dataset, num_samples=5, output_dir="./visualizations"):
    """Visualize model predictions on random samples from the dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    
    for idx in indices:
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        true_masks = sample['masks']
        labels = sample['labels']
        
        # Extract point prompts from the ground truth masks
        point_coords = []
        point_labels = []
        
        for m in range(true_masks.shape[0]):
            mask = true_masks[m].numpy()
            if mask.sum() > 0:
                y_indices, x_indices = np.where(mask > 0)
                x_center = x_indices.mean()
                y_center = y_indices.mean()
                point_coords.append([x_center, y_center])
                point_labels.append(1)
        
        if not point_coords:
            continue
        
        point_coords = torch.tensor(point_coords, dtype=torch.float, device=device)
        point_labels = torch.tensor(point_labels, dtype=torch.int, device=device)
        
        point_coords = point_coords.unsqueeze(0)
        point_labels = point_labels.unsqueeze(0)
        
        # Get model predictions
        with torch.no_grad():
            mask_preds, iou_preds, cls_preds = model(image, point_coords, point_labels)
        
        # Convert predictions to numpy for visualization
        pred_masks = mask_preds.sigmoid().cpu().numpy() > 0.5
        pred_classes = cls_preds.argmax(dim=1).cpu().numpy()
        
        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        img_np = image[0].cpu().numpy().transpose(1, 2, 0)
        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Ground truth masks
        axes[1].imshow(img_np)
        
        for m in range(true_masks.shape[0]):
            mask = true_masks[m].numpy()
            label = dataset.labels[labels[m].item()]
            
            # Create a colored overlay for this mask
            color = np.random.rand(3)
            mask_rgb = np.zeros((*mask.shape, 3))
            mask_rgb[mask > 0] = color
            
            axes[1].imshow(mask_rgb, alpha=0.5)
            
            # Mark the center of mass
            if mask.sum() > 0:
                y_indices, x_indices = np.where(mask > 0)
                x_center = x_indices.mean()
                y_center = y_indices.mean()
                axes[1].plot(x_center, y_center, 'ro', markersize=6)
                
                # Add label text
                axes[1].text(x_center, y_center, label, 
                            color='white', fontsize=8, 
                            bbox=dict(facecolor='black', alpha=0.7))
        
        axes[1].set_title("Ground Truth Masks")
        axes[1].axis('off')
        
        # Predicted masks
        axes[2].imshow(img_np)
        
        for m in range(pred_masks.shape[1]):
            mask = pred_masks[0, m]
            if m < len(pred_classes):
                label = dataset.labels[pred_classes[m]]
            else:
                label = "Unknown"
            
            # Create a colored overlay for this mask
            color = np.random.rand(3)
            mask_rgb = np.zeros((*mask.shape, 3))
            mask_rgb[mask > 0] = color
            
            axes[2].imshow(mask_rgb, alpha=0.5)
            
            # Mark the center of mass
            if mask.sum() > 0:
                y_indices, x_indices = np.where(mask > 0)
                x_center = x_indices.mean()
                y_center = y_indices.mean()
                axes[2].plot(x_center, y_center, 'ro', markersize=6)
                
                # Add label text
                axes[2].text(x_center, y_center, label, 
                            color='white', fontsize=8, 
                            bbox=dict(facecolor='black', alpha=0.7))
        
        axes[2].set_title("Predicted Masks")
        axes[2].axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"prediction_{idx}.png"))
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune SAM model on custom dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset containing images and annotations folders')
    parser.add_argument('--model_type', type=str, default='vit_b', choices=['vit_h', 'vit_l', 'vit_b'], help='SAM model type')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to SAM checkpoint file')
    parser.add_argument('--output_dir', type=str, default='./fine_tuned_sam', help='Output directory for fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze the image encoder')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions after training')
    parser.add_argument('--num_vis', type=int, default=5, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Train the model
    model = train_sam(
        dataset_path=args.dataset,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        freeze_image_encoder=args.freeze_encoder
    )
    
    # Visualize results if requested
    if args.visualize:
        # Re-initialize dataset to access all samples
        images_dir = os.path.join(args.dataset, "images")
        annotations_dir = os.path.join(args.dataset, "annotations")
        dataset = CustomSAMDataset(images_dir, annotations_dir)
        
        visualize_predictions(
            model=model,
            dataset=dataset,
            num_samples=args.num_vis,
            output_dir=os.path.join(args.output_dir, "visualizations")
        )