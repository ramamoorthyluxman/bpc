import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from torchvision import transforms, models
import os
import argparse
from tqdm import tqdm

class PoseDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        image_path = row['image_path']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract rotation matrix from CSV
        R = np.array([
            [row['r11'], row['r12'], row['r13']],
            [row['r21'], row['r22'], row['r23']],
            [row['r31'], row['r32'], row['r33']]
        ], dtype=np.float32)
        
        # Convert rotation matrix to quaternion
        quaternion = rotation_matrix_to_quaternion(R)
        
        # Crop image to mask region
        image = self.crop_to_mask(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, quaternion
    
    def crop_to_mask(self, image):
        # Create binary mask from non-black pixels
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = gray > 0
        
        # Find bounding box of mask
        coords = np.column_stack(np.where(mask))
        if len(coords) == 0:
            # If no mask found, return center crop
            h, w = image.shape[:2]
            size = min(h, w) // 2
            center_y, center_x = h // 2, w // 2
            y1 = max(0, center_y - size)
            y2 = min(h, center_y + size)
            x1 = max(0, center_x - size)
            x2 = min(w, center_x + size)
            return image[y1:y2, x1:x2]
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add small padding
        padding = 10
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        y_max = min(image.shape[0], y_max + padding)
        x_max = min(image.shape[1], x_max + padding)
        
        # Crop image
        cropped = image[y_min:y_max, x_min:x_max]
        
        return cropped

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]"""
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z], dtype=np.float32)

class PoseEstimationNet(nn.Module):
    def __init__(self):
        super(PoseEstimationNet, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Remove final classification layer
        self.backbone.fc = nn.Identity()
        
        # Add pose regression head
        self.pose_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4)  # Output quaternion [w, x, y, z]
        )
        
    def forward(self, x):
        features = self.backbone(x)
        quaternion = self.pose_head(features)
        
        # Normalize to unit quaternion
        quaternion = quaternion / torch.norm(quaternion, dim=1, keepdim=True)
        
        return quaternion

def quaternion_geodesic_loss(q_pred, q_gt):
    """Geodesic loss on quaternion manifold"""
    # Ensure quaternions are unit quaternions
    q_pred = q_pred / torch.norm(q_pred, dim=1, keepdim=True)
    q_gt = q_gt / torch.norm(q_gt, dim=1, keepdim=True)
    
    # Compute dot product
    dot_product = torch.abs(torch.sum(q_pred * q_gt, dim=1))
    
    # Clamp to avoid numerical issues
    dot_product = torch.clamp(dot_product, 0.0, 1.0)
    
    # Geodesic distance
    loss = 1.0 - dot_product
    
    return loss.mean()

def train_model(csv_path, model_save_path, num_epochs=100, batch_size=16, lr=1e-4):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = PoseDataset(csv_path, transform=transform)
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = PoseEstimationNet().to(device)
    
    # Loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (images, quaternions) in enumerate(train_pbar):
            images = images.to(device)
            quaternions = quaternions.to(device)
            
            optimizer.zero_grad()
            
            pred_quaternions = model(images)
            loss = quaternion_geodesic_loss(pred_quaternions, quaternions)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for batch_idx, (images, quaternions) in enumerate(val_pbar):
                images = images.to(device)
                quaternions = quaternions.to(device)
                
                pred_quaternions = model(images)
                loss = quaternion_geodesic_loss(pred_quaternions, quaternions)
                
                val_loss += loss.item()
                val_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, model_save_path)
            print(f'Saved best model with validation loss: {val_loss:.6f}')
    
    print(f'Training completed. Best validation loss: {best_val_loss:.6f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train pose estimation network')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--model_path', type=str, default='pose_model.pth', help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(args.csv_path, args.model_path, args.epochs, args.batch_size, args.lr)