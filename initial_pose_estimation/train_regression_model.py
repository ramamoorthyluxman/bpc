import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
import copy

# Configure seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class PoseDataset(Dataset):
    def __init__(self, root_dir, object_id, transform=None, split='train', train_ratio=0.8):
        """
        Dataset for pose regression
        
        Args:
            root_dir: Root directory of the dataset
            object_id: Object ID as string (e.g., '000000')
            transform: Transforms to apply to images
            split: 'train' or 'val'
            train_ratio: Ratio of training data
        """
        self.root_dir = root_dir
        self.object_id = object_id
        self.transform = transform
        self.split = split
        
        # Get image paths
        self.image_dir = os.path.join(root_dir, object_id)
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        
        # Load rotation matrices
        rotation_file = os.path.join(root_dir, 'rotation_matrices', f"{object_id}.txt")
        self.rotations = self._load_rotations(rotation_file)
        
        # Check if rotations were loaded properly
        if len(self.rotations) == 0:
            raise ValueError(f"No rotation matrices found in {rotation_file}")
        
        # Check if rotations and images match
        if len(self.rotations) != len(self.image_files):
            raise ValueError(f"Number of rotation matrices ({len(self.rotations)}) does not match "
                             f"number of images ({len(self.image_files)})")
        
        # Split into train/val
        num_samples = len(self.image_files)
        indices = list(range(num_samples))
        split_idx = int(train_ratio * num_samples)
        
        if split == 'train':
            self.indices = indices[:split_idx]
        else:  # val
            self.indices = indices[split_idx:]
    
    def _load_rotations(self, file_path):
        """Load rotation matrices from file"""
        rotations = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                values = [float(val) for val in line.strip().split()]
                if len(values) == 9:
                    # Reshape into a 3x3 matrix
                    rotation = np.array(values).reshape(3, 3)
                    rotations.append(rotation)
        
        return rotations
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        true_idx = self.indices[idx]
        img_name = os.path.join(self.image_dir, self.image_files[true_idx])
        image = Image.open(img_name).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get rotation matrix and flatten
        rotation = self.rotations[true_idx].flatten()
        rotation = torch.tensor(rotation, dtype=torch.float32)
        
        return image, rotation

class PoseRegressor(nn.Module):
    def __init__(self, pretrained=True):
        """
        Neural network for pose regression
        
        Args:
            pretrained: Whether to use pretrained weights for the backbone
        """
        super(PoseRegressor, self).__init__()
        
        # Use ResNet50 as backbone
        resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Regression head for 9 values (flattened 3x3 rotation matrix)
        self.fc = nn.Linear(2048, 9)
        
        # Initialize the last layer with smaller weights
        nn.init.xavier_uniform_(self.fc.weight, gain=0.01)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def rotation_matrix_loss(pred, target):
    """
    Custom loss function for rotation matrices
    
    Combines:
    1. MSE loss for direct regression
    2. Orthogonality constraint (rotation matrices should be orthogonal)
    3. Determinant constraint (determinant should be 1)
    """
    # MSE loss
    mse_loss = nn.MSELoss()(pred, target)
    
    # Reshape to batch of 3x3 matrices
    batch_size = pred.size(0)
    pred_matrices = pred.view(batch_size, 3, 3)
    target_matrices = target.view(batch_size, 3, 3)
    
    # Orthogonality constraint: R * R^T should be identity
    # Calculate R * R^T for predicted matrices
    pred_transpose = torch.transpose(pred_matrices, 1, 2)
    orthogonality = torch.bmm(pred_matrices, pred_transpose)
    
    # Identity matrix
    identity = torch.eye(3, device=pred.device).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Orthogonality loss is the difference from identity
    orthogonality_loss = nn.MSELoss()(orthogonality, identity)
    
    # Determinant constraint: det(R) should be 1
    # Calculate determinants
    det_loss = 0
    for i in range(batch_size):
        det = torch.det(pred_matrices[i])
        det_loss += (det - 1.0) ** 2
    det_loss /= batch_size
    
    # Combine losses with weights
    total_loss = mse_loss + 0.1 * orthogonality_loss + 0.1 * det_loss
    
    return total_loss

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25):
    """
    Train the model
    
    Args:
        model: The neural network
        dataloaders: Dictionary with 'train' and 'val' dataloaders
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run on (cuda/cpu)
        num_epochs: Number of training epochs
    """
    since = time.time()
    
    # Keep track of training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                # Track history only in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            # Store loss history
            history[f'{phase}_loss'].append(epoch_loss)
            
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
            # Deep copy the model if best validation loss
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'object_id': args.object_id,
                }, os.path.join(args.output_dir, f'{args.object_id}_best_model.pth'))
                print(f'Saved new best model with loss: {best_loss:.4f}')
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Plot the loss history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss History for Object {args.object_id}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, f'{args.object_id}_loss_history.png'))
    
    return model, history

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Create datasets
    datasets = {
        'train': PoseDataset(args.data_dir, args.object_id, transform=data_transforms['train'], split='train'),
        'val': PoseDataset(args.data_dir, args.object_id, transform=data_transforms['val'], split='val')
    }
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        'val': DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    }
    
    print(f"Training with {len(datasets['train'])} samples, validating with {len(datasets['val'])} samples")
    
    # Initialize model
    model = PoseRegressor(pretrained=True)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = rotation_matrix_loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train the model
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=args.epochs
    )
    
    # Save the final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': history['val_loss'][-1],
        'object_id': args.object_id,
    }, os.path.join(args.output_dir, f'{args.object_id}_final_model.pth'))
    
    print(f"Model saved to {os.path.join(args.output_dir, f'{args.object_id}_final_model.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a pose regression model')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Root directory of the dataset')
    parser.add_argument('--object_id', type=str, required=True,
                        help='Object ID to train on')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for regularization')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    
    args = parser.parse_args()
    main(args)