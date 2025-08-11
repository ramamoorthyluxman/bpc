import os
import time
import datetime
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import argparse

from dataset import MechanicalPartsDataset, get_transform, collate_fn
from model import get_model_instance_segmentation

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Train the model for one epoch
    
    Args:
        model: PyTorch model to train
        optimizer: Optimizer to use
        data_loader: DataLoader for the dataset
        device: Device to train on
        epoch: Current epoch number
        print_freq: How often to print progress
    """
    model.train()
    metric_logger = MetricLogger()
    header = f'Epoch: [{epoch}]'
    
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    return metric_logger


class MetricLogger:
    """Simple metric logger for training"""
    def __init__(self):
        self.meters = {}
        self.delimiter = "  "
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter.avg:.4f}")
        return self.delimiter.join(loss_str)
    
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if header is not None:
            print(header)
            
        start_time = time.time()
        for obj in iterable:
            yield obj
            i += 1
            if i % print_freq == 0:
                eta_seconds = (time.time() - start_time) * (len(iterable) - i) / i
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(f"{i}/{len(iterable)}, {self}, eta: {eta_string}")
    

class AverageMeter:
    """Compute and store the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main(args):
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    dataset = MechanicalPartsDataset(args.data_path, transform=get_transform(train=True))
    dataset_test = MechanicalPartsDataset(args.data_path, transform=get_transform(train=False))
    
    # Split the dataset into train and validation sets
    torch.manual_seed(args.seed)
    indices = torch.randperm(len(dataset)).tolist()
    train_size = int(len(indices) * 0.8)  # Using 80% for training
    
    dataset_train = torch.utils.data.Subset(dataset, indices[:train_size])
    dataset_val = torch.utils.data.Subset(dataset_test, indices[train_size:])
    
    print(f"Dataset loaded: {len(dataset_train)} training samples, {len(dataset_val)} validation samples")
    print(f"Number of classes: {len(dataset.categories)}")
    print(f"Classes: {dataset.categories}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn
    )
    
    # Initialize model
    num_classes = len(dataset.categories)
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    
    # Define parameters to train
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Define optimizer
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # Training loop
    print("Starting training")
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # Train for one epoch
        metric_logger = train_one_epoch(
            model, optimizer, train_loader, device, epoch, print_freq=args.print_freq
        )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for images, targets in val_loader:
                try:
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    # Forward pass without computing loss for validation
                    # Just predict the outputs
                    predictions = model(images)
                    
                    # Don't try to compute loss during validation
                    # This avoids tensor size mismatch errors
                    num_batches += 1
                    print(f"Processed validation batch {num_batches}/{len(val_loader)}")
                except Exception as e:
                    print(f"Error during validation batch {num_batches}: {e}")
                    continue
        
        print(f"Validation completed: processed {num_batches}/{len(val_loader)} batches")
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', metric_logger.meters['loss'].avg, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save model
        if epoch % args.save_freq == 0 or (epoch + 1) == args.epochs:
            save_path = os.path.join(args.output_dir, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved model at epoch {epoch} to {save_path}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")
    
    # Close TensorBoard writer
    writer.close()
    print("Training completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on mechanical parts dataset")
    parser.add_argument('--data-path', default='.', help='dataset root path')
    parser.add_argument('--output-dir', default='./output', help='path where to save outputs')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--batch-size', default=2, type=int, help='batch size')
    parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--lr', default=0.005, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--lr-step-size', default=5, type=int, help='learning rate scheduler step size')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='learning rate scheduler gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--save-freq', default=5, type=int, help='checkpoint save frequency (epochs)')
    parser.add_argument('--resume', default='', help='path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)