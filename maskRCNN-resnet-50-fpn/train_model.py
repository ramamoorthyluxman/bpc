#!/usr/bin/env python3
"""
Train a segmentation model on the custom dataset.

This script implements training and validation loops for a Mask R-CNN model
from torchvision, which is suitable for instance segmentation.

Usage:
    python train_model.py --dataset_dir path/to/dataset --output_dir path/to/save/model
"""

import os
import argparse
import time
import datetime
import json
import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Import the custom dataset
from pytorch_dataset import SegmentationDataset, get_transforms, collate_fn

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--dataset_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', required=True, help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--img_size', type=int, default=512, help='Image size for training')
    return parser.parse_args()

def get_model(num_classes, pretrained=True):
    """
    Get a Mask R-CNN model with a ResNet-50-FPN backbone.
    
    Args:
        num_classes (int): Number of classes (including background)
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        model: Mask R-CNN model
    """
    # Load Mask R-CNN model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights='DEFAULT' if pretrained else None,
        progress=True
    )
    
    # Get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get the number of input features for mask prediction
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        optimizer: The optimizer
        data_loader: DataLoader for training data
        device: Device to train on
        epoch: Current epoch number
        print_freq: How often to print progress
        
    Returns:
        loss_dict: Dictionary of losses
    """
    model.train()
    
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    
    # Update learning rate
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    
    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = []
        targets = []
        
        # Process batch
        for b in batch:
            image = b['image']
            image = image.to(device)
            images.append(image)
            
            # Create target dict in the format expected by the model
            target = {}
            target['boxes'] = get_boxes_from_masks(b['masks']).to(device)
            target['labels'] = b['labels'].to(device)
            target['masks'] = b['masks'].to(device)
            targets.append(target)
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())
        
        # Reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        loss_value = losses_reduced.item()
        
        if not np.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            return
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(model, data_loader, device):
    """
    Evaluate the model on the validation set.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for validation data
        device: Device to evaluate on
        
    Returns:
        evaluation metrics
    """
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Validation:'
    
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 100, header):
            images = []
            targets = []
            
            # Process batch
            for b in batch:
                image = b['image']
                image = image.to(device)
                images.append(image)
                
                # Create target dict
                target = {}
                target['boxes'] = get_boxes_from_masks(b['masks']).to(device)
                target['labels'] = b['labels'].to(device)
                target['masks'] = b['masks'].to(device)
                targets.append(target)
            
            # Forward pass
            model_time = time.time()
            outputs = model(images)
            model_time = time.time() - model_time
            
            # Compute mAP
            evaluator_time = time.time()
            # Note: In a real-world scenario, you would use COCO evaluation metrics here
            # For simplicity, we're just logging loss values
            evaluator_time = time.time() - evaluator_time
            
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
    
    # Return metrics
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def get_boxes_from_masks(masks):
    """
    Get bounding boxes from masks.
    
    Args:
        masks: Tensor of shape [N, H, W] where N is the number of masks
        
    Returns:
        boxes: Tensor of shape [N, 4] with boxes in (x1, y1, x2, y2) format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    
    n = masks.shape[0]
    boxes = torch.zeros((n, 4), dtype=torch.float32)
    
    for i in range(n):
        mask = masks[i]
        pos = torch.where(mask > 0)
        
        if pos[0].numel() > 0:
            boxes[i, 0] = torch.min(pos[1]).float()  # x1
            boxes[i, 1] = torch.min(pos[0]).float()  # y1
            boxes[i, 2] = torch.max(pos[1]).float()  # x2
            boxes[i, 3] = torch.max(pos[0]).float()  # y2
    
    return boxes

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a window."""
    
    def __init__(self, window_size=20, fmt=None):
        self.deque = []
        self.total = 0.0
        self.count = 0
        self.window_size = window_size
        self.fmt = fmt
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
        
        if len(self.deque) > self.window_size:
            self.total -= self.deque.pop(0)
            self.count -= 1
    
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()
    
    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    
    @property
    def global_avg(self):
        return self.total / self.count
    
    @property
    def value(self):
        return self.deque[-1] if self.deque else None
    
    def __str__(self):
        if self.fmt is None:
            if self.value is None:
                return ""
            return f"{self.value:.4f} ({self.avg:.4f})"
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            value=self.value
        )

class MetricLogger:
    """Log metrics during training and evaluation."""
    
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(v)
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if header is not None:
            print(header)
        
        start_time = time.time()
        end = start_time
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        
        # Calculate width for progress format
        num_digits = len(str(len(iterable)))
        
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                print(
                    f"{header} [{i:>{num_digits}}/{len(iterable)}] "
                    f"eta: {eta_string} "
                    f"time: {iter_time} "
                    f"data: {data_time} "
                    f"{str(self)}"
                )
            
            i += 1
            end = time.time()
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")

def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that all processes
    have the average results.
    
    Args:
        input_dict (dict): Dictionary of values to be reduced
        average (bool): Whether to average or sum the values
        
    Returns:
        dict: Reduced dictionary
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return input_dict
    
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        
        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)
        
        if average:
            values /= world_size
        
        reduced_dict = {k: v for k, v in zip(names, values)}
    
    return reduced_dict

def save_checkpoint(model, optimizer, epoch, args, filename='checkpoint.pth'):
    """Save model checkpoint."""
    checkpoint_path = os.path.join(args.output_dir, filename)
    
    # Make a dictionary with model state and optimizer state
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def main():
    args = parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets and data loaders
    print("Creating data loaders...")
    
    # Training dataset
    train_dataset = SegmentationDataset(
        dataset_dir=args.dataset_dir,
        transform=get_transforms(train=True, target_size=(args.img_size, args.img_size))
    )
    
    # Validation dataset
    val_dataset = SegmentationDataset(
        dataset_dir=args.dataset_dir,
        transform=get_transforms(train=False, target_size=(args.img_size, args.img_size))
    )
    
    # Use a subset for validation (20% of the data)
    dataset_size = len(train_dataset)
    indices = torch.randperm(dataset_size).tolist()
    val_split = int(dataset_size * 0.2)
    
    train_subset = torch.utils.data.Subset(train_dataset, indices[val_split:])
    val_subset = torch.utils.data.Subset(val_dataset, indices[:val_split])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    print(f"Created data loaders: {len(train_subset)} training samples, {len(val_subset)} validation samples")
    
    # Create model
    num_classes = len(train_dataset.classes) + 1  # +1 for background
    model = get_model(num_classes, pretrained=args.pretrained)
    model.to(device)
    
    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Define learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        else:
            print(f"No checkpoint found at: {args.resume}")
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Save the class mapping
    class_mapping = {i: cls for i, cls in enumerate(train_dataset.classes)}
    with open(os.path.join(args.output_dir, 'class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print("Starting training...")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_metrics = train_one_epoch(
            model, optimizer, train_loader, device, epoch, print_freq=10
        )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        print("Evaluating on validation set...")
        val_metrics = evaluate(model, val_loader, device)
        
        # Log to tensorboard
        for k, v in train_metrics.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        
        for k, v in val_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch, args,
            filename=f'checkpoint_epoch_{epoch+1}.pth'
        )
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs, args, filename='model_final.pth')
    print("Training complete!")

if __name__ == "__main__":
    main()