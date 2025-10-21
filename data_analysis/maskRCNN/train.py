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


class Trainer:
    """Trainer class for Mask R-CNN on mechanical parts dataset with resume functionality"""
    
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

    class MetricLogger:
        """Simple metric logger for training"""
        def __init__(self):
            self.meters = {}
            self.delimiter = "  "
        
        def update(self, **kwargs):
            for k, v in kwargs.items():
                if k not in self.meters:
                    self.meters[k] = Trainer.AverageMeter()
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
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Load datasets
        print(f"Loading dataset from {self.args.data_path}")
        dataset = MechanicalPartsDataset(args.data_path, transform=get_transform(train=True))
        dataset_test = MechanicalPartsDataset(args.data_path, transform=get_transform(train=False))
        
        # Split dataset
        torch.manual_seed(self.args.seed)
        indices = torch.randperm(len(dataset)).tolist()
        train_size = int(len(indices) * 0.8)
        self.dataset_train = torch.utils.data.Subset(dataset, indices[:train_size])
        self.dataset_val = torch.utils.data.Subset(dataset_test, indices[train_size:])
        
        print(f"Dataset loaded: {len(self.dataset_train)} training samples, {len(self.dataset_val)} validation samples")
        print(f"Number of classes: {len(dataset.categories)}")
        print(f"Classes: {dataset.categories}")
        self.num_classes = len(dataset.categories)
        
        # Data loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset_train, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, collate_fn=collate_fn
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.dataset_val, batch_size=1, shuffle=False,
            num_workers=args.workers, collate_fn=collate_fn
        )
        
        # Model
        self.model = get_model_instance_segmentation(self.num_classes)
        self.model.to(self.device)
        
        # Optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
        
        # Scheduler
        self.lr_scheduler = StepLR(self.optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        
        # TensorBoard
        self.writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
        
        # Resume from checkpoint if provided
        self.start_epoch = 0
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print(f"Loading checkpoint from {self.args.resume}")
                checkpoint = torch.load(self.args.resume, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.start_epoch = checkpoint.get('epoch', 0) + 1
                print(f"Resumed from epoch {self.start_epoch}")
            else:
                print(f"Warning: Resume checkpoint {self.args.resume} not found. Starting from scratch.")

    def train_one_epoch(self, epoch):
        self.model.train()
        metric_logger = self.MetricLogger()
        header = f'Epoch: [{epoch}]'
        
        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(self.train_loader) - 1)
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )
        
        for images, targets in metric_logger.log_every(self.train_loader, self.args.print_freq, header):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
            losses_reduced = sum(loss_dict_reduced.values())
            
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
        
        return metric_logger

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for images, targets in self.val_loader:
                try:
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    _ = self.model(images)
                    num_batches += 1
                    print(f"Processed validation batch {num_batches}/{len(self.val_loader)}")
                except Exception as e:
                    print(f"Error during validation batch {num_batches}: {e}")
                    continue
        print(f"Validation completed: processed {num_batches}/{len(self.val_loader)} batches")
        return val_loss
    
    def save_checkpoint(self, epoch):
        save_path = os.path.join(self.args.output_dir, f'model_epoch_{epoch}.pth')
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved model at epoch {epoch} to {save_path}")
        
        checkpoint_path = os.path.join(self.args.output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint at epoch {epoch} to {checkpoint_path}")
    
    def train(self):
        print("Starting training")
        best_val_loss = float('inf')
        for epoch in range(self.start_epoch, self.args.epochs):
            metric_logger = self.train_one_epoch(epoch)
            self.lr_scheduler.step()
            val_loss = self.validate()
            
            self.writer.add_scalar('Loss/train', metric_logger.meters['loss'].avg, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            if epoch % self.args.save_freq == 0 or (epoch + 1) == self.args.epochs:
                self.save_checkpoint(epoch)
        
        self.writer.close()
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
    
    trainer = Trainer(args)
    trainer.train()
