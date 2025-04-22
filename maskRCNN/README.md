# Mask R-CNN for Mechanical Parts Detection and Segmentation

A complete implementation of Mask R-CNN for detecting and segmenting mechanical parts such as brackets, bolts, nuts, and other metal components.

## Features

- Custom dataset loader for mechanical parts with polygon mask annotations
- Automatic category detection from dataset
- Training script with learning rate scheduling and checkpointing
- Evaluation script with COCO metrics
- Inference script for images and videos
- Pretrained model support for transfer learning

## Installation

### Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Format

The dataset should be organized with the following structure:

```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations/
    ├── annotation1.json
    ├── annotation2.json
    └── ...
```

Each annotation JSON file should follow this format:

```json
{
  "image_path": "images/image1.jpg",
  "height": 500,
  "width": 800,
  "masks": [
    {
      "mask_path": "masks/mask1.png",
      "label": "bracket",
      "points": [[x1, y1], [x2, y2], ...]
    },
    ...
  ]
}
```

## Usage

### 1. Prepare your dataset

Organize your images and annotations as described above.

### 2. Test the dataset loader

```bash
python dataset.py /path/to/your/dataset
```

This will:
- Verify your dataset can be loaded correctly
- Generate a `categories.txt` file for inference

### 3. Train the model

```bash
python train.py --data-path /path/to/your/dataset --batch-size 2 --epochs 20
```

Additional training options:
```
--output-dir: Directory to save model checkpoints (default: ./output)
--lr: Learning rate (default: 0.005)
--workers: Number of data loading workers (default: 4)
--resume: Path to checkpoint to resume training from
```

### 4. Evaluate the model

```bash
python evaluation.py --data-path /path/to/your/dataset --model-path ./output/model_final.pth --evaluate --visualize
```

### 5. Run inference on new images

```bash
python inference.py /path/to/image.jpg --model-path ./output/model_final.pth --categories-file categories.txt
```

For batch processing:
```bash
python inference.py /path/to/images/directory --model-path ./output/model_final.pth --categories-file categories.txt
```

For video processing:
```bash
python inference.py /path/to/video.mp4 --model-path ./output/model_final.pth --categories-file categories.txt
```

## Scripts Overview

- **dataset.py**: Custom dataset loader that handles polygon mask annotations
- **model.py**: Model architecture configuration with pretrained backbone options
- **train.py**: Training script with validation and checkpointing
- **evaluate.py**: Evaluation script that calculates precision, recall, and COCO metrics
- **inference.py**: Script for running the model on new images or videos
- **utils.py**: Utilities for dataset analysis and visualization

## Advanced Usage

### Resume Training

To resume training from a checkpoint:

```bash
python train.py --data-path /path/to/your/dataset --resume ./output/checkpoint_epoch_5.pth
```

### Using a Different Backbone

You can modify the model.py file to use a different backbone for potentially better performance. The default is ResNet-50 with FPN.

### Class Imbalance

If your dataset has significant class imbalance, consider:
1. Adjusting class weights in the loss function
2. Balancing your dataset with augmentation

## Acknowledgments

This implementation is based on:
- Mask R-CNN paper: [https://arxiv.org/abs/1703.06870](https://arxiv.org/abs/1703.06870)
- PyTorch's torchvision implementation of Mask R-CNN