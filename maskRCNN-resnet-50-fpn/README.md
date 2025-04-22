# Instance Segmentation with Custom Dataset Format

This repository contains a complete pipeline for training and running instance segmentation models on a custom dataset format.

## Overview

This implementation provides:
1. Custom dataset handling with polygon-based instance segmentation
2. Training pipeline for Mask R-CNN models
3. Inference and visualization tools
4. Utilities for dataset analysis and management

## Requirements

```
pip install -r requirements.txt
```

## Dataset Format

The dataset follows a custom format with a specific directory structure:
```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── annotations/
│   ├── image1.json
│   ├── image2.json
│   └── ...
└── masks/
    ├── image1_0.png
    ├── image1_1.png
    └── ...
```

Each annotation JSON file contains:
- Image path, dimensions
- A list of masks with their class labels and polygon points
- Paths to mask images

Example annotation:
```json
{
  "image_path": "images/2007_000032.jpg",
  "height": 281,
  "width": 500,
  "masks": [
    {
      "mask_path": "masks/2007_000032_0.png",
      "label": "aeroplane",
      "points": [[105.0, 128.0], [198.0, 135.0], ...]
    },
    ...
  ]
}
```

## Architecture

### Model

The implementation uses Mask R-CNN with a ResNet-50-FPN backbone, a proven architecture for instance segmentation tasks. This model:
- Identifies regions of interest in the image
- Classifies each region
- Generates a pixel-wise mask for each instance

### Training Pipeline

The training process includes:
- Data augmentation (horizontal flips, color jitter)
- Learning rate scheduling
- Checkpoint saving
- TensorBoard monitoring
- Training/validation split

## Scripts

### `pytorch_dataset.py`

A PyTorch Dataset implementation that:
- Loads images and annotations from the custom format
- Converts polygons to masks when necessary
- Applies image transformations
- Outputs tensors ready for model training

### `train_model.py`

Handles the training loop:
- Sets up the Mask R-CNN model architecture
- Implements training and validation cycles
- Manages learning rate scheduling
- Saves model checkpoints and class mappings
- Logs metrics to TensorBoard

### `inference_fixed.py`

Performs inference using a trained model:
- Loads a trained model and class mapping
- Processes single images or directories
- Visualizes results with bounding boxes and masks
- Saves output for further use

### `visualize_dataset.py`

Helps analyze and visualize the dataset:
- Shows original images with polygon annotations
- Displays instance masks
- Provides a tool to examine annotation quality

### `dataset_stats.py`

Generates statistics about the dataset:
- Class distribution
- Image sizes
- Instances per image
- Polygon complexity

### `generate_masks.py`

Utility to generate or regenerate mask images from polygon points.

## Usage

### Setting Up

1. Organize your dataset according to the format above
2. Run `generate_masks.py` to create mask files from polygons

```bash
python3 generate_masks.py --dataset_dir path/to/dataset
```

### Training

```bash
python3 train_model.py --dataset_dir path/to/dataset --output_dir path/to/save/model --batch_size 2 --epochs 10 --pretrained
```

### Inference

```bash
python3 inference.py --model_path path/to/model_final.pth --input path/to/image.jpg --output path/to/results
```

## Approach & Design Considerations

### Polygon vs Raster Masks
The approach stores segmentation information in two complementary formats:
- Polygon points: Compact representation, easy to edit and manipulate
- Raster masks: Necessary for training and visualization

### Design Flexibility
The implementation accommodates different use cases:
- Can work with missing mask files by generating them from polygons
- Supports both training and inference workflows
- Provides visualization tools for quality control

### Performance Optimizations
- Efficient data loading with PyTorch DataLoader
- GPU acceleration for both training and inference
- Batch processing for faster model convergence

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- Pillow
- opencv-python
- tqdm

## Future Improvements

Potential enhancements:
- Support for more model architectures (Detectron2, YOLOv8, etc.)
- More data augmentation options
- Model export to ONNX/TorchScript
- Real-time inference options
- Integration with popular annotation tools

## Tips

If you observe that your Mask R-CNN model is correctly predicting bounding boxes and classes, but the pixel-level segmentation masks are missing in some images. This is a common issue with instance segmentation models and can happen for several reasons:

Confidence threshold for masks: The mask confidence might be below the threshold you're using. While the bounding box detector might be confident about an object's presence, the mask predictor might be less confident.
Training imbalance: The model might have learned to detect objects (bounding boxes) better than segmenting them (masks). This can happen if:

The loss weighting during training favored bounding box detection over mask prediction
The training data had clearer boundaries for boxes than for detailed masks


Complex geometries: Mechanical parts often have intricate shapes with thin structures, holes, or complex edges that are harder to segment accurately compared to simpler objects.
Resolution issues: Mask prediction works better on higher-resolution features. If your objects are small in the images, the mask branch might struggle.
Domain shift: If your test images differ significantly from training images (lighting, perspective, etc.), mask prediction can degrade more quickly than box detection.

To improve mask predictions, you could try:

Lower the mask threshold: In the inference.py script, you might have a threshold for displaying masks - try lowering it.
Train longer with emphasis on mask loss: You can adjust the loss weights to give more importance to the mask branch.
Increase image resolution: Try processing images at a higher resolution during inference.
Data augmentation: If retraining, use more aggressive augmentation focusing on lighting and perspective changes.
Adjust ROI size: In the model architecture, you might increase the size of the ROI features used for mask prediction.

You could also verify this is happening by explicitly printing the mask confidence scores alongside the box confidence scores in your inference script to see if they differ significantly.RetryClaude can make mistakes. Please double-check responses.