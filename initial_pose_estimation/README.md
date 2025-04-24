# Deep Pose Estimation

A robust and accurate approach for estimating 3D object poses using deep learning. This system directly regresses rotation matrices from single RGB images, providing high accuracy and robustness to scale, rotation, and lighting variations.

## Overview

This project implements a direct pose regression approach using deep neural networks. Instead of template matching or keypoint detection, we train a CNN to directly predict the rotation matrix from an input image. This approach has several advantages:

- **Robustness**: Handles scale, rotation, and lighting variations
- **Accuracy**: Learns from the entire dataset to make precise predictions
- **Efficiency**: Fast inference with a single forward pass through the network
- **Generalization**: Can interpolate between training poses for smoother results

## Project Structure

```
.
├── 01_train_regression_model.py    # Training script for pose regression
├── 02_predict_pose.py              # Inference script for pose prediction
├── requirements.txt                # Required Python packages
└── README.md                       # This file
```

## Installation

1. Clone this repository
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Dataset Requirements

The code expects a dataset with the following structure:

```
defined_poses/
├── 000000/                    # Object ID folder
│   ├── rot_000.png            # Pose images
│   ├── rot_001.png
│   └── ...
├── 000001/
│   └── ...
└── rotation_matrices/         # Rotation matrix files
    ├── 000000.txt             # Each line contains a 3x3 matrix (9 values)
    ├── 000001.txt
    └── ...
```

Each object folder should contain images of the object in different orientations, and the rotation_matrices folder should contain corresponding rotation matrices.

## Usage

### 1. Training a Pose Regression Model

To train a model for a specific object:

```bash
python 01_train_regression_model.py --data_dir /path/to/defined_poses --object_id 000000 --output_dir /path/to/models
```

Parameters:
- `--data_dir`: Root directory containing the dataset
- `--object_id`: ID of the object to train on
- `--output_dir`: Directory to save trained models
- `--batch_size`: Batch size for training (default: 32)
- `--num_workers`: Number of data loading workers (default: 4)
- `--learning_rate`: Initial learning rate (default: 0.001)
- `--weight_decay`: Weight decay for regularization (default: 1e-4)
- `--epochs`: Number of training epochs (default: 25)

The script automatically:
- Splits the data into training and validation sets
- Applies data augmentation for better robustness
- Uses a custom loss function that ensures valid rotation matrices
- Saves the best model based on validation loss
- Generates a training loss plot

### 2. Predicting Poses

To predict the pose for test images:

```bash
# Basic usage
python 02_predict_pose.py --model_path /path/to/models/000000_best_model.pth --test_images /path/to/test_image.png

# With visualization and saving results
python 02_predict_pose.py --model_path /path/to/models/000000_best_model.pth --test_images /path/to/test_image1.png /path/to/test_image2.png --display --save_visualization --save_results
```

Parameters:
- `--model_path`: Path to the trained model checkpoint
- `--test_images`: Paths to test images (can specify multiple)
- `--output_dir`: Directory to save results (default: 'results')
- `--display`: Display visualization (flag)
- `--save_visualization`: Save visualizations to file (flag)
- `--save_results`: Save rotation matrices to text files (flag)

## Advanced Tips

### Improving Model Performance

1. **More Training Data**: The model benefits from seeing more poses. If possible, add more training images.

2. **Data Augmentation**: The training script already includes basic augmentation. For more robust models, consider adding:
   - Random noise
   - Background changes
   - More aggressive lighting variations

3. **Fine-tuning**: If you have a small dataset for a specific object, consider first pre-training on all objects, then fine-tuning.

4. **Ensemble Models**: For critical applications, train multiple models and ensemble their predictions.

### Handling Challenging Cases

- **Symmetrical Objects**: Objects with symmetry may have multiple valid rotations. Consider using a more appropriate representation like quaternions.

- **Occlusions**: If objects may be partially occluded, include such examples in training.

- **Similar Background**: If the background in test images might be similar to the object, train with diverse backgrounds.

## Evaluation

To evaluate your model, you can:

1. **Calculate Angular Error**: Compute the angular difference between predicted and ground truth rotations.

2. **Visualize Predictions**: Use the visualization in the prediction script to assess quality visually.

3. **Test on Varied Conditions**: Validate on images with different lighting, scale, and viewpoints.