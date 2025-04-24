# Intial Object Pose Estimation - for better registration

This repository contains a robust pipeline for object pose estimation using a hybrid approach combining CNN features and keypoint matching. Given a test image and an object ID, it finds the closest matching pose from a database of known poses and returns the corresponding rotation matrix.

## Overview

The system uses a three-stage approach:
1. **Feature Extraction**: Extract deep features from images using a pre-trained ResNet-50 model
2. **Feature Database**: Build a searchable database of features with dimensionality reduction and efficient nearest neighbor search
3. **Pose Matching**: Match test images to the database using a hybrid approach of CNN features and keypoint matching

## Project Structure

```
.
├── 01_extract_features.py     # Extract CNN features from all database images
├── 02_build_indices.py        # Build search indices and load rotation matrices
├── 03_match_pose.py           # Match test images to database poses
├── requirements.txt           # List of required Python packages
└── README.md                  # This file
```

## Installation

1. Clone this repository
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The code assumes a dataset with the following structure:
```
defined_poses/
├── 000000/                     # Object ID folder
│   ├── rot_000.png            # Pose images
│   ├── rot_001.png
│   └── ...
├── 000001/
│   └── ...
└── rotation_matrices/          # Rotation matrix files
    ├── 000000.txt             # One 3x3 matrix per line
    ├── 000001.txt
    └── ...
```

## Usage

### 1. Extract Features

Extract deep features from all images in the dataset:

```bash
python 01_extract_features.py --data_dir /path/to/defined_poses --output_dir /path/to/features

# Process specific objects
python 01_extract_features.py --data_dir /path/to/defined_poses --output_dir /path/to/features --object_ids 000000 000001
```

Parameters:
- `--data_dir`: Root directory of the dataset
- `--output_dir`: Directory to save extracted features
- `--object_ids`: (Optional) List of specific object IDs to process
- `--batch_size`: Batch size for feature extraction (default: 32)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--force`: Force re-extraction of features even if they already exist

### 2. Build Search Indices

Build search indices and load rotation matrices:

```bash
python 02_build_indices.py --feature_dir /path/to/features --rotation_dir /path/to/defined_poses/rotation_matrices --output_dir /path/to/indices
```

Parameters:
- `--feature_dir`: Directory containing extracted features
- `--rotation_dir`: Directory containing rotation matrices
- `--output_dir`: Directory to save indices
- `--object_ids`: (Optional) List of specific object IDs to process
- `--n_components`: Number of PCA components to use (default: 128)
- `--force`: Force rebuilding of indices even if they already exist

### 3. Match Pose

Match a test image to the closest pose in the database:

```bash
# Basic usage (CNN features only)
python 03_match_pose.py --test_image /path/to/test_image.png --object_id 000000 --index_dir /path/to/indices

# With keypoint verification and visualization
python 03_match_pose.py --test_image /path/to/test_image.png --object_id 000000 --index_dir /path/to/indices --use_keypoints --display --show_keypoints
```

Parameters:
- `--test_image`: Path to the test image
- `--object_id`: Object ID for the test image
- `--index_dir`: Directory containing index files
- `--use_keypoints`: Use keypoint matching for verification
- `--top_k`: Number of top candidates to consider for keypoint verification (default: 5)
- `--display`: Display the matching results
- `--show_keypoints`: Show keypoint matches in the visualization

## Pipeline Details

### CNN Feature Extraction
The system uses ResNet-50 pre-trained on ImageNet to extract 2048-dimensional feature vectors from images. These features capture semantic information about the objects and their poses.

### Dimensionality Reduction
PCA is applied to reduce the dimensionality of feature vectors to 128 dimensions, making similarity search more efficient while preserving most of the variance.

### Nearest Neighbor Search
The system uses scikit-learn's NearestNeighbors to find the closest matching images based on cosine similarity between feature vectors.

### Keypoint Verification (Optional)
For increased robustness, a second stage of verification uses ORB keypoints to match local features between the test image and top candidates from CNN matching.

## Tips and Troubleshooting

- For best results, use the `--use_keypoints` flag when matching poses
- If keypoint matching is slow, decrease the `--top_k` parameter
- The feature extraction step is computationally intensive but only needs to be run once per dataset
- You can process a subset of objects by using the `--object_ids` parameter