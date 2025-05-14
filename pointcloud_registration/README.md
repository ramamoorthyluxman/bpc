# Point Cloud Registration and Placement Tool

A comprehensive tool for aligning and placing 3D point clouds with visualization, debugging, and size optimization capabilities.

## Overview

This tool provides a complete pipeline for point cloud registration (alignment) and placement:

1. **Registration**: Aligns a source point cloud (Region of Interest) with a reference model
2. **Placement**: Positions the aligned reference model within a larger scene
3. **Size Optimization**: Intelligently reduces output file size while preserving detail

The tool uses a sophisticated multi-stage registration approach:
- Global registration using RANSAC with FPFH features
- Fine-tuning with Iterative Closest Point (ICP)
- Comprehensive visualization and debugging output

## Features

- Adaptive downsampling to handle large point clouds
- Statistical outlier removal for noise handling
- Normal estimation and feature computation for robust matching
- Scale normalization to handle differently-sized models
- Detailed visualization of each registration step
- Automatic size optimization for target file size
- Comprehensive debug information and transformation analysis
- Registration metrics plotting (fitness and RMSE)

## Installation

### Prerequisites

- Python 3.7+
- The dependencies listed in `requirements.txt`

### Setup

```bash
# Clone this repository (if applicable)
git clone https://github.com/yourusername/pointcloud-registration.git
cd pointcloud-registration

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The tool can be used in three different modes:

### 1. Full Pipeline (Registration + Placement)

```bash
python pointcloud_processor.py --source roi.pcd --reference model.ply --scene scene.pcd --output result.pcd
```

### 2. Registration Only

```bash
python pointcloud_processor.py --source roi.pcd --reference model.ply
```

### 3. Placement Only (with pre-computed transformation)

```bash
python pointcloud_processor.py --source roi.pcd --reference model.ply --scene scene.pcd --transform transform.txt --output result.pcd
```

## Command Line Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `--source` | Path to source (ROI) PCD file | Yes |
| `--reference` | Path to reference PLY/PCD file | Yes |
| `--scene` | Path to scene PCD file | No |
| `--transform` | Path to pre-computed transformation matrix | No |
| `--output` | Output file path for combined result (default: combined_result.pcd) | No |
| `--output_dir` | Directory for output files (default: registration_results) | No |
| `--max_size` | Maximum output file size in MB (default: 10) | No |

## Output Files

The tool generates various output files in the specified `--output_dir`:

### Registration Files

- `transformation_matrix.txt`: The computed transformation matrix
- `source_center.txt` & `reference_center.txt`: Center points of source and reference
- `registration_metrics.png`: Plot of registration fitness and RMSE
- `registration_vis_before.pcd` & `registration_vis_after.pcd`: Visualizations before/after registration

### Preprocessing Files (for debugging)

- `preprocess_input_*.pcd`: Input point clouds with coordinate frames
- `preprocess_centered_*.pcd`: Centered point clouds
- `preprocess_scaled_*.pcd`: Normalized and scaled point clouds
- `preprocess_final_*.pcd`: Final preprocessed point clouds
- `preprocess_details_*.txt`: Details of preprocessing steps

### Debug and Visualization Files

- `registration_debug.txt`: Detailed debug information
- `transformation_visualization.txt`: Human-readable transformation details
- `coordinate_frames.pcd`: Visualization of coordinate frames
- `transform_sequence.pcd`: Visualization of transformations
- `transform_sequence_legend.txt`: Legend for transform sequence visualization

### Placement Files (when using scene)

- `placement_vis_before.pcd` & `placement_vis_after.pcd`: Scene placement visualizations

## Example Workflow

1. **Prepare your point clouds**:
   - `source.pcd`: The Region of Interest (ROI) you want to align
   - `reference.ply`: The reference model
   - `scene.pcd`: The larger scene (optional)

2. **Run registration to align source to reference**:
   ```bash
   python pointcloud_processor.py --source source.pcd --reference reference.ply
   ```

3. **Examine the registration results** in the output directory

4. **Run placement to position the reference in the scene**:
   ```bash
   python pointcloud_processor.py --source source.pcd --reference reference.ply --scene scene.pcd --transform registration_results/transformation_matrix.txt --output final_result.pcd
   ```

5. **View the final result** using your preferred point cloud viewer

## Customization

The main processing parameters are defined in the `PointCloudProcessor` class initialization. You can modify these parameters directly in the code to adjust:

- Registration parameters (voxel size, RANSAC parameters, ICP threshold)
- Size reduction parameters (target point counts)
- Visualization settings

## Viewing Results

The output PCD files can be viewed with any point cloud viewer that supports the PCD format, such as:

- [Open3D Visualization](http://www.open3d.org/docs/latest/tutorial/visualization/visualization.html)
- [CloudCompare](https://www.danielgm.net/cc/)
- [MeshLab](https://www.meshlab.net/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.