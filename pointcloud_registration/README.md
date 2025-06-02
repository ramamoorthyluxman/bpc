# Point Cloud Registration and Placement Tool

A comprehensive Python tool for 3D point cloud registration, placement, and visualization with intelligent confidence scoring and size optimization.

## Features

- **Robust Registration**: RANSAC-based global registration followed by iterative ICP refinement
- **Confidence Scoring**: Multi-metric confidence assessment with detailed reporting
- **Flexible Input**: Supports PCD and PLY formats (including mesh sampling)
- **Size Optimization**: Intelligent downsampling to meet file size constraints
- **Visualization**: Automatic 2D comparison visualizations and progress plots
- **Multiple Modes**: Registration-only, placement-only, or full pipeline execution
- **Transformation Formats**: Support for both R,t format and 4x4 transformation matrices

## Installation

### Dependencies

```bash
pip install numpy open3d matplotlib scikit-learn scipy pillow
```

### Required Libraries
- `numpy` - Numerical computations
- `open3d` - Point cloud processing and visualization
- `matplotlib` - Plotting and visualization
- `scikit-learn` - Nearest neighbors for analysis
- `scipy` - Spatial transformations
- `PIL (Pillow)` - Image processing for visualizations

## Usage

### Command Line Interface

#### 1. Registration Only
Compute transformation between source and reference point clouds:

```bash
python point_cloud_processor.py --source roi.pcd --reference model.ply
```

#### 2. Full Pipeline (Registration + Placement)
Register and place reference model in scene:

```bash
python point_cloud_processor.py --source roi.pcd --reference model.ply --scene scene.pcd --output result.ply
```

#### 3. Placement Only (with pre-computed transformation)
Place reference using existing transformation:

```bash
python point_cloud_processor.py --source roi.pcd --reference model.ply --scene scene.pcd --transform transformation_rt.txt --output result.ply
```

### Command Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--source` | str | Yes | Path to source (ROI) point cloud file |
| `--reference` | str | Yes | Path to reference point cloud/mesh file |
| `--scene` | str | No | Path to scene point cloud file |
| `--transform` | str | No | Path to pre-computed transformation file |
| `--output` | str | No | Output file path (default: registration.ply) |
| `--output_dir` | str | No | Output directory (default: registration_results) |
| `--max_size` | float | No | Maximum output file size in MB (default: 10) |
| `--no_save` | flag | No | Skip saving visualizations and intermediate files |

### Programmatic Interface

#### Registration Only

```python
from point_cloud_processor import PointCloudProcessor
import open3d as o3d

# Initialize processor
processor = PointCloudProcessor()

# Load point clouds
source = o3d.io.read_point_cloud("roi.pcd")
reference = o3d.io.read_point_cloud("model.ply")

# Perform registration
R_str, t_str, source_center, reference_center, confidence = processor.run_registration_only(
    source_pcl=source, 
    reference_pcl=reference
)

print(f"Confidence: {confidence:.3f}")
print(f"Rotation: {R_str}")
print(f"Translation: {t_str}")
```

#### Full Pipeline

```python
# Load all point clouds
source = o3d.io.read_point_cloud("roi.pcd")
scene = o3d.io.read_point_cloud("scene.pcd")
reference_path = "model.ply"

# Run full pipeline
R_str, t_str, confidence = processor.run_full_pipeline(
    source_pcl=source,
    reference_file_path=reference_path,
    scene_pcl=scene,
    output_file="result.ply"
)
```

## Configuration Parameters

### Registration Parameters
```python
# Modify these in PointCloudProcessor.__init__()
self.voxel_size = 0.01          # Downsampling voxel size
self.distance_threshold = 0.1    # RANSAC distance threshold
self.ransac_iter = 100000       # RANSAC iterations
self.icp_threshold = 0.1        # ICP convergence threshold
self.icp_max_iter = 100         # Maximum ICP iterations
```

### Size Reduction Parameters
```python
self.reference_points = 5000     # Target points for reference model
self.max_scene_points = 25000    # Maximum points for scene
self.max_file_size_mb = 10       # Maximum output file size
```

## Output Files

The tool generates several output files in the specified output directory:

### Core Results
- `registration.ply` - Combined point cloud with placed reference model
- `transformation_rt.txt` - Transformation in R,t format (rotation matrix + translation vector)
- `transformation_matrix.txt` - Full 4x4 transformation matrix

### Analysis and Reports
- `confidence_report.txt` - Detailed confidence analysis and recommendations
- `registration_comparison_2d.png` - 3-part visualization showing registration results
- `registration_metrics.png` - Registration convergence plots

### Intermediate Data
- `source_center.txt` - Source point cloud center coordinates
- `reference_center.txt` - Reference point cloud center coordinates

## Transformation Formats

### R,t Format
The tool outputs transformations in R,t format where:
- **R**: 3x3 rotation matrix (flattened to 9 space-separated values)
- **t**: 3x1 translation vector in millimeters (3 space-separated values)

Example `transformation_rt.txt`:
```
0.998629 -0.052336 0.000000 0.052336 0.998629 0.000000 0.000000 0.000000 1.000000
45.123456 -12.987654 3.456789
```

### 4x4 Matrix Format
Standard homogeneous transformation matrix saved as space-separated values.

## Confidence Scoring

The tool provides a comprehensive confidence score (0-1) based on:

- **Fitness Score** (30%): Overlap between registered point clouds
- **RMSE Score** (25%): Root mean square error, normalized by point cloud scale
- **Convergence Score** (20%): Quality of ICP convergence
- **Geometry Score** (15%): Geometric consistency of the alignment
- **Transform Score** (10%): Reasonableness of the computed transformation

### Confidence Levels
- **0.8-1.0**: High (Excellent registration)
- **0.6-0.8**: Medium-High (Good registration)
- **0.4-0.6**: Medium (Acceptable registration)
- **0.2-0.4**: Low (Poor registration)
- **0.0-0.2**: Very Low (Failed registration)

## Examples

### Basic Registration
```bash
# Register a scanned object to a reference model
python point_cloud_processor.py \
    --source scanned_part.pcd \
    --reference reference_model.ply
```

### Complete Workflow with Scene Integration
```bash
# Register and place in scene with 5MB size limit
python point_cloud_processor.py \
    --source roi_extraction.pcd \
    --reference cad_model.ply \
    --scene full_scene.pcd \
    --output integrated_scene.ply \
    --max_size 5 \
    --output_dir results/
```

### Batch Processing Example
```python
import os
from point_cloud_processor import PointCloudProcessor

processor = PointCloudProcessor()
processor.max_file_size_mb = 15

# Process multiple files
roi_files = ["roi1.pcd", "roi2.pcd", "roi3.pcd"]
reference = "reference_model.ply"

for roi_file in roi_files:
    source = o3d.io.read_point_cloud(roi_file)
    reference_pcl = o3d.io.read_point_cloud(reference)
    
    R_str, t_str, _, _, confidence = processor.run_registration_only(
        source_pcl=source, 
        reference_pcl=reference_pcl
    )
    
    print(f"{roi_file}: Confidence = {confidence:.3f}")
    if confidence > 0.6:
        print(f"  Good registration: R={R_str[:20]}...")
    else:
        print(f"  Poor registration - review parameters")
```

## Troubleshooting

### Common Issues

1. **Low Confidence Scores**
   - Increase RANSAC iterations: `processor.ransac_iter = 200000`
   - Adjust distance threshold: `processor.distance_threshold = 0.05`
   - Ensure sufficient overlap between point clouds

2. **Large Output Files**
   - Reduce max_scene_points: `processor.max_scene_points = 15000`
   - Lower max_file_size_mb: `processor.max_file_size_mb = 5`

3. **Poor Convergence**
   - Increase ICP iterations: `processor.icp_max_iter = 200`
   - Adjust voxel size: `processor.voxel_size = 0.005`

4. **Memory Issues with Large Point Clouds**
   - Use downsampling before processing
   - Process in chunks for very large scenes

### Performance Optimization

- **For faster processing**: Increase `voxel_size` to 0.02-0.05
- **For higher accuracy**: Decrease `voxel_size` to 0.005-0.001
- **For large datasets**: Use `--no_save` flag to skip visualizations

## Algorithm Overview

1. **Preprocessing**: Centering, scaling, downsampling, outlier removal, normal estimation
2. **Feature Extraction**: FPFH (Fast Point Feature Histograms) computation
3. **Global Registration**: RANSAC-based feature matching
4. **Local Refinement**: Iterative ICP (Iterative Closest Point)
5. **Confidence Assessment**: Multi-metric evaluation
6. **Placement**: Transformation application and scene integration
7. **Size Optimization**: Intelligent downsampling to meet constraints

## License

This tool is provided as-is for research and educational purposes. Please ensure you have appropriate licenses for the Open3D library and other dependencies.

## Contributing

When contributing to this tool, please ensure:
- Code follows existing style conventions
- New features include appropriate documentation
- Test cases are provided for new functionality
- Confidence scoring methodology is preserved