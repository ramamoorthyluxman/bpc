# Object Pose Estimation Pipeline

A comprehensive computer vision pipeline for 6DOF object pose estimation using RGB-D images. This system combines object detection, point cloud generation, and point cloud registration to accurately estimate the pose of multiple objects in a scene.

## 🚀 Features

- **Multi-object Detection**: Uses MaskRCNN for robust object detection and segmentation
- **Point Cloud Generation**: Converts RGB-D images to 3D point clouds using camera intrinsics
- **ROI Extraction**: Extracts object-specific point cloud regions based on detection masks
- **Parallel Processing**: Registers multiple objects simultaneously for improved performance
- **6DOF Pose Estimation**: Provides full rotation and translation matrices for each detected object
- **Confidence Scoring**: Returns confidence scores for pose estimation quality assessment
- **Comprehensive Logging**: Detailed timing and progress information throughout the pipeline
- **Flexible Configuration**: YAML-based configuration system for easy parameter adjustment

## 📋 Requirements

### Dependencies

- Python 3.7+
- PyTorch
- OpenCV
- PCL (Point Cloud Library)
- NumPy
- PyYAML
- Open3D (for point cloud processing)

### Hardware Requirements

- GPU recommended for MaskRCNN inference
- RGB-D camera (tested with Photoneo sensors)
- Sufficient RAM for point cloud processing (8GB+ recommended)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd pose_estimation_pipeline
```

2. **Install dependencies**
```bash
pip install torch torchvision opencv-python pyyaml numpy open3d
# Additional system dependencies may be required for PCL
```

3. **Set up the project structure**
```
pose_estimation_pipeline/
├── maskRCNN/              # MaskRCNN detection module
├── pcl_builder/           # Point cloud generation
├── roi_extraction/        # ROI extraction utilities
├── pointcloud_registration/ # Point cloud registration
├── config.yaml           # Configuration file
├── pose_estimation_pipeline.py
└── reference_models/      # 3D reference models (.ply files)
```

## ⚙️ Configuration

Create a `config.yaml` file with the following structure:

```yaml
# MaskRCNN Configuration
maskrcnn_model_path: "path/to/maskrcnn/model.pth"
maskrcnn_output_path: "output/maskrcnn/"
maskrcnn_category_txt_path: "path/to/categories.txt"
maskrcnn_confidence_threshold: 0.7
maskrcnn_visualization_and_save: true

# Point Cloud Configuration
save_pcl: true

# ROI Extraction Configuration
roi_extraction_output_path: "output/roi/"
save_rois: true

# Registration Configuration
register_reference_models_path: "reference_models/"
register_output_path: "output/registration/"
registration_visualization_and_save: true
```

## 🎯 Usage

### Basic Usage

```python
from pose_estimation_pipeline import pose_estimation_pipeline

# Initialize the pipeline
pipeline = pose_estimation_pipeline("config.yaml")

# Define input paths
rgb_image = "path/to/rgb_image.png"
depth_image = "path/to/depth_image.png"
camera_params = "path/to/camera_parameters.json"

# Run the complete pipeline
results = pipeline.pose_estimate_pipeline(
    rgb_image=rgb_image,
    depth_image=depth_image, 
    camera_parameters_file_path=camera_params
)

# Process results
for result in results:
    if result['success']:
        print(f"Object: {result['label']}")
        print(f"Transformation: {result['transformation']}")
        print(f"Confidence: {result['confidence_score']}")
```

### Advanced Usage with Custom Parameters

```python
# Initialize with custom config
pipeline = pose_estimation_pipeline("custom_config.yaml")

# Run with specific worker count for parallel processing
results = pipeline.register_all_objects_parallel(
    extracted_clouds_list=extracted_objects,
    scene_pcl=scene_point_cloud,
    max_workers=4  # Limit parallel workers
)
```

## 📊 Pipeline Stages

### 1. Object Detection (MaskRCNN)
- Detects and segments objects in RGB images
- Outputs bounding boxes, masks, and class labels
- Configurable confidence threshold

### 2. Point Cloud Generation
- Converts RGB-D images to 3D point clouds
- Uses camera intrinsic parameters for accurate projection
- Supports various camera formats

### 3. ROI Extraction
- Extracts object-specific point cloud regions
- Uses detection masks to isolate individual objects
- Filters and cleans point cloud data

### 4. Parallel Registration
- Registers extracted objects against reference models
- Uses ICP (Iterative Closest Point) and other registration algorithms
- Parallel processing for multiple objects simultaneously
- Returns 6DOF transformation matrices

## 📈 Output Format

The pipeline returns a list of registration results with the following structure:

```python
{
    'object_index': 0,
    'label': 'object_class_name',
    'success': True,
    'transformation': (rotation_matrix, translation_vector),
    'confidence_score': 0.95,
    'object_info': {...},
    'output_file': 'path/to/registered_object.ply'
}
```

### Saved Outputs

- **Point Clouds**: Individual object point clouds (`.ply` format)
- **Registered Models**: Aligned object models with poses
- **Transformations**: YAML file with all transformation matrices
- **Visualizations**: Optional visualization outputs for debugging

## 🎛️ Camera Parameters

Camera parameters should be provided in JSON format:

```json
{
    "fx": 1234.5,
    "fy": 1234.5,
    "cx": 640.0,
    "cy": 480.0,
    "width": 1280,
    "height": 960,
    "depth_scale": 1000.0
}
```

## 🚀 Performance Optimization

### Parallel Processing
- The pipeline automatically detects optimal worker count
- Manual control available via `max_workers` parameter
- ThreadPoolExecutor used for GPU-memory friendly processing

### Memory Management
- Efficient point cloud handling
- Configurable visualization and saving options
- Automatic cleanup of intermediate results

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in MaskRCNN
   - Limit parallel workers
   - Use CPU-only mode if necessary

2. **Point Cloud Registration Failures**
   - Check reference model quality
   - Verify camera calibration
   - Adjust registration parameters

3. **Detection Issues**
   - Lower confidence threshold
   - Retrain MaskRCNN with domain-specific data
   - Improve lighting conditions

### Debug Mode
Enable detailed logging by setting environment variable:
```bash
export DEBUG_POSE_PIPELINE=1
```

## 📚 API Reference

### Main Classes

#### `pose_estimation_pipeline`
Main pipeline orchestrator class.

**Methods:**
- `__init__(config_path)`: Initialize with configuration
- `pose_estimate_pipeline(rgb_image, depth_image, camera_params)`: Run complete pipeline
- `register_all_objects_parallel(objects, scene_pcl, max_workers)`: Parallel registration
- `print_registration_summary(results)`: Display results summary

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper testing
4. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- MaskRCNN implementation
- Point Cloud Library (PCL)
- Open3D community
- PyTorch team

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review configuration examples

---

**Note**: This pipeline is designed for research and development purposes. For production use, additional optimization and testing may be required.