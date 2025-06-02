# Point Cloud ROI Extractor

A Python tool for extracting labeled regions of interest (ROIs) from 3D point clouds using 2D polygon annotations. This tool maps 2D polygon coordinates to 3D point cloud data, enabling segmentation and extraction of specific objects or regions.

## Features

- 🎯 **Precise Extraction**: Extract point cloud regions based on 2D polygon annotations
- ⚡ **Optimized Performance**: Vectorized operations for fast processing of large point clouds
- 🎨 **Preserve Attributes**: Maintains colors and normals from original point cloud
- 📁 **Organized Output**: Automatically organizes extracted point clouds by label
- 🔄 **Flexible Usage**: Can save to files or return in-memory objects for further processing
- 🖥️ **Command Line Interface**: Easy-to-use CLI with customizable parameters

## Installation

### Requirements

```bash
pip install numpy open3d matplotlib
```

### Dependencies

- **numpy**: Numerical operations and array handling
- **open3d**: Point cloud I/O and manipulation
- **matplotlib**: 2D polygon containment checking
- **json**: JSON file parsing (built-in)
- **os**: File system operations (built-in)
- **collections**: Data structures (built-in)

## Usage

### Command Line Interface

```bash
python point_cloud_extractor.py --pcd input.pcd --json polygon.json --output extracted_pcls
```

#### Parameters

- `--pcd`: Path to the input point cloud file (.pcd format)
- `--json`: Path to the JSON file containing polygon coordinates and labels
- `--output`: Directory where extracted point clouds will be saved
- `--no-save`: Optional flag to skip saving files and only return data in memory

### Python API

```python
import open3d as o3d
import json
from point_cloud_extractor import extract_labeled_point_clouds

# Load your data
point_cloud = o3d.io.read_point_cloud("input.pcd")
with open("polygon.json", 'r') as f:
    json_data = json.load(f)

# Extract point clouds
extracted_clouds = extract_labeled_point_clouds(
    point_cloud=point_cloud,
    json_data=json_data,
    output_dir="extracted_pcls",
    save_rois=True
)

# Process results
for point_cloud_obj, object_id in extracted_clouds:
    label = object_id['label']
    mask_index = object_id['mask_index']
    point_count = object_id['point_count']
    print(f"Extracted {label} (mask {mask_index}): {point_count} points")
```

## File Formats

### Input Point Cloud (.pcd)

Standard Open3D point cloud format supporting:
- 3D coordinates (required)
- RGB colors (optional)
- Normal vectors (optional)

### Input JSON Format

The JSON file should contain polygon annotations with the following structure:

```json
{
  "width": 1920,
  "height": 1080,
  "masks": [
    {
      "label": "object_name",
      "points": [
        [x1, y1],
        [x2, y2],
        [x3, y3],
        ...
      ]
    },
    {
      "label": "another_object",
      "points": [
        [x1, y1],
        [x2, y2],
        [x3, y3],
        ...
      ]
    }
  ]
}
```

#### JSON Field Descriptions

- **width/height**: Dimensions of the image/projection used for polygon coordinates
- **masks**: Array of polygon objects
- **label**: Name/category of the object (used for organizing output)
- **points**: Array of [x, y] coordinates defining the polygon boundary

### Output Structure

```
extracted_pcls/
├── object_name/
│   ├── object_name_roi_0.pcd
│   ├── object_name_roi_1.pcd
│   └── ...
├── another_object/
│   ├── another_object_roi_0.pcd
│   └── ...
└── ...
```

## Algorithm Overview

1. **Polygon Loading**: Parse JSON file to extract polygon coordinates and labels
2. **Coordinate Mapping**: Map 2D polygon coordinates to 3D point cloud indices
3. **Optimized Masking**: Use vectorized operations with bounding box optimization
4. **Point Filtering**: Remove invalid points (NaN, infinite values)
5. **Cloud Extraction**: Create new point cloud objects with extracted points
6. **Attribute Preservation**: Copy colors and normals if available
7. **Output Organization**: Save or return organized point cloud data

## Performance Optimizations

- **Bounding Box Filtering**: Reduces search space for each polygon
- **Vectorized Operations**: Uses numpy for fast array operations
- **Memory Efficient**: Processes polygons individually to minimize memory usage
- **Invalid Point Filtering**: Automatically removes problematic data points

## Examples

### Basic Extraction

```python
# Simple extraction with file saving
extracted_clouds = extract_labeled_point_clouds(
    point_cloud=my_point_cloud,
    json_data=polygon_data,
    output_dir="results"
)
```

### In-Memory Processing

```python
# Extract without saving files
extracted_clouds = extract_labeled_point_clouds(
    point_cloud=my_point_cloud,
    json_data=polygon_data,
    save_rois=False
)

# Process each extracted region
for cloud, metadata in extracted_clouds:
    # Perform analysis on individual point cloud regions
    volume = cloud.get_oriented_bounding_box().volume()
    print(f"{metadata['label']}: {volume:.2f} cubic units")
```

### Batch Processing

```bash
# Process multiple files
for file in *.pcd; do
    python point_cloud_extractor.py --pcd "$file" --json "${file%.pcd}_polygons.json" --output "extracted_${file%.pcd}"
done
```

## Error Handling

The tool handles various edge cases:

- **Invalid Polygons**: Skips polygons with fewer than 3 points
- **Empty Regions**: Reports when no points are found within a polygon
- **Invalid Points**: Filters out NaN and infinite coordinate values
- **Missing Attributes**: Gracefully handles point clouds without colors/normals
- **File I/O Errors**: Provides clear error messages for file access issues

## Output Information

The tool provides detailed console output including:

- Number of polygons processed
- Points found per polygon
- Valid points after filtering
- File save locations
- Processing statistics

## Troubleshooting

### Common Issues

1. **No points extracted**: Check polygon coordinates match image dimensions
2. **Memory errors**: Process fewer polygons at once or use smaller point clouds
3. **Coordinate mismatch**: Ensure JSON width/height match the projection used
4. **File format errors**: Verify .pcd file is in Open3D-compatible format

### Performance Tips

- Use smaller bounding boxes around regions of interest
- Filter point clouds to remove unnecessary points before processing
- Process labels separately for very large datasets
- Use appropriate coordinate precision in polygon definitions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is open source. Please check the license file for specific terms.

## Contact

For questions, issues, or feature requests, please open an issue in the project repository.