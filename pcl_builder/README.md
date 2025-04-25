# Organized Point Cloud Generator

This tool creates organized point clouds from RGB and depth images using camera intrinsic parameters. It maintains a one-to-one mapping between image pixels and 3D points in the resulting point cloud.

## Features

- Creates organized point clouds where each point corresponds to a pixel in the input images
- Preserves RGB color information for each point
- Uses vectorized operations for efficient processing
- Handles invalid depth values properly
- Saves results in standard PLY format
- Includes metadata about organization structure

## Requirements

- Python 3.6+
- NumPy
- OpenCV
- Open3D
- Argparse

For detailed version requirements, see `requirements.txt`.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/organized-point-cloud-generator.git
cd organized-point-cloud-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python create_pointcloud.py --rgb <path_to_rgb_image> --depth <path_to_depth_image> --camera <path_to_camera_json> --output <output_path>
```

### Arguments

- `--rgb`: Path to the RGB image file
- `--depth`: Path to the depth image file
- `--camera`: Path to the camera parameters JSON file
- `--output`: Output path for the point cloud file (.ply format)

### Example

```bash
python create_pointcloud.py --rgb data/rgb.jpg --depth data/depth.png --camera data/camera.json --output output/point_cloud.ply
```

## Camera Parameters Format

The camera parameters should be provided in a JSON file with the following format:

```json
{
  "cx": 1954.1872863769531,
  "cy": 1103.6978149414062,
  "depth_scale": 0.1,
  "fx": 3981.985991142684,
  "fy": 3981.985991142684,
  "height": 2160,
  "width": 3840
}
```

Where:
- `cx`, `cy`: Principal point coordinates (pixels)
- `fx`, `fy`: Focal lengths (pixels)
- `depth_scale`: Scale factor for depth values
- `width`, `height`: Image dimensions (pixels)

## Understanding Organized Point Clouds

An organized point cloud maintains the same structure as the input images:
- Every pixel has a corresponding 3D point (even those with invalid depth)
- Points are ordered in the same way as pixels in the image
- Total number of points equals width × height of the input images
- Invalid depth points (z ≤ 0) have coordinates (0,0,0)

## Output Files

The script generates two files:
1. `<output_path>.ply`: The point cloud file in PLY format
2. `<output_path>_metadata.json`: Metadata file containing organization information

## Technical Details

- Depth values of 0 or negative are considered invalid
- RGB values are normalized to the range [0,1]
- The point cloud is created using vectorized NumPy operations for performance
- Organization metadata is stored separately since Open3D's PointCloud object doesn't support custom attributes

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.