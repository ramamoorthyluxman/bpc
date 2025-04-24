# ROS PointCloud Generator

This package provides a ROS node for generating and saving colored point clouds from RGB and depth camera data. The node synchronizes camera information, pose data, RGB images, and depth images to create accurate 3D point clouds with color information.

## Features

- Generates colored 3D point clouds from RGB and depth images
- Applies camera pose transformations to place points in the correct world coordinates
- Supports downsampling for more efficient processing
- Saves point clouds in PCD format with timestamp-based filenames
- Publishes point clouds as ROS messages for real-time visualization
- Handles invalid depth values and other edge cases

## Prerequisites

- ROS (tested with ROS Noetic, but should work with Melodic or Kinetic)
- Python 3 (or Python 2.7 for older ROS distributions)
- NumPy
- OpenCV
- Open3D
- cv_bridge
- tf

## Installation

1. Clone this repository into your catkin workspace:
   ```bash
   cd ~/catkin_ws/src
   git clone https://github.com/your_username/ros_pointcloud_generator.git
   ```

2. Install Python dependencies:
   ```bash
   pip install numpy opencv-python open3d
   ```

3. Build your workspace:
   ```bash
   cd ~/catkin_ws
   catkin_make
   ```

4. Source your workspace:
   ```bash
   source ~/catkin_ws/devel/setup.bash
   ```

## Usage

### Basic Usage

Run the point cloud generator node:

```bash
rosrun ros_pointcloud_generator pointcloud_generator.py
```

### Topics

The node subscribes to the following topics:
- `/camera/info` (sensor_msgs/CameraInfo) - Camera calibration information
- `/camera/pose` (geometry_msgs/Pose) - Camera pose in world coordinates
- `/camera/rgb` (sensor_msgs/Image) - RGB image from the camera
- `/camera/depth` (sensor_msgs/Image) - Depth image from the camera

The node publishes:
- `/camera/point_cloud` (sensor_msgs/PointCloud2) - Generated point cloud for visualization

### Parameters

- `~save_dir` (string, default: "~/pointclouds") - Directory to save point cloud files
- `~frame_id` (string, default: "camera_link") - Frame ID for the published point cloud
- `~downsample_factor` (int, default: 1) - Factor to downsample images (1 = no downsampling)

Example with custom parameters:
```bash
rosrun ros_pointcloud_generator pointcloud_generator.py _save_dir:=/data/pointclouds _downsample_factor:=2 _frame_id:=my_camera
```

### Launch File Example

You can also use a launch file for easier configuration:

```xml
<launch>
  <node name="pointcloud_generator" pkg="ros_pointcloud_generator" type="pointcloud_generator.py" output="screen">
    <param name="save_dir" value="/data/pointclouds"/>
    <param name="downsample_factor" value="2"/>
    <param name="frame_id" value="camera_optical_frame"/>
  </node>
</launch>
```

## Visualization

You can visualize the published point clouds in real-time using RViz:

1. Open RViz:
   ```bash
   rosrun rviz rviz
   ```

2. Add a PointCloud2 display and set its topic to `/camera/point_cloud`

3. Set the Fixed Frame to match your `frame_id` parameter (default: "camera_link")

## Data Format

The saved point clouds are in PCD (Point Cloud Data) format, which can be opened with various point cloud processing tools like:

- Open3D
- PCL (Point Cloud Library)
- CloudCompare
- MeshLab

## Additional Features

### AOLP and DOLP Support

The code can be extended to incorporate Angle of Linear Polarization (AOLP) and Degree of Linear Polarization (DOLP) information if provided. This would require small modifications to the callback function.

### Custom Point Cloud Processing

You can easily extend the script to perform custom processing on the point cloud before saving, such as:
- Filtering outliers
- Voxel-based downsampling
- Normal estimation
- Surface reconstruction

## Troubleshooting

- If depth and RGB images are not perfectly aligned, you may need to apply additional registration or use a properly calibrated RGB-D camera.
- For large point clouds, consider increasing the downsampling factor.
- If the node crashes due to memory limitations, try processing the images in chunks.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses Open3D for point cloud processing
- Thanks to the ROS community for the excellent sensor_msgs and tf packages