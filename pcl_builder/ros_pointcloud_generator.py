#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from geometry_msgs.msg import Pose
import sensor_msgs.point_cloud2 as pc2
import tf
import os
import struct
import open3d as o3d

class PointCloudGenerator:
    def __init__(self):
        rospy.init_node('pointcloud_generator', anonymous=True)
        
        # Create CV bridge
        self.bridge = CvBridge()
        
        # Parameters
        self.save_dir = rospy.get_param('~save_dir', os.path.expanduser('~/pointclouds'))
        self.frame_id = rospy.get_param('~frame_id', 'camera_link')
        self.downsample_factor = rospy.get_param('~downsample_factor', 1)  # 1 = no downsampling
        
        # Make sure the save directory exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Create subscribers
        self.camera_info_sub = message_filters.Subscriber('/camera/info', CameraInfo)
        self.pose_sub = message_filters.Subscriber('/camera/pose', Pose)
        self.rgb_sub = message_filters.Subscriber('/camera/rgb', Image)
        self.depth_sub = message_filters.Subscriber('/camera/depth', Image)
        
        # Synchronize the messages using ApproximateTimeSynchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.camera_info_sub, self.pose_sub, self.rgb_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.callback)
        
        # Publisher for the point cloud (for visualization)
        self.pc_pub = rospy.Publisher('/camera/point_cloud', PointCloud2, queue_size=1)
        
        rospy.loginfo("Point Cloud Generator Initialized")

    def callback(self, info_msg, pose_msg, rgb_msg, depth_msg):
        try:
            # Convert images from ROS message to OpenCV format
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")  # Assuming depth is in meters
            
            # Handle NaN or infinite values in depth image
            depth_img = np.nan_to_num(depth_img, nan=0, posinf=0, neginf=0)
            
            # Downsample if needed
            if self.downsample_factor > 1:
                rgb_img = rgb_img[::self.downsample_factor, ::self.downsample_factor]
                depth_img = depth_img[::self.downsample_factor, ::self.downsample_factor]
            
            # Get camera intrinsic parameters
            fx = info_msg.K[0]
            fy = info_msg.K[4]
            cx = info_msg.K[2]
            cy = info_msg.K[5]
            
            # If downsampling, adjust the intrinsic parameters
            if self.downsample_factor > 1:
                fx /= self.downsample_factor
                fy /= self.downsample_factor
                cx /= self.downsample_factor
                cy /= self.downsample_factor
            
            # Create point cloud from depth image
            height, width = depth_img.shape
            points = []
            colors = []
            
            for v in range(height):
                for u in range(width):
                    depth = depth_img[v, u]
                    
                    # Skip invalid depth values
                    if depth <= 0.01:
                        continue
                    
                    # Calculate 3D point from depth
                    x = (u - cx) * depth / fx
                    y = (v - cy) * depth / fy
                    z = depth
                    
                    # Get color for the point
                    color = rgb_img[v, u]
                    
                    # Add to lists
                    points.append([x, y, z])
                    colors.append(color)
            
            # Convert to numpy arrays
            points = np.array(points)
            colors = np.array(colors) / 255.0  # Normalize color values to [0, 1]
            
            if len(points) == 0:
                rospy.logwarn("No valid points found in depth image.")
                return
            
            # Apply camera pose transformation
            # Extract position and orientation from pose message
            position = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
            orientation = [pose_msg.orientation.x, pose_msg.orientation.y, 
                           pose_msg.orientation.z, pose_msg.orientation.w]
            
            # Create transformation matrix
            transformation = np.eye(4)
            
            # Convert quaternion to rotation matrix
            rotation_matrix = tf.transformations.quaternion_matrix(orientation)
            transformation[:3, :3] = rotation_matrix[:3, :3]
            
            # Set translation
            transformation[:3, 3] = position
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Transform the point cloud
            pcd.transform(transformation)
            
            # Create filename based on timestamp
            timestamp = rospy.Time.now().to_sec()
            filename = os.path.join(self.save_dir, f"pointcloud_{timestamp:.2f}.pcd")
            
            # Save point cloud
            o3d.io.write_point_cloud(filename, pcd)
            rospy.loginfo(f"Point cloud saved to {filename}")
            
            # Publish point cloud for visualization
            # Convert Open3D point cloud to ROS PointCloud2 message
            header = rospy.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.frame_id
            
            # Create list of points with color
            pc_points = []
            for i in range(len(points)):
                point = pcd.points[i]
                color = pcd.colors[i]
                r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                
                # Pack RGB into a single integer
                rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
                
                pc_points.append([point[0], point[1], point[2], rgb])
            
            # Create and publish PointCloud2 message
            pc_msg = pc2.create_cloud(header, 
                [
                    pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                    pc2.PointField('rgb', 12, pc2.PointField.UINT32, 1),
                ], 
                pc_points)
            
            self.pc_pub.publish(pc_msg)
            
        except Exception as e:
            rospy.logerr(f"Error processing point cloud: {e}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        generator = PointCloudGenerator()
        generator.run()
    except rospy.ROSInterruptException:
        pass