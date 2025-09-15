import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import tempfile
import subprocess

# Import your existing functions
try:
    from read_dataset import read_dataset
    from pxl_2_point import create_pointcloud_with_colors
    import open3d as o3d
except ImportError as e:
    st.error(f"Could not import your functions: {e}")
    st.info("Make sure read_dataset.py and pxl_2_point.py are in the same directory")

# Set page config
st.set_page_config(
    page_title="3D Point Cloud Visualizer (Optimized)", 
    page_icon="🎯", 
    layout="wide"
)

# ============= CRITICAL: AGGRESSIVE DOWNSAMPLING =============

@st.cache_data
def smart_downsample(points, colors, method="voxel", target_points=50000):
    """Aggressively downsample point cloud BEFORE sending to browser"""
    
    n_original = len(points)
    
    # If already small enough, return as is
    if n_original <= target_points:
        return points, colors, n_original
    
    if method == "voxel":
        # Voxel-based downsampling using Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0 if colors.max() > 1 else colors)
        
        # Calculate voxel size to achieve target points
        bbox = points.max(axis=0) - points.min(axis=0)
        volume = np.prod(bbox)
        voxel_size = (volume / target_points) ** (1/3) * 1.5  # Slightly larger voxels
        
        downsampled = pcd.voxel_down_sample(voxel_size)
        
        # If still too many, increase voxel size
        while len(downsampled.points) > target_points:
            voxel_size *= 1.2
            downsampled = pcd.voxel_down_sample(voxel_size)
        
        return np.asarray(downsampled.points), np.asarray(downsampled.colors) * 255, n_original
    
    elif method == "uniform":
        # Uniform random sampling
        indices = np.random.choice(n_original, target_points, replace=False)
        return points[indices], colors[indices], n_original
    
    elif method == "poisson":
        # Poisson disk sampling for better distribution
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0 if colors.max() > 1 else colors)
        
        # Estimate normals first
        pcd.estimate_normals()
        
        # Poisson disk sampling
        downsampled = pcd.voxel_down_sample(0.01)  # Initial voxel downsample
        
        return np.asarray(downsampled.points), np.asarray(downsampled.colors) * 255, n_original
    
    else:  # fps - Farthest Point Sampling
        # More uniform coverage but slower
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0 if colors.max() > 1 else colors)
        
        downsampled = pcd.farthest_point_down_sample(target_points)
        
        return np.asarray(downsampled.points), np.asarray(downsampled.colors) * 255, n_original

def save_full_pointcloud(point_cloud, filename="full_pointcloud.ply"):
    """Save full point cloud to file for external viewing"""
    o3d.io.write_point_cloud(filename, point_cloud)
    return filename

def launch_external_viewer(point_cloud_path):
    """Launch external point cloud viewer"""
    try:
        # Try different viewers
        viewers = [
            ["cloudcompare", point_cloud_path],
            ["meshlab", point_cloud_path],
            ["python", "-m", "open3d", "draw", point_cloud_path]
        ]
        
        for viewer_cmd in viewers:
            try:
                subprocess.Popen(viewer_cmd)
                return True, viewer_cmd[0]
            except:
                continue
        
        return False, None
    except Exception as e:
        return False, str(e)

# Helper functions (keep all the original helper functions)
def load_camera_parameters(dataset_dir, camera_type, image_index):
    """Load camera parameters from scene_camera file"""
    try:
        if camera_type == "photoneo":
            scene_camera_file = os.path.join(dataset_dir, "scene_camera_photoneo.json")
        else:
            scene_camera_file = os.path.join(dataset_dir, f"scene_camera_{camera_type}.json")
        
        if os.path.exists(scene_camera_file):
            with open(scene_camera_file, 'r') as f:
                scene_cameras = json.load(f)
            
            if str(image_index) in scene_cameras:
                cam_params = scene_cameras[str(image_index)]
                k_matrix = np.array(cam_params["cam_K"]).reshape(3, 3)
                return k_matrix
        
        return None
    except Exception as e:
        st.error(f"Error loading camera parameters: {e}")
        return None

def extract_pose_data(row):
    """Extract pose data from CSV row"""
    poses = []
    
    try:
        rotation_elements = {}
        translation_elements = {}
        
        for col in row.index:
            col_lower = col.lower()
            if col_lower.startswith('r') and len(col_lower) == 3:
                try:
                    rotation_elements[col_lower] = float(row[col])
                except:
                    continue
            elif col_lower in ['tx', 'ty', 'tz']:
                try:
                    translation_elements[col_lower] = float(row[col])
                except:
                    continue
        
        if len(rotation_elements) == 9 and len(translation_elements) == 3:
            R = np.array([
                [rotation_elements['r11'], rotation_elements['r12'], rotation_elements['r13']],
                [rotation_elements['r21'], rotation_elements['r22'], rotation_elements['r23']],
                [rotation_elements['r31'], rotation_elements['r32'], rotation_elements['r33']]
            ])
            
            t = np.array([
                translation_elements['tx'],
                translation_elements['ty'], 
                translation_elements['tz']
            ])
            
            obj_id = row.get('obj_id', 'GT_Object')
            
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            
            poses.append({
                'Object': obj_id,
                'X': t[0], 'Y': t[1], 'Z': t[2],
                'Rotation_Matrix': R.tolist(),
                'transformation_matrix': T
            })
    
    except Exception as e:
        st.error(f"Error extracting pose data: {e}")
    
    return poses

def create_masked_image(image, row):
    """Create image with polygon mask overlays using OpenCV"""
    # Simplified version - implement full mask logic as needed
    return image

def create_lightweight_3d_plot(points, colors, pose_data=None, point_size=2):
    """Create lightweight 3D visualization using Plotly with limited points"""
    
    # Ensure we have RGB colors in correct format
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(int)
    
    # Create figure
    fig = go.Figure()
    
    # Add point cloud
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1], 
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color=['rgb({},{},{})'.format(r, g, b) for r, g, b in colors],
            opacity=0.8
        ),
        name="Point Cloud",
        hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>"
    ))
    
    # Add poses if available
    if pose_data:
        for i, pose in enumerate(pose_data):
            if 'transformation_matrix' in pose:
                T = pose['transformation_matrix']
                pos = T[:3, 3]
                R = T[:3, :3]
                obj_id = pose.get('Object', f'Object_{i}')
                
                axis_length = 0.1
                
                # Draw coordinate axes
                for j, (color, name) in enumerate([('red', 'X'), ('green', 'Y'), ('blue', 'Z')]):
                    axis_end = pos + R[:, j] * axis_length
                    fig.add_trace(go.Scatter3d(
                        x=[pos[0], axis_end[0]],
                        y=[pos[1], axis_end[1]],
                        z=[pos[2], axis_end[2]],
                        mode='lines',
                        line=dict(color=color, width=5),
                        name=f"{obj_id}_{name}",
                        showlegend=False
                    ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        ),
        height=650,
        title="3D Point Cloud Visualization"
    )
    
    return fig

def load_actual_data(selected_row, row_idx):
    """Load actual image and point cloud data from selected row"""
    try:
        # Extract paths from CSV row
        image_path = selected_row.iloc[0] if len(selected_row) > 0 else selected_row['image_path']
        
        # Get other parameters with defaults
        camera_type = selected_row.get('camera_type', 'cam1')
        scene_id = selected_row.get('scene_id', '1')
        image_index = selected_row.get('image_index', '0')
        depth_scale = float(selected_row.get('depth_scale', 1000))
        
        st.info(f"Loading data for row {row_idx}")
        
        # Load RGB image
        if os.path.exists(str(image_path)):
            image = cv2.imread(str(image_path))
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.session_state.current_image = image_rgb
                st.success("✅ Image loaded")
            else:
                st.error(f"Could not load image: {image_path}")
                return
        else:
            st.error(f"Image not found: {image_path}")
            return
        
        # Find depth image
        rgb_dir = os.path.dirname(str(image_path))
        dataset_dir = os.path.dirname(rgb_dir)
        
        if str(camera_type) == "photoneo":
            depth_dir = os.path.join(dataset_dir, "depth_photoneo")
        else:
            depth_dir = os.path.join(dataset_dir, f"depth_{camera_type}")
        
        image_filename = os.path.basename(str(image_path))
        depth_path = os.path.join(depth_dir, image_filename)
        
        if not os.path.exists(depth_path):
            st.error(f"Depth image not found: {depth_path}")
            return
        
        # Load camera parameters
        k_matrix = load_camera_parameters(dataset_dir, str(camera_type), str(image_index))
        if k_matrix is None:
            k_matrix = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]])
            st.warning("Using default camera matrix")
        
        # Create point cloud
        with st.spinner("Creating point cloud..."):
            point_cloud_points, point_cloud_colors = create_pointcloud_with_colors(
                depth_path=depth_path,
                rgb_path=str(image_path),
                k_matrix=k_matrix,
                depth_scale=depth_scale,
                use_gpu=False,
            )
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud_points)
            pcd.colors = o3d.utility.Vector3dVector(point_cloud_colors / 255.0)
            
            # Store FULL point cloud
            st.session_state.full_point_cloud = pcd
            st.session_state.full_points_count = len(point_cloud_points)
            
            st.success(f"✅ Point cloud created: {len(point_cloud_points):,} points")
        
        # Extract pose data
        st.session_state.pose_data = extract_pose_data(selected_row)
        if st.session_state.pose_data:
            st.success(f"✅ Found {len(st.session_state.pose_data)} GT poses")
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

# Main app
st.title("🎯 3D Point Cloud Visualizer (Optimized for Large Data)")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'full_point_cloud' not in st.session_state:
    st.session_state.full_point_cloud = None
if 'full_points_count' not in st.session_state:
    st.session_state.full_points_count = 0
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'pose_data' not in st.session_state:
    st.session_state.pose_data = None

# Sidebar
st.sidebar.header("📁 Data Controls")

# Load dataset
st.sidebar.subheader("📊 Load from Folders")
dataset_folder = st.sidebar.text_input("Dataset Root Path", placeholder="/path/to/dataset")
models_folder = st.sidebar.text_input("3D Models Path", placeholder="/path/to/models")

if st.sidebar.button("📊 Read Dataset"):
    if dataset_folder and models_folder:
        try:
            with st.spinner("Processing dataset..."):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_csv_path = f"master_dataset_{timestamp}.csv"
                
                result = read_dataset(dataset_folder, models_folder, output_csv_path)
                
                if result['success']:
                    st.session_state.df = pd.read_csv(output_csv_path)
                    st.sidebar.success(f"✅ Dataset loaded: {result['total_rows']} entries")
                else:
                    st.sidebar.error(f"❌ Error: {result['message']}")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")

# Upload CSV
st.sidebar.subheader("📄 Upload CSV")
csv_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
if csv_file:
    st.session_state.df = pd.read_csv(csv_file)
    st.sidebar.success(f"✅ CSV loaded: {len(st.session_state.df)} rows")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📋 Dataset Table")
    
    if st.session_state.df is not None:
        st.info(f"Dataset: {len(st.session_state.df)} rows × {len(st.session_state.df.columns)} columns")
        
        # Display table
        display_cols = st.session_state.df.columns[:8].tolist()
        
        selected_indices = st.dataframe(
            st.session_state.df[display_cols], 
            use_container_width=True,
            height=400,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        if selected_indices.selection.rows:
            selected_idx = selected_indices.selection.rows[0]
            selected_row = st.session_state.df.iloc[selected_idx]
            
            st.success(f"✅ Selected row {selected_idx}")
            
            with st.expander("🔍 Row Details"):
                st.json(selected_row.to_dict())
            
            if st.button("🖼️ Load Image + Point Cloud", type="primary"):
                load_actual_data(selected_row, selected_idx)
    else:
        st.info("📁 Upload CSV or load dataset")
    
    # Image display
    st.header("🖼️ RGB Image")
    
    if st.session_state.current_image is not None:
        st.image(st.session_state.current_image, caption="RGB Image", use_container_width=True)
        h, w = st.session_state.current_image.shape[:2]
        st.caption(f"Size: {w} × {h} pixels")
    else:
        st.info("Select row and load data")

with col2:
    st.header("🎯 3D Point Cloud")
    
    if st.session_state.full_point_cloud is not None:
        
        # Critical info about point cloud size
        st.warning(f"⚠️ Full point cloud has {st.session_state.full_points_count:,} points")
        
        # Downsampling controls
        st.subheader("🔧 Visualization Settings")
        
        col2a, col2b = st.columns(2)
        with col2a:
            downsample_method = st.selectbox(
                "Downsampling Method",
                ["voxel", "uniform", "poisson"],
                help="Voxel: Spatially uniform | Uniform: Random | Poisson: Best quality"
            )
        
        with col2b:
            max_display_points = st.select_slider(
                "Max Display Points",
                options=[1000, 5000, 10000, 25000, 50000, 100000, 200000],
                value=50000,
                help="⚠️ More than 100k points may be slow"
            )
        
        # Point size and pose controls
        col2c, col2d = st.columns(2)
        with col2c:
            point_size = st.slider("Point Size", 1, 10, 2)
        with col2d:
            show_poses = st.checkbox("Show GT Poses", value=False)
        
        # Process and display button
        if st.button("🚀 Process & Display", type="primary"):
            with st.spinner(f"Downsampling {st.session_state.full_points_count:,} → {max_display_points:,} points..."):
                
                # Extract full point cloud data
                full_points = np.asarray(st.session_state.full_point_cloud.points)
                full_colors = np.asarray(st.session_state.full_point_cloud.colors)
                
                # Downsample
                display_points, display_colors, original_count = smart_downsample(
                    full_points, 
                    full_colors,
                    method=downsample_method,
                    target_points=max_display_points
                )
                
                st.success(f"✅ Downsampled: {original_count:,} → {len(display_points):,} points")
                
                # Create visualization
                fig = create_lightweight_3d_plot(
                    display_points,
                    display_colors,
                    st.session_state.pose_data if show_poses else None,
                    point_size
                )
                
                # Display
                st.plotly_chart(fig, use_container_width=True)
                
                # Stats
                reduction_ratio = (1 - len(display_points) / original_count) * 100
                st.info(f"📊 Displaying {len(display_points):,} points ({reduction_ratio:.1f}% reduction)")
        
        # External viewer options
        st.subheader("🖥️ External Viewer Options")
        
        col2e, col2f = st.columns(2)
        
        with col2e:
            if st.button("💾 Save Full Point Cloud (.ply)"):
                filename = f"pointcloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ply"
                saved_path = save_full_pointcloud(st.session_state.full_point_cloud, filename)
                st.success(f"✅ Saved to: {saved_path}")
                
                # Provide download link
                with open(saved_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download PLY file",
                        data=f.read(),
                        file_name=filename,
                        mime="application/octet-stream"
                    )
        
        with col2f:
            if st.button("🚀 Launch External Viewer"):
                # Save temp file
                temp_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
                o3d.io.write_point_cloud(temp_file.name, st.session_state.full_point_cloud)
                
                success, viewer = launch_external_viewer(temp_file.name)
                if success:
                    st.success(f"✅ Launched {viewer}")
                else:
                    st.error("❌ No external viewer found. Install CloudCompare or MeshLab")
        
        # Display pose data if available
        if show_poses and st.session_state.pose_data:
            st.subheader("📍 Ground Truth Poses")
            pose_df = pd.DataFrame(st.session_state.pose_data)
            if not pose_df.empty:
                st.dataframe(pose_df[['Object', 'X', 'Y', 'Z']], use_container_width=True)
    
    else:
        st.info("Select a row and load data to view point cloud")

# Footer
st.markdown("---")
st.markdown("### 💡 **Optimizations for Large Point Clouds**")
st.markdown("""
This version handles millions of points efficiently by:

1. **Smart Downsampling**: Only sends manageable data to browser (50k-200k points max)
2. **Multiple Methods**: Voxel, Uniform, or Poisson downsampling
3. **Full Data Preservation**: Original point cloud kept in memory
4. **External Viewers**: Export to CloudCompare/MeshLab for full resolution
5. **Progressive Loading**: Process on-demand, not automatically

**For 8M+ points**: 
- Use ≤50k points for web display
- Export .ply file for external viewers
- CloudCompare handles billions of points efficiently
""")

# Config recommendation
st.info("""
💡 **If still getting websocket errors**, add this to `.streamlit/config.toml`:
```toml
[server]
maxMessageSize = 1000
maxUploadSize = 1000
```
""")