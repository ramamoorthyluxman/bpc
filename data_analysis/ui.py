import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import csv
import cv2
import numpy as np
from datetime import datetime
from read_dataset import read_dataset
from pxl_2_point import create_pointcloud_with_colors
import open3d as o3d
import json

class TabsUI:
    def __init__(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Test Results Visualizer")
        self.root.geometry("800x600")
        
        # Create notebook (tab container)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create first tab - Test Results
        self.test_results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.test_results_frame, text="Visualize Test Results")
        
        # Create second tab - Ground Truths
        self.ground_truths_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ground_truths_frame, text="Visualize Ground Truths")
        
        # Setup Ground Truths tab content
        self.setup_ground_truths_tab()
        
    def setup_ground_truths_tab(self):
        # Frame to hold buttons side by side
        buttons_frame = ttk.Frame(self.ground_truths_frame)
        buttons_frame.pack(anchor='nw', padx=10, pady=10)
        
        # Browse button for dataset root folder
        browse_button = ttk.Button(
            buttons_frame, 
            text="Browse Dataset Root Folder", 
            command=self.browse_folder
        )
        browse_button.pack(side='left', padx=(0, 10))
        
        # Browse button for 3d models path
        models_browse_button = ttk.Button(
            buttons_frame, 
            text="Browse 3D Models Path", 
            command=self.browse_models_folder
        )
        models_browse_button.pack(side='left', padx=(0, 10))
        
        # Read Dataset button
        self.read_button = ttk.Button(
            buttons_frame, 
            text="Read Dataset", 
            command=self.read_dataset_clicked
        )
        self.read_button.pack(side='left')
        
        # Frame to hold path labels side by side
        paths_frame = ttk.Frame(self.ground_truths_frame)
        paths_frame.pack(anchor='nw', padx=10, pady=5)
        
        # Label to display selected dataset path
        self.path_label = ttk.Label(
            paths_frame, 
            text="No folder selected", 
            wraplength=350,
            foreground="gray"
        )
        self.path_label.pack(side='left', padx=(0, 10))
        
        # Label to display selected 3d models path
        self.models_path_label = ttk.Label(
            paths_frame, 
            text="No folder selected", 
            wraplength=350,
            foreground="gray"
        )
        self.models_path_label.pack(side='left')
        
        # Status label for showing progress/results
        self.status_label = ttk.Label(
            self.ground_truths_frame,
            text="Ready to process dataset",
            foreground="blue"
        )
        self.status_label.pack(anchor='nw', padx=10, pady=10)
        
        # CSV Display Area
        self.setup_csv_display()
        
        # Initialize paths
        self.dataset_path = ""
        self.models_path = ""
        self.current_csv_path = ""
    
    def setup_csv_display(self):
        """Setup the CSV display table with scrollbars"""
        # Label for CSV display
        csv_label = ttk.Label(self.ground_truths_frame, text="Dataset CSV Content:")
        csv_label.pack(anchor='nw', padx=10, pady=(20, 5))
        
        # Frame for the table and scrollbars - fixed height, full width
        table_frame = ttk.Frame(self.ground_truths_frame)
        table_frame.pack(fill='x', padx=10, pady=5)
        
        # Create Treeview for displaying CSV data
        self.csv_tree = ttk.Treeview(table_frame, height=10)
        
        # Vertical scrollbar
        v_scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.csv_tree.yview)
        self.csv_tree.configure(yscrollcommand=v_scrollbar.set)
        
        # Horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(table_frame, orient='horizontal', command=self.csv_tree.xview)
        self.csv_tree.configure(xscrollcommand=h_scrollbar.set)
        
        # Grid layout for table and scrollbars
        self.csv_tree.grid(row=0, column=0, sticky='ew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        # Configure grid weights for resizing
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Load image + PCL button
        self.load_button = ttk.Button(
            self.ground_truths_frame,
            text="Load Image + PCL",
            command=self.load_image_pcl_clicked
        )
        self.load_button.pack(anchor='nw', padx=10, pady=10)
        
        # Setup image and 3D display area
        self.setup_display_area()
        
    def browse_folder(self):
        folder_path = filedialog.askdirectory(title="Select Dataset Root Folder")
        if folder_path:
            self.dataset_path = folder_path
            self.path_label.config(text=f"Selected Path: {folder_path}", foreground="black")
        else:
            self.path_label.config(text="No folder selected", foreground="gray")
            
    def browse_models_folder(self):
        folder_path = filedialog.askdirectory(title="Select 3D Models Path")
        if folder_path:
            self.models_path = folder_path
            self.models_path_label.config(text=f"Selected Path: {folder_path}", foreground="black")
        else:
            self.models_path_label.config(text="No folder selected", foreground="gray")
    
    def read_dataset_clicked(self):
        """Handle Read Dataset button click"""
        # Validate paths are selected
        if not self.dataset_path:
            messagebox.showerror("Error", "Please select Dataset Root Folder first")
            return
        
        if not self.models_path:
            messagebox.showerror("Error", "Please select 3D Models Path first")
            return
        
        # Create meta_data folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        meta_data_dir = os.path.join(script_dir, "meta_data")
        os.makedirs(meta_data_dir, exist_ok=True)
        
        # Check if CSV already exists
        existing_csvs = [f for f in os.listdir(meta_data_dir) if f.endswith('.csv')]
        
        if existing_csvs:
            # Show dialog asking user what to do
            choice = self.ask_existing_csv_choice(existing_csvs)
            
            if choice == "use_existing":
                # Use existing CSV
                latest_csv = max(existing_csvs)  # Get the most recent one alphabetically
                csv_path = os.path.join(meta_data_dir, latest_csv)
                self.current_csv_path = csv_path
                self.status_label.config(
                    text=f"✓ Using existing CSV: {latest_csv}", 
                    foreground="green"
                )
                # Load and display the CSV
                self.load_and_display_csv(csv_path)
                messagebox.showinfo(
                    "Using Existing CSV", 
                    f"Using existing dataset CSV:\n{csv_path}"
                )
                return
            elif choice == "create_new":
                # Continue with creating new CSV
                pass
            else:
                # User cancelled
                return
        
        # Generate output CSV path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_path = os.path.join(meta_data_dir, f"master_dataset_{timestamp}.csv")
        
        # Disable button and show processing status
        self.read_button.config(state='disabled')
        self.status_label.config(text="Processing dataset... Please wait", foreground="orange")
        
        # Start processing in separate thread to avoid blocking UI
        thread = threading.Thread(
            target=self.process_dataset_thread,
            args=(self.dataset_path, self.models_path, output_csv_path)
        )
        thread.daemon = True
        thread.start()
    
    def ask_existing_csv_choice(self, existing_csvs):
        """Show dialog asking user what to do with existing CSV"""
        # Create custom dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Dataset CSV Exists")
        dialog.geometry("400x150")
        dialog.resizable(False, False)
        
        # Center the dialog
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Result variable
        result = None
        
        def on_use_existing():
            nonlocal result
            result = "use_existing"
            dialog.destroy()
        
        def on_create_new():
            nonlocal result
            result = "create_new"
            dialog.destroy()
        
        def on_close():
            nonlocal result
            result = "cancel"
            dialog.destroy()
        
        dialog.protocol("WM_DELETE_WINDOW", on_close)
        
        # Message
        message = f"Dataset CSV already exists in meta_data folder.\nFound {len(existing_csvs)} CSV file(s).\n\nWhat would you like to do?"
        msg_label = ttk.Label(dialog, text=message, justify='center')
        msg_label.pack(pady=20)
        
        # Buttons frame
        buttons_frame = ttk.Frame(dialog)
        buttons_frame.pack(pady=10)
        
        use_btn = ttk.Button(buttons_frame, text="Use Existing", command=on_use_existing)
        use_btn.pack(side='left', padx=10)
        
        create_btn = ttk.Button(buttons_frame, text="Create New", command=on_create_new)
        create_btn.pack(side='left', padx=10)
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result
    
    def load_and_display_csv(self, csv_path):
        """Load CSV file and display in the treeview"""
        try:
            # Clear existing data
            for item in self.csv_tree.get_children():
                self.csv_tree.delete(item)
            
            # Read CSV file
            with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                
                # Read header
                headers = next(csv_reader)
                
                # Configure treeview columns
                self.csv_tree['columns'] = headers
                self.csv_tree['show'] = 'headings'
                
                # Configure column headings and widths
                for header in headers:
                    self.csv_tree.heading(header, text=header)
                    # Set reasonable column width
                    self.csv_tree.column(header, width=120, minwidth=80)
                
                # Read and insert data rows
                row_count = 0
                for row in csv_reader:
                    # Ensure row has same number of columns as headers
                    while len(row) < len(headers):
                        row.append('')
                    
                    self.csv_tree.insert('', 'end', values=row)
                    row_count += 1
                
                # Update status with row count
                current_status = self.status_label.cget('text')
                if "✓" in current_status:
                    self.status_label.config(
                        text=f"{current_status} | Displaying {row_count} rows",
                        foreground="green"
                    )
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file:\n{str(e)}")
            self.status_label.config(text="✗ Error loading CSV", foreground="red")
    
    def setup_display_area(self):
        """Setup the image and 3D point cloud display area"""
        # Frame for image and 3D viewer side by side
        display_frame = ttk.Frame(self.ground_truths_frame)
        display_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left side - Image display with reasonable size for large images
        image_frame = ttk.LabelFrame(display_frame, text="RGB Image")
        image_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Small zoom buttons frame positioned above the image canvas
        zoom_buttons_frame = ttk.Frame(image_frame)
        zoom_buttons_frame.pack(anchor='ne', padx=5, pady=2)
        
        # Small zoom buttons
        self.zoom_in_btn = tk.Button(
            zoom_buttons_frame, 
            text="+", 
            font=("Arial", 10, "bold"), 
            command=self.zoom_in, 
            width=2, 
            height=1
        )
        self.zoom_in_btn.pack(side='left', padx=(0, 2))
        
        self.zoom_out_btn = tk.Button(
            zoom_buttons_frame, 
            text="-", 
            font=("Arial", 10, "bold"), 
            command=self.zoom_out, 
            width=2, 
            height=1
        )
        self.zoom_out_btn.pack(side='left')
        
        # Container for image canvas and zoom buttons
        image_container = ttk.Frame(image_frame)
        image_container.pack(fill='both', expand=True)
        
        # Canvas for image with scrollbars
        self.image_canvas = tk.Canvas(image_container, bg='white', width=600, height=400)
        
        # Scrollbars for image canvas
        img_v_scroll = ttk.Scrollbar(image_container, orient='vertical', command=self.image_canvas.yview)
        img_h_scroll = ttk.Scrollbar(image_container, orient='horizontal', command=self.image_canvas.xview)
        self.image_canvas.configure(yscrollcommand=img_v_scroll.set, xscrollcommand=img_h_scroll.set)
        
        # Grid layout for image canvas and scrollbars
        self.image_canvas.grid(row=0, column=0, sticky='nsew')
        img_v_scroll.grid(row=0, column=1, sticky='ns')
        img_h_scroll.grid(row=1, column=0, sticky='ew')
        
        # Configure grid weights
        image_container.grid_rowconfigure(0, weight=1)
        image_container.grid_columnconfigure(0, weight=1)
        
        # Bind mouse events for panning
        self.image_canvas.bind("<Button-1>", self.start_pan)
        self.image_canvas.bind("<B1-Motion>", self.pan_image)
        
        # Right side - 3D Point Cloud with Open3D button
        pcl_frame = ttk.LabelFrame(display_frame, text="3D Point Cloud")
        pcl_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Button to open Open3D visualization
        self.open3d_button = ttk.Button(
            pcl_frame,
            text="Open 3D Viewer",
            command=self.open_3d_viewer
        )
        self.open3d_button.pack(pady=20)
        
        # Status label for point cloud
        self.pcl_status_label = ttk.Label(
            pcl_frame,
            text="Load data to view point cloud",
            foreground="gray"
        )
        self.pcl_status_label.pack(pady=10)
        
        # Initialize variables for image handling
        self.current_image = None
        self.image_scale = 0.2  # Start at 20% for large images
        self.image_id = None
        self.current_point_cloud = None
    
    def start_pan(self, event):
        """Start panning the image"""
        self.image_canvas.scan_mark(event.x, event.y)
    
    def pan_image(self, event):
        """Pan the image"""
        self.image_canvas.scan_dragto(event.x, event.y, gain=1)
    
    def zoom_in(self):
        """Zoom in the image"""
        if self.current_image is None:
            return
        self.image_scale *= 1.2
        self.image_scale = min(self.image_scale, 2.0)  # Max 200%
        self.update_image_display()
    
    def zoom_out(self):
        """Zoom out the image"""
        if self.current_image is None:
            return
        self.image_scale /= 1.2
        self.image_scale = max(self.image_scale, 0.1)  # Min 10%
        self.update_image_display()
    
    def update_image_display(self):
        """Update the image display with current scale"""
        if self.current_image is None:
            return
        
        # Resize image
        height, width = self.current_image.shape[:2]
        new_width = int(width * self.image_scale)
        new_height = int(height * self.image_scale)
        
        resized_image = cv2.resize(self.current_image, (new_width, new_height))
        
        # Convert to RGB and then to PhotoImage
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        pil_image = Image.fromarray(rgb_image)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        if self.image_id:
            self.image_canvas.delete(self.image_id)
        
        self.image_id = self.image_canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
    
    def open_3d_viewer(self):
        """Open Open3D visualization window"""
        if self.current_point_cloud is None:
            messagebox.showwarning("No Data", "Please load image and point cloud data first")
            return
        
        try:
            # Debug: Check point cloud data
            points = np.asarray(self.current_point_cloud.points)
            colors = np.asarray(self.current_point_cloud.colors)
            print(f"Opening viewer with {len(points)} points")
            print(f"Point cloud bounds: X[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], Y[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], Z[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
            print(f"Has colors: {len(colors) > 0}")
            print(f"Color range: [{colors.min():.3f}, {colors.max():.3f}]")
            
            if len(points) == 0:
                messagebox.showerror("Error", "Point cloud is empty")
                return
            
            # Test with a simple point cloud first
            print("Creating test point cloud...")
            test_pcd = o3d.geometry.PointCloud()
            test_points = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,1]])
            test_colors = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1]])
            test_pcd.points = o3d.utility.Vector3dVector(test_points)
            test_pcd.colors = o3d.utility.Vector3dVector(test_colors)
            
            # Show both test and actual point cloud
            geometries = [self.current_point_cloud, test_pcd]
            
            # Use the simple draw_geometries function for automatic camera setup
            print("Opening visualization...")
            o3d.visualization.draw_geometries(
                geometries,
                window_name="Point Cloud Viewer",
                width=800,
                height=600,
                point_show_normal=False
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open 3D viewer:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_image_pcl_clicked(self):
        """Handle Load Image + PCL button click"""
        # Get selected row
        selected_items = self.csv_tree.selection()
        if not selected_items:
            messagebox.showwarning("No Selection", "Please select a row from the table first")
            return
        
        # Get the selected row data
        item = selected_items[0]
        row_values = self.csv_tree.item(item, 'values')
        
        if not row_values:
            messagebox.showerror("Error", "No data in selected row")
            return
        
        try:
            # Parse row data
            image_path = row_values[0]  # image_path
            camera_type = row_values[1]  # camera_type
            scene_id = row_values[2]  # scene_id
            image_index = row_values[3]  # image_index
            depth_scale = row_values[45]
            
            print(f"Loading: {image_path}")
            print(f"Camera: {camera_type}, Scene: {scene_id}, Index: {image_index}")
            
            # Find corresponding depth image path
            rgb_dir = os.path.dirname(image_path)
            dataset_dir = os.path.dirname(rgb_dir)
            
            # Construct depth image path based on camera type
            if camera_type == "photoneo":
                depth_dir = os.path.join(dataset_dir, "depth_photoneo")
            elif camera_type.startswith("cam"):
                depth_dir = os.path.join(dataset_dir, f"depth_{camera_type}")
            else:
                # Try generic depth folder names
                depth_dir = os.path.join(dataset_dir, f"depth_cam{camera_type}")
                if not os.path.exists(depth_dir):
                    depth_dir = os.path.join(dataset_dir, "depth")
            
            image_filename = os.path.basename(image_path)
            depth_path = os.path.join(depth_dir, image_filename)
            
            print(f"Looking for depth at: {depth_path}")
            
            # Check if files exist
            if not os.path.exists(image_path):
                messagebox.showerror("Error", f"RGB image not found:\n{image_path}")
                return
            
            if not os.path.exists(depth_path):
                messagebox.showerror("Error", f"Depth image not found:\n{depth_path}")
                return
            
            # Load camera parameters from scene_camera file
            k_matrix = self.load_camera_parameters(dataset_dir, camera_type, image_index)
            if k_matrix is None:
                messagebox.showerror("Error", "Could not load camera parameters")
                return
            
            print(f"K matrix:\n{k_matrix}")
            
            # Load and display RGB image
            self.load_and_display_image(image_path)
            
            # Create and display point cloud
            self.create_and_display_pointcloud(depth_path, image_path, k_matrix, depth_scale)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image and point cloud:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_camera_parameters(self, dataset_dir, camera_type, image_index):
        """Load camera parameters from scene_camera file"""
        try:
            # Construct scene_camera filename
            if camera_type == "photoneo":
                scene_camera_file = os.path.join(dataset_dir, "scene_camera_photoneo.json")
            elif camera_type.startswith("cam"):
                scene_camera_file = os.path.join(dataset_dir, f"scene_camera_{camera_type}.json")
            else:
                # Try numeric camera ID
                scene_camera_file = os.path.join(dataset_dir, f"scene_camera_cam{camera_type}.json")
            
            print(f"Loading camera params from: {scene_camera_file}")
            
            if not os.path.exists(scene_camera_file):
                print(f"Camera file not found, trying alternative...")
                # Try alternative naming
                scene_camera_file = os.path.join(dataset_dir, f"scene_camera_{camera_type}.json")
                if not os.path.exists(scene_camera_file):
                    return None
            
            with open(scene_camera_file, 'r') as f:
                scene_cameras = json.load(f)
            
            # Get camera parameters for this image
            if str(image_index) in scene_cameras:
                cam_params = scene_cameras[str(image_index)]
            elif int(image_index) in scene_cameras:
                cam_params = scene_cameras[int(image_index)]
            else:
                print(f"Image index {image_index} not found in camera file")
                return None
            
            # Extract K matrix
            k_matrix = np.array(cam_params["cam_K"]).reshape(3, 3)
            return k_matrix
            
        except Exception as e:
            print(f"Error loading camera parameters: {e}")
            return None
    
    def load_and_display_image(self, image_path):
        """Load and display RGB image at original size"""
        try:
            # Load image at original size
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                raise ValueError("Could not load image")
            
            # Start at 20% scale for large images (3840x2160)
            height, width = self.current_image.shape[:2]
            if width > 2000:  # Large image
                self.image_scale = 0.75
            elif width > 1000:  # Medium image
                self.image_scale = 0.8
            else:  # Small image
                self.image_scale = 1.0
            
            print(f"Loaded image: {width}x{height}, starting scale: {self.image_scale}")
            
            # Display image
            self.update_image_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def create_and_display_pointcloud(self, depth_path, rgb_path, k_matrix, depth_scale):
        """Create and display 3D point cloud using Open3D"""
        try:
            print(f"Creating point cloud from depth: {depth_path}")
            
            # Check depth image properties first
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                raise ValueError("Could not load depth image")
            
            print(f"Depth image shape: {depth_image.shape}")
            print(f"Depth image dtype: {depth_image.dtype}")
            print(f"Depth min/max: {depth_image.min()}/{depth_image.max()}")
            
            # Create point cloud using the provided function
            point_cloud_original, colors_original = create_pointcloud_with_colors(
                depth_path=depth_path,
                rgb_path=rgb_path,
                k_matrix=k_matrix,
                depth_scale=depth_scale,
                use_gpu=False,
            )
            
            print(f"Original point cloud created with {len(point_cloud_original)} points")
            
            if len(point_cloud_original) > 0:
                # Filter out invalid points (inf, nan, zero depth)
                valid_mask = np.isfinite(point_cloud_original).all(axis=1)
                valid_mask &= (point_cloud_original[:, 2] > 0)  # Positive Z values
                
                if np.sum(valid_mask) > 0:
                    valid_points = point_cloud_original[valid_mask]
                    valid_colors = colors_original[valid_mask]
                    
                    print(f"Filtered {len(valid_points)} valid points from {len(point_cloud_original)} total")
                    
                    # Create Open3D point cloud
                    self.current_point_cloud = o3d.geometry.PointCloud()
                    self.current_point_cloud.points = o3d.utility.Vector3dVector(valid_points)
                    self.current_point_cloud.colors = o3d.utility.Vector3dVector(valid_colors / 255.0)
                    
                    print(f"Open3D point cloud created with {len(self.current_point_cloud.points)} points")
                    print(f"Point range - X: {valid_points[:, 0].min():.3f} to {valid_points[:, 0].max():.3f}")
                    print(f"Point range - Y: {valid_points[:, 1].min():.3f} to {valid_points[:, 1].max():.3f}")
                    print(f"Point range - Z: {valid_points[:, 2].min():.3f} to {valid_points[:, 2].max():.3f}")
                    print(f"Color range: {valid_colors.min():.1f} to {valid_colors.max():.1f}")
                    
                    # Update status
                    self.pcl_status_label.config(
                        text=f"Point cloud ready ({len(self.current_point_cloud.points)} points)\nClick 'Open 3D Viewer' to visualize",
                        foreground="green"
                    )
                    
                    # Enable the Open3D button
                    self.open3d_button.config(state='normal')
                else:
                    self.current_point_cloud = None
                    self.pcl_status_label.config(
                        text="No valid points after filtering",
                        foreground="red"
                    )
                
            else:
                self.current_point_cloud = None
                self.pcl_status_label.config(
                    text="No valid points found - check depth scale and image format",
                    foreground="red"
                )
                print("No valid points found - check depth scale and image format")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create point cloud:\n{str(e)}")
            print(f"Point cloud error: {e}")
            import traceback
            traceback.print_exc()
            
            # Update status on error
            self.pcl_status_label.config(
                text="Error loading point cloud",
                foreground="red"
            )
            self.current_point_cloud = None
    
    def process_dataset_thread(self, dataset_path, models_path, output_csv_path):
        """Process dataset in separate thread"""
        try:
            # Call the read_dataset function
            result = read_dataset(dataset_path, models_path, output_csv_path)
            
            # Update UI in main thread
            self.root.after(0, self.dataset_processing_complete, result, output_csv_path)
            
        except Exception as e:
            # Handle unexpected errors
            error_result = {
                'success': False,
                'message': f"Unexpected error: {str(e)}",
                'total_rows': 0
            }
            self.root.after(0, self.dataset_processing_complete, error_result, output_csv_path)
    
    def dataset_processing_complete(self, result, output_csv_path):
        """Handle completion of dataset processing"""
        # Re-enable button
        self.read_button.config(state='normal')
        
        if result['success']:
            # Success
            self.current_csv_path = output_csv_path
            self.status_label.config(
                text=f"✓ Complete! {result['total_rows']} entries created in {os.path.basename(output_csv_path)}", 
                foreground="green"
            )
            # Load and display the CSV
            self.load_and_display_csv(output_csv_path)
            messagebox.showinfo(
                "Success", 
                f"Dataset processed successfully!\n\n"
                f"Total entries: {result['total_rows']}\n"
                f"Output file: {output_csv_path}"
            )
        else:
            # Error
            self.status_label.config(text="✗ Error occurred", foreground="red")
            messagebox.showerror("Error", result['message'])
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = TabsUI()
    app.run()