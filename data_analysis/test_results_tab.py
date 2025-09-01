import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import csv
import cv2
import numpy as np
import json
from PIL import Image, ImageTk, ImageDraw, ImageFont
import open3d as o3d
from read_test_dataset import read_test_dataset
from pxl_2_point import create_pointcloud_with_colors
import sys
from process_scene_data import process_scene_data
from collections import defaultdict, Counter
import time

import gc

class TestResultsTab:
    def __init__(self, notebook):
        self.notebook = notebook
        
        # Create the tab frame
        self.frame = ttk.Frame(notebook)
        self.notebook.add(self.frame, text="Visualize Test Results")
        
        # Initialize paths for test results tab
        self.test_dataset_path = ""
        self.current_test_csv_path = ""
        
        # Initialize variables for image handling
        self.current_image = None
        self.current_image_with_masks = None
        self.image_scale = 0.2  # Start at 20% for large images
        self.image_id = None
        self.current_point_cloud = None
        
        # New variables for detection results navigation
        self.detection_results = None
        self.detection_images = []
        self.detection_polygons = []
        self.detection_polygons_object_correspondence = [] # index of the consolidated objects results the polygon corresponds to
        self.detection_camera_ids = []
        self.detection_image_types = []  # Track whether each image is 'detection' or 'correspondence'
        self.current_detection_index = 0
        self.current_image_with_detections = None

        # Store coordinate frames for visualization - SIMPLE LIST
        self.detected_poses_transformation_matrix = []  # Store coordinate frames for detection results
        
        # Initialize image zoom tracking
        self.image_zooms = {}  # Track zoom level for each image
        self.image_photos = {}  # Keep references to PhotoImages
        
        # Setup the tab content
        self.setup_tab()
    
    def setup_tab(self):
        # Frame to hold buttons side by side
        buttons_frame = ttk.Frame(self.frame)
        buttons_frame.pack(anchor='nw', padx=10, pady=10)
        
        # Dataset Path button
        dataset_path_button = ttk.Button(
            buttons_frame, 
            text="Dataset Path", 
            command=self.browse_test_dataset_path
        )
        dataset_path_button.pack(side='left', padx=(0, 10))
        
        # Read Dataset button - MAKE SURE IT CALLS THE TEST METHOD
        self.test_read_button = ttk.Button(
            buttons_frame, 
            text="Process Test Dataset", 
            command=self.process_test_dataset_clicked
        )
        self.test_read_button.pack(side='left')
        
        # Frame to hold path label
        paths_frame = ttk.Frame(self.frame)
        paths_frame.pack(anchor='nw', padx=10, pady=5)
        
        # Label to display selected dataset path
        self.test_dataset_path_label = ttk.Label(
            paths_frame, 
            text="No dataset path selected", 
            wraplength=700,
            foreground="gray"
        )
        self.test_dataset_path_label.pack(side='left')
        
        # Status label for showing progress/results
        self.test_status_label = ttk.Label(
            self.frame,
            text="Select dataset path to begin",
            foreground="blue"
        )
        self.test_status_label.pack(anchor='nw', padx=10, pady=10)
        
        # Setup CSV Display Area for test results
        self.setup_test_csv_display()

    def browse_test_dataset_path(self):
        """Browse and select dataset path for test results"""
        folder_path = filedialog.askdirectory(title="Select Dataset Path for Test Results")
        if folder_path:
            self.test_dataset_path = folder_path
            self.test_dataset_path_label.config(
                text=f"Dataset Path: {folder_path}", 
                foreground="black"
            )
            self.update_test_status()
        else:
            self.test_dataset_path_label.config(
                text="No dataset path selected", 
                foreground="gray"
            )

    def update_test_status(self):
        """Update status based on selected dataset path"""
        if self.test_dataset_path:
            self.test_status_label.config(
                text="✓ Dataset path selected - ready to process",
                foreground="green"
            )
        else:
            self.test_status_label.config(
                text="Select dataset path to begin",
                foreground="blue"
            )

    def setup_test_csv_display(self):
        """Setup the CSV display table with scrollbars for test results"""
        # Label for CSV display
        csv_label = ttk.Label(self.frame, text="Test Results CSV Content:")
        csv_label.pack(anchor='nw', padx=10, pady=(20, 5))
        
        # Frame for the table and scrollbars - fixed height, full width
        table_frame = ttk.Frame(self.frame)
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create Treeview for displaying CSV data
        self.test_csv_tree = ttk.Treeview(table_frame, height=10)
        
        # Vertical scrollbar
        test_v_scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.test_csv_tree.yview)
        self.test_csv_tree.configure(yscrollcommand=test_v_scrollbar.set)
        
        # Horizontal scrollbar
        test_h_scrollbar = ttk.Scrollbar(table_frame, orient='horizontal', command=self.test_csv_tree.xview)
        self.test_csv_tree.configure(xscrollcommand=test_h_scrollbar.set)
        
        # Grid layout for table and scrollbars
        self.test_csv_tree.grid(row=0, column=0, sticky='nsew')
        test_v_scrollbar.grid(row=0, column=1, sticky='ns')
        test_h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        # Configure grid weights for resizing
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        # Frame for action buttons (Load Image + PCL, Run Detection, Open 3D Viewer)
        action_buttons_frame = ttk.Frame(self.frame)
        action_buttons_frame.pack(anchor='nw', padx=10, pady=10)
        
        # Load image + PCL button
        self.load_button = ttk.Button(
            action_buttons_frame,
            text="Load Image + PCL",
            command=self.load_image_pcl_clicked
        )
        self.load_button.pack(side='left', padx=(0, 10))
        
        # Run Detection button
        self.detect_button = ttk.Button(
            action_buttons_frame,
            text="Run Detection",
            command=self.run_detection_clicked
        )
        self.detect_button.pack(side='left', padx=(0, 10))
        
        # Open 3D Viewer button
        self.open3d_button = ttk.Button(
            action_buttons_frame,
            text="Open 3D Viewer",
            command=self.open_3d_viewer
        )
        self.open3d_button.pack(side='left')

        self.pcl_status_label = ttk.Label(
            action_buttons_frame,
            text="Load data to view results",
            foreground="gray"
        )
        self.pcl_status_label.pack(side='right')
        
        # Setup image and results display area
        self.setup_display_area()

    def load_and_display_test_csv(self, csv_path):
        """Load CSV file and display in the test results treeview"""
        try:
            # Clear existing data
            for item in self.test_csv_tree.get_children():
                self.test_csv_tree.delete(item)
            
            # Read CSV file
            with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                
                # Read header
                headers = next(csv_reader)
                
                # Configure treeview columns
                self.test_csv_tree['columns'] = headers
                self.test_csv_tree['show'] = 'headings'
                
                # Configure column headings and widths
                for header in headers:
                    self.test_csv_tree.heading(header, text=header)
                    # Set reasonable column width
                    self.test_csv_tree.column(header, width=120, minwidth=80)
                
                # Read and insert data rows
                row_count = 0
                for row in csv_reader:
                    # Ensure row has same number of columns as headers
                    while len(row) < len(headers):
                        row.append('')
                    
                    self.test_csv_tree.insert('', 'end', values=row)
                    row_count += 1
                
                # Update status with row count
                current_status = self.test_status_label.cget('text')
                if "✓" in current_status:
                    self.test_status_label.config(
                        text=f"{current_status} | Displaying {row_count} rows",
                        foreground="green"
                    )
                else:
                    self.test_status_label.config(
                        text=f"CSV loaded successfully | Displaying {row_count} rows",
                        foreground="green"
                    )
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file:\n{str(e)}")
            self.test_status_label.config(text="✗ Error loading CSV", foreground="red")

    def process_test_dataset_clicked(self):
        """Handle Process Test Dataset button click - COMPLETELY SEPARATE from ground truth"""
        # Validate path is selected
        if not self.test_dataset_path:
            messagebox.showerror("Error", "Please select Dataset Path first")
            return
        
        # Create meta_data folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        meta_data_dir = os.path.join(script_dir, "meta_data")
        os.makedirs(meta_data_dir, exist_ok=True)
        
        # Check if test_dataset.csv already exists
        test_csv_path = os.path.join(meta_data_dir, "test_dataset.csv")
        
        if os.path.exists(test_csv_path):
            # Show dialog asking user what to do
            choice = self.ask_existing_test_csv_choice()
            
            if choice == "use_existing":
                # Use existing CSV
                self.current_test_csv_path = test_csv_path
                self.test_status_label.config(
                    text="✓ Using existing test_dataset.csv", 
                    foreground="green"
                )
                # Load and display the CSV
                self.load_and_display_test_csv(test_csv_path)
                messagebox.showinfo(
                    "Using Existing CSV", 
                    f"Using existing test dataset CSV:\n{test_csv_path}"
                )
                return
            elif choice == "create_new":
                # Continue with creating new CSV
                pass
            else:
                # User cancelled
                return
        
        # Disable button and show processing status
        self.test_read_button.config(state='disabled')
        self.test_status_label.config(text="Processing test dataset... Please wait", foreground="orange")
        
        # Start processing in separate thread to avoid blocking UI
        thread = threading.Thread(
            target=self.process_test_dataset_thread,
            args=(self.test_dataset_path, test_csv_path)
        )
        thread.daemon = True
        thread.start()

    def ask_existing_test_csv_choice(self):
        """Show dialog asking user what to do with existing test CSV"""
        # Create custom dialog
        dialog = tk.Toplevel()
        dialog.title("Test Dataset CSV Exists")
        dialog.geometry("400x150")
        dialog.resizable(False, False)
        
        # Center the dialog
        dialog.transient()
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
        message = "test_dataset.csv already exists in meta_data folder.\n\nWhat would you like to do?"
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

    def process_test_dataset_thread(self, dataset_path, output_csv_path):
        """Process dataset in separate thread"""
        try:
            # Call the read_dataset function
            result = read_test_dataset(dataset_path, output_csv_path)
            
            # Update UI in main thread
            # Note: We need to get the root window reference
            root = self.notebook.nametowidget(self.notebook.winfo_toplevel())
            root.after(0, self.test_dataset_processing_complete, result, output_csv_path)
            
        except Exception as e:
            # Handle unexpected errors
            error_result = {
                'success': False,
                'message': f"Unexpected error: {str(e)}",
                'total_rows': 0
            }
            root = self.notebook.nametowidget(self.notebook.winfo_toplevel())
            root.after(0, self.test_dataset_processing_complete, error_result, output_csv_path)

    def test_dataset_processing_complete(self, result, output_csv_path):
        """Handle completion of test dataset processing"""
        # Re-enable button
        self.test_read_button.config(state='normal')
        
        if result['success']:
            # Success
            self.current_test_csv_path = output_csv_path
            self.test_status_label.config(
                text=f"✓ Complete! {result['total_rows']} entries created in test_dataset.csv", 
                foreground="green"
            )
            # Load and display the CSV
            self.load_and_display_test_csv(output_csv_path)
            messagebox.showinfo(
                "Success", 
                f"Test dataset processed successfully!\n\n"
                f"Total entries: {result['total_rows']}\n"
                f"Output file: {output_csv_path}"
            )
        else:
            # Error
            self.test_status_label.config(text="✗ Error occurred", foreground="red")
            messagebox.showerror("Error", result['message'])

    def setup_display_area(self):
        """Setup the image and results display area"""
        # Frame for image and results side by side
        display_frame = ttk.Frame(self.frame)
        display_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left side - Image display with reasonable size for large images
        image_frame = ttk.LabelFrame(display_frame, text="RGB Image")
        image_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Camera ID label (hidden by default)
        self.camera_id_label = ttk.Label(
            image_frame, 
            text="", 
            font=("Arial", 10, "bold"), 
            foreground="blue"
        )
        self.camera_id_label.pack(anchor='nw', padx=5, pady=2)
        
        # Navigation and zoom buttons frame positioned above the image canvas
        controls_frame = ttk.Frame(image_frame)
        controls_frame.pack(anchor='ne', padx=5, pady=2)
        
        # Left/Right navigation buttons (hidden by default)
        nav_frame = ttk.Frame(controls_frame)
        nav_frame.pack(side='left', padx=(0, 10))
        
        self.left_btn = tk.Button(
            nav_frame, 
            text="◀", 
            font=("Arial", 10, "bold"), 
            command=self.navigate_left, 
            width=3, 
            height=1
        )
        self.left_btn.pack(side='left', padx=(0, 2))
        
        self.right_btn = tk.Button(
            nav_frame, 
            text="▶", 
            font=("Arial", 10, "bold"), 
            command=self.navigate_right, 
            width=3, 
            height=1
        )
        self.right_btn.pack(side='left')
        
        # Hide navigation buttons initially
        self.left_btn.pack_forget()
        self.right_btn.pack_forget()
        
        # Zoom buttons frame
        zoom_buttons_frame = ttk.Frame(controls_frame)
        zoom_buttons_frame.pack(side='right')
        
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
        
        # Container for image canvas
        image_container = ttk.Frame(image_frame)
        image_container.pack(fill='both', expand=True)
        
        # Canvas for image with scrollbars
        self.image_canvas = tk.Canvas(image_container, bg='white', width=100, height=400)
        
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

        # Right side - Results (text only now)
        results_frame = ttk.LabelFrame(display_frame, text="Results")
        results_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Create scrollable frame for results content
        results_canvas = tk.Canvas(results_frame)
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=results_canvas.yview)
        self.results_scrollable_frame = ttk.Frame(results_canvas)
        
        self.results_scrollable_frame.bind(
            "<Configure>",
            lambda e: results_canvas.configure(scrollregion=results_canvas.bbox("all"))
        )
        
        results_canvas.create_window((0, 0), window=self.results_scrollable_frame, anchor="nw")
        results_canvas.configure(yscrollcommand=results_scrollbar.set)
        
        # Bind mouse wheel for scrolling
        def _on_mousewheel(event):
            results_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _on_mousewheel_linux(event):
            results_canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
        
        results_canvas.bind("<MouseWheel>", _on_mousewheel)  # Windows
        results_canvas.bind("<Button-4>", _on_mousewheel_linux)  # Linux
        results_canvas.bind("<Button-5>", _on_mousewheel_linux)  # Linux
        
        results_canvas.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")
        
        # Time taken label
        self.time_taken_label = ttk.Label(
            self.results_scrollable_frame,
            text="",
            font=("Arial", 12),
            anchor='w'
        )
        self.time_taken_label.pack(pady=10, anchor='w')
        
        # MaskRCNN detections label
        self.maskrcnn_label = ttk.Label(
            self.results_scrollable_frame,
            text="",
            font=("Arial", 10),
            anchor='w'
        )
        self.maskrcnn_label.pack(anchor='w')
        
        # Consolidated objects label
        self.objects_label = ttk.Label(
            self.results_scrollable_frame,
            text="",
            font=("Arial", 10),
            anchor='w',
            wraplength=300
        )
        self.objects_label.pack(anchor='w', pady=(5, 10))

        # Results summary label - NEW
        self.results_summary_label = ttk.Label(
            self.results_scrollable_frame,
            text="",
            font=("Arial", 9),
            anchor='w',
            wraplength=700,
            justify='left'
        )
        self.results_summary_label.pack(anchor='w', pady=(10, 0))

    def navigate_left(self):
        """Navigate to previous detection image"""
        if len(self.detection_images) > 1:
            self.current_detection_index = (self.current_detection_index - 1) % len(self.detection_images)
            self.display_current_detection_image()

    def navigate_right(self):
        """Navigate to next detection image"""
        if len(self.detection_images) > 1:
            self.current_detection_index = (self.current_detection_index + 1) % len(self.detection_images)
            self.display_current_detection_image()

    def display_current_detection_image(self):
        """Display the current detection image with polygons or correspondence image"""
        if self.current_detection_index < len(self.detection_images):
            image = self.detection_images[self.current_detection_index]
            image_type = self.detection_image_types[self.current_detection_index] if self.current_detection_index < len(self.detection_image_types) else 'detection'
            
            if image_type == 'detection':
                # Handle detection result images with polygons
                if (self.current_detection_index < len(self.detection_polygons) and 
                    self.current_detection_index < len(self.detection_camera_ids)):
                    
                    polygons = self.detection_polygons[self.current_detection_index]
                    camera_id = self.detection_camera_ids[self.current_detection_index]
                    
                    # Draw polygons on image
                    self.current_image_with_detections = self.draw_polygons_on_image(image, polygons, self.current_detection_index)
                    
                    # Update camera label
                    self.camera_id_label.config(text=f"Detection Result - Camera: {camera_id}")
                else:
                    # Detection image without polygons
                    self.current_image_with_detections = image
                    self.camera_id_label.config(text=f"Detection Result {self.current_detection_index + 1}")
            else:
                # Handle SuperGlue correspondence images
                self.current_image_with_detections = image
                correspondence_index = self.current_detection_index - len([t for t in self.detection_image_types[:self.current_detection_index] if t == 'detection']) + 1
                total_correspondences = len([t for t in self.detection_image_types if t == 'correspondence'])
                self.camera_id_label.config(text=f"SuperGlue Correspondence {correspondence_index}/{total_correspondences}")
            
            # Update image display
            self.update_image_display()

    def display_current_viz_image(self):
        """Display the current SuperGlue correspondence image"""
        if self.current_viz_index < len(self.viz_images):
            # Set the current image for display
            self.current_image_with_detections = self.viz_images[self.current_viz_index]
            
            # Update camera label to show viz image info
            self.camera_id_label.config(text=f"SuperGlue Correspondence {self.current_viz_index + 1}/{len(self.viz_images)}")
            
            # Update image display
            self.showing_viz_images = True
            self.update_image_display()

    def show_superglue_correspondences_clicked(self):
        """Handle Show SuperGlue Correspondences button click"""
        # Check if viz_images is empty
        if not self.viz_images:
            messagebox.showwarning("No Correspondences", "No SuperGlue correspondence images available.")
            return
        
        # Reset index and show first image
        self.current_viz_index = 0
        self.showing_viz_images = True
        
        # Show navigation buttons if multiple images
        if len(self.viz_images) > 1:
            self.left_btn.pack(side='left', padx=(0, 2))
            self.right_btn.pack(side='left')
        else:
            self.left_btn.pack_forget()
            self.right_btn.pack_forget()
        
        # Display first viz image
        self.display_current_viz_image()

    def load_image_pcl_clicked(self):
        """Handle Load Image + PCL button click"""
        # Get selected row
        selected_items = self.test_csv_tree.selection()

        if not selected_items:
            messagebox.showwarning("No Selection", "Please select a row from the table first")
            return
        
        # Get the selected row data
        item = selected_items[0]
        row_values = self.test_csv_tree.item(item, 'values')
        
        if not row_values:
            messagebox.showerror("Error", "No data in selected row")
            return
        
        try:
            # Parse row data
            image_path = row_values[0]  # image_path
            camera_type = row_values[1]  # camera_type
            scene_id = row_values[2]  # scene_id
            image_index = row_values[3]  # image_index
            depth_scale = row_values[45] if len(row_values) > 45 else "1000"
            
            # print(f"Loading: {image_path}")
            # print(f"Camera: {camera_type}, Scene: {scene_id}, Index: {image_index}")
            
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
            
            # print(f"Looking for depth at: {depth_path}")
            
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
            
            # print(f"K matrix:\n{k_matrix}")
            
            # Load and display RGB image
            self.load_and_display_image(image_path)
            
            # Create and display point cloud
            self.create_and_display_pointcloud(depth_path, image_path, k_matrix, depth_scale)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image and point cloud:\n{str(e)}")
            import traceback
            traceback.print_exc()

    
    def create_coordinate_frame_from_pose(self, R_matrix, T_vector, size=0.05):
        """Create a coordinate frame marker from rotation matrix and translation vector"""
        try:
            # Create coordinate frame at origin
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
            
            # Create transformation matrix (4x4 homogeneous)
            transformation = np.eye(4)
            transformation[:3, :3] = R_matrix  # Set rotation part
            transformation[:3, 3] = T_vector   # Set translation part
            
            # Apply the full transformation at once
            frame.transform(transformation)
            
            # Debug print
            # print(f"Applied transformation with T: {T_vector}")
            
            return frame
        except Exception as e:
            # print(f"Error creating coordinate frame: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_detection_clicked(self):
        """Handle Run Detection button click - Enhanced with point cloud and poses"""
        selected_items = self.test_csv_tree.selection()

        if not selected_items:
            messagebox.showwarning("No Selection", "Please select a row from the table first")
            return

        # Clear previous results
        self.clear_results_display()

        item = selected_items[0]
        row_values = self.test_csv_tree.item(item, 'values')
        headers = self.test_csv_tree['columns']
        selected_row = dict(zip(headers, row_values))
        
        selected_scene_id = selected_row['scene_id']
        selected_image_id = selected_row['image_index']
        
        # Now iterate through treeview like csv.DictReader
        headers = self.test_csv_tree['columns']
        grouped_data = defaultdict(list)
        
        for item_id in self.test_csv_tree.get_children():
            row_values = self.test_csv_tree.item(item_id, 'values')
            row = dict(zip(headers, row_values))  # Same as csv.DictReader row
            
            scene_id = row['scene_id']
            image_id = row['image_index']
            key = (scene_id, image_id)
            grouped_data[key].append(row)
        
        # Access the group for selected row
        selected_key = (selected_scene_id, selected_image_id)
        matching_rows = grouped_data[selected_key]

        # Find photoneo camera data for point cloud creation
        photoneo_row = None
        for row in matching_rows:
            if row['camera_type'] == 'photoneo':
                photoneo_row = row
                break
        
        if photoneo_row is None:
            messagebox.showerror("Error", "No photoneo camera data found in selected scene. Cannot create world coordinate point cloud.")
            return

        start_time = time.time()
        scene_info = process_scene_data(matching_rows)
        start_time = time.time()
        scene_info.mask_objects()
        end = time.time()
        print(f"MaskRcnn took: {end - start_time:.6f} seconds")
        start_time = time.time()
        scene_info.consolidate_detections()
        end = time.time()
        print(f"Consolidating took: {end - start_time:.6f} seconds")
        start_time = time.time()
        scene_info.do_feature_matchings()
        end = time.time()
        print(f"Feature matching took: {end - start_time:.6f} seconds")
        start_time = time.time()
        scene_info.compute_6d_poses()
        end = time.time()
        print(f"6D pose computation took: {end - start_time:.6f} seconds")


        end_time = time.time()
        time_taken = end_time - start_time

        # Create mapping from row index to cluster number
        row_to_cluster = {row_idx: cluster_num 
                        for cluster_num, cluster in enumerate(scene_info.consolidated_detections) 
                        for row_idx in cluster}
        
        self.detection_polygons_object_correspondence = np.zeros(len(row_to_cluster), dtype=int)

        for row_idx, cluster_num in row_to_cluster.items():
            self.detection_polygons_object_correspondence[row_idx] = cluster_num+1

        # Display results
        self.display_detection_results(scene_info, time_taken)
        
        # Create point cloud with poses
        self.create_detection_pointcloud_with_poses(photoneo_row, scene_info)

        del scene_info
        gc.collect()

    

    def create_detection_pointcloud_with_poses(self, photoneo_row, scene_info):
        self.detected_poses_transformation_matrix = []
        """Create point cloud from photoneo data and add pose markers"""
        try:
            # Extract photoneo camera data
            image_path = photoneo_row['rgb_image_path']
            depth_scale = photoneo_row['depth_scale']
            image_index = photoneo_row['image_index']
            
            # print(f"Creating detection point cloud from photoneo: {image_path}")
            
            # Find corresponding depth image path
            rgb_dir = os.path.dirname(image_path)
            dataset_dir = os.path.dirname(rgb_dir)
            depth_dir = os.path.join(dataset_dir, "depth_photoneo")
            
            image_filename = os.path.basename(image_path)
            depth_path = os.path.join(depth_dir, image_filename)
            
            # print(f"Looking for photoneo depth at: {depth_path}")
            
            # Check if files exist
            if not os.path.exists(image_path):
                raise ValueError(f"Photoneo RGB image not found: {image_path}")
            
            if not os.path.exists(depth_path):
                raise ValueError(f"Photoneo depth image not found: {depth_path}")
            
            # Load camera parameters
            k_matrix = self.load_camera_parameters(dataset_dir, "photoneo", image_index)
            if k_matrix is None:
                raise ValueError("Could not load photoneo camera parameters")
            
            # print(f"Photoneo K matrix:\n{k_matrix}")

            # Check depth image properties first
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                raise ValueError("Could not load depth image")
            
            # print(f"Depth image shape: {depth_image.shape}")
            # print(f"Depth image dtype: {depth_image.dtype}")
            # print(f"Depth min/max: {depth_image.min()}/{depth_image.max()}")
            
            # Create base point cloud using existing function
            point_cloud_original, colors_original = create_pointcloud_with_colors(
                depth_path=depth_path,
                rgb_path=image_path,
                k_matrix=k_matrix,
                depth_scale=depth_scale,
                use_gpu=True,
            )
            
            # print(f"Original photoneo point cloud created with {len(point_cloud_original)} points")
            
            if len(point_cloud_original) > 0:
                # Filter out invalid points
                valid_mask = np.isfinite(point_cloud_original).all(axis=1)
                valid_mask &= (point_cloud_original[:, 2] > 0)  # Positive Z values
                
                if np.sum(valid_mask) > 0:
                    valid_points = point_cloud_original[valid_mask]
                    valid_colors = colors_original[valid_mask]
                    
                    # print(f"Filtered {len(valid_points)} valid points from {len(point_cloud_original)} total")


                    # Create the mask
                    mask = (np.abs(valid_points[:, 0]) < 4000) & (np.abs(valid_points[:, 1]) < 4000) & (np.abs(valid_points[:, 2]) < 4000)

                    # Filter before creating the point cloud
                    filtered_points = valid_points[mask]
                    filtered_colors = valid_colors[mask]

                    # print(f"Outliers filtered {len(filtered_points)} valid points from {len(point_cloud_original)} total")
                    
                    # Create Open3D point cloud
                    self.current_point_cloud = o3d.geometry.PointCloud()
                    self.current_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
                    self.current_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors / 255.0)
                    
                    results_summary = scene_info.display_results.get("results_summary", [])
                    
                    for i, result in enumerate(results_summary):
                        try:
                            R_test_world = result.get("R_test_world")
                            T_test_world = result.get("T_test_world")
                            object_id = result.get("object ID", "Unknown")
                            camera_id = result.get("camera ID", "Unknown")
                            # print("R_test_world: ", R_test_world)
                            # print("T_test_world: ", T_test_world)
                            if R_test_world is not None and T_test_world is not None:
                                 # Convert to numpy arrays

                                R_matrix = np.array(R_test_world, dtype=float)
                                T_vector = np.array(T_test_world, dtype=float)


                                # print("R_matrix: ", R_matrix)
                                # print("T_vector: ", T_vector)
                                
                                
                                # Create transformation matrix
                                transformation_matrix = np.eye(4)
                                transformation_matrix[:3, :3] = R_matrix
                                transformation_matrix[:3, 3] = T_vector

                                
                                self.detected_poses_transformation_matrix.append(transformation_matrix)
                                
                        except Exception as e:
                            # print(f"Error creating pose marker for result {i}: {e}")
                            continue
                    
                    # Update status
                    num_poses = len(self.detected_poses_transformation_matrix) 
                    self.pcl_status_label.config(
                        text=f"Detection point cloud ready ({len(self.current_point_cloud.points)} points, {num_poses} object poses)\nClick 'Open 3D Viewer' to visualize",
                        foreground="green"
                    )
                    
                    # print(f"Detection point cloud created with {len(self.current_point_cloud.points)} points and {num_poses} pose markers")
                    
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
                
        except Exception as e:
            # print(f"Error creating detection point cloud: {e}")
            import traceback
            traceback.print_exc()
            
            self.pcl_status_label.config(
                text="Error creating detection point cloud",
                foreground="red"
            )
            self.current_point_cloud = None
            messagebox.showerror("Error", f"Failed to create detection point cloud:\n{str(e)}")

    def open_3d_viewer(self):
        """Open Open3D visualization window - Enhanced for detection results with poses"""
        if self.current_point_cloud is None:
            messagebox.showwarning("No Data", "Please load image and point cloud data first")
            return
        
        try:

            
            # Start with the point cloud
            geometries = [self.current_point_cloud]

            # add simple origin frame (for basic point cloud loading)
            points = np.asarray(self.current_point_cloud.points)
            pc_min = points.min(axis=0)
            pc_max = points.max(axis=0)
            pc_size = np.max(pc_max - pc_min)
            # print("Test pc_size: ", pc_size)
            axis_size = pc_size * 0.05
            
            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
            origin_frame.translate([0, 0, 0])
            geometries.append(origin_frame)
            # print("Added basic origin coordinate frame")
            
            # Add coordinate frames if they exist (from detection results)
        
            # print("Adding detection coordinate frames to 3D viewer:")
            for i, frame in enumerate(self.detected_poses_transformation_matrix):
                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
                # print("Test transformation matrix: ", frame)
                coord_frame.transform(frame)
                geometries.append(coord_frame)
                center = np.asarray(coord_frame.get_center())
                # print(f"Frame {i} center: {center}")      
                # Add red sphere marker at original pose location
                pose_position = frame[:3, 3]
                marker_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size*0.01)
                marker_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red marker
                marker_sphere.translate(pose_position)
                geometries.append(marker_sphere)          
            # print(f"Added {len(self.detected_poses_transformation_matrix)} coordinate frames")
            
            
            # print(f"Total geometries to display: {len(geometries)}")
            
            # Determine window title based on content
            if hasattr(self, 'detected_poses_transformation_matrix') and self.detected_poses_transformation_matrix:
                window_title = "Detection Results with Object Poses"
            else:
                window_title = "Point Cloud Viewer"
            
            # Use the draw_geometries function
            # print(f"Opening {window_title}...")
            o3d.visualization.draw_geometries(
                geometries,
                window_name=window_title,
                width=1200,
                height=800,
                point_show_normal=False
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open 3D viewer:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def clear_results_display(self):
        """Clear all results display elements - Enhanced to clear detection coordinate frames"""
        self.time_taken_label.config(text="")
        self.maskrcnn_label.config(text="")
        self.objects_label.config(text="")
        self.results_summary_label.config(text="")
        
        # Clear detection data
        self.detection_images = []
        self.detection_polygons = []
        self.detection_camera_ids = []
        self.detection_image_types = []
        self.current_detection_index = 0
        self.current_image_with_detections = None
        
        # Clear detection coordinate frames
        self.detection_coordinate_frames = []
        
        # Hide navigation buttons and camera label
        self.left_btn.pack_forget()
        self.right_btn.pack_forget()
        self.camera_id_label.config(text="")

    def format_matrix_compact(self, matrix):
        """Format matrix in a compact readable format"""
        if matrix is None:
            return "None"
        
        matrix = np.array(matrix)
        if matrix.ndim == 1:
            # Translation vector
            formatted_values = [f"{val:.3f}" for val in matrix]
            return f"[{', '.join(formatted_values)}]"
        elif matrix.ndim == 2:
            # Rotation matrix
            rows = []
            for row in matrix:
                formatted_row = [f"{val:.3f}" for val in row]
                rows.append(f"[{', '.join(formatted_row)}]")
            return f"[{', '.join(rows)}]"
        else:
            return str(matrix)

    def display_detection_results(self, scene_info, time_taken):
        """Display detection results in the results box"""
        # Display time taken
        self.time_taken_label.config(text=f"Time taken: {time_taken:.3f} seconds")
        
        # Display MaskRCNN detections count
        nb_detections = scene_info.display_results.get("nb_maskrcnn_detections", 0)
        self.maskrcnn_label.config(text=f"Total MaskRCNN detections: {nb_detections}")

        
        
        # Display consolidated detected objects
        detected_objects = scene_info.display_results.get("detected_objects", [])
        if detected_objects:
            # Count clusters per object_id
            object_cluster_count = defaultdict(int)

            for cluster in scene_info.consolidated_detections:
                # Get the first row index from the cluster
                first_row_idx = cluster[0]
                
                # Get the object_id from that row in detections
                object_id = scene_info.detections[first_row_idx][0]
                
                # Increment the count for this object_id
                object_cluster_count[object_id] += 1

            # Convert to regular dict (optional)
            cluster_summary = dict(object_cluster_count)

            # Create formatted text string
            cluster_summary_text = ", ".join([f"obj {obj_id} x {count}" for obj_id, count in cluster_summary.items()])
            
            self.objects_label.config(text="Conslidated detections: " + cluster_summary_text)
            
        else:
            self.objects_label.config(text="No objects detected")
        
        # Display results summary - NEW
        results_summary = scene_info.display_results.get("results_summary", [])
        if results_summary:
            summary_text_parts = ["6D Pose Results:"]
            
            for i, result in enumerate(results_summary):
                camera_id = result.get("camera ID", "Unknown")
                object_id = result.get("object ID", "Unknown")
                rmse = result.get("rmse", 0)
                
                R_test_cam = result.get("R_test_cam")
                T_test_cam = result.get("T_test_cam")
                R_test_world = result.get("R_test_world")
                T_test_world = result.get("T_test_world")
                
                summary_text_parts.append(f"\n--- Result {i+1} ---")
                summary_text_parts.append(f"Camera: {camera_id}, Object: {object_id}")
                summary_text_parts.append(f"RMSE: {rmse:.3f}")
                summary_text_parts.append(f"R_test_cam: {self.format_matrix_compact(R_test_cam)}")
                summary_text_parts.append(f"T_test_cam: {self.format_matrix_compact(T_test_cam)}")
                summary_text_parts.append(f"R_test_world: {self.format_matrix_compact(R_test_world)}")
                summary_text_parts.append(f"T_test_world: {self.format_matrix_compact(T_test_world)}")
            
            summary_text = "\n".join(summary_text_parts)
            self.results_summary_label.config(text=summary_text)
        else:
            self.results_summary_label.config(text="No 6D pose results available")
        
        # Store detection images and polygons
        self.detection_images = scene_info.display_results.get("images", [])
        self.detection_polygons = scene_info.display_results.get("polygons", [])
        self.detection_camera_ids = scene_info.display_results.get("camera_ids", [])
        self.current_detection_index = 0
        
        # Initialize image types for detection images
        self.detection_image_types = ['detection'] * len(self.detection_images)
        
        # Append SuperGlue correspondence images automatically
        viz_images = scene_info.display_results.get("feature_matching_images", [])
        if viz_images:
            self.detection_images.extend(viz_images)
            self.detection_image_types.extend(['correspondence'] * len(viz_images))
            # For correspondence images, we don't have polygons or camera_ids, so extend with empty values
            self.detection_polygons.extend([[] for _ in viz_images])
            self.detection_camera_ids.extend(['' for _ in viz_images])
        
        # Show navigation buttons if multiple images (detection + correspondence)
        if len(self.detection_images) > 1:
            self.left_btn.pack(side='left', padx=(0, 2))
            self.right_btn.pack(side='left')
        
        # Display first image if available
        if self.detection_images:
            self.display_current_detection_image()

    def draw_polygons_on_image(self, image, polygons, cluster_idx):
        """Draw polygons on image and return the result"""
        image_copy = image.copy()
        
        # Colors for different polygons
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

        
        
        for i, polygon in enumerate(polygons):
            if len(polygon["points"]) >= 3:  # Need at least 3 points for a polygon
                try:
                    color = colors[i % len(colors)]
                    
                    # Convert polygon to numpy array and ensure it's the right shape
                    polygon_np = np.array(polygon["points"], dtype=np.int32)
                    if polygon_np.shape[1] == 2:  # Make sure it's (n, 2) shape
                        # Draw polygon outline
                        cv2.polylines(image_copy, [polygon_np], True, color, 2)
                        
                        # Optional: Fill polygon with semi-transparent color
                        overlay = image_copy.copy()
                        cv2.fillPoly(overlay, [polygon_np], color)
                        cv2.addWeighted(overlay, 0.3, image_copy, 0.7, 0, image_copy)
                        cv2.putText(image_copy, polygon["label"], tuple(polygon_np[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
                        cv2.circle(image_copy, polygon["geometric_center"], 5, (255, 255, 255), -1)
                        detection_idx = sum(len(cluster) for cluster in self.detection_polygons[:cluster_idx]) + i
                        cv2.putText(image_copy, "Result " + str(self.detection_polygons_object_correspondence[detection_idx]), tuple(polygon_np[int(len(polygon_np)/2)]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

                except Exception as e:
                    # print(f"Error drawing polygon {i}: {e}")
                    continue
        
        return image_copy
    
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
            
            # print(f"Loading camera params from: {scene_camera_file}")
            
            if not os.path.exists(scene_camera_file):
                # print(f"Camera file not found, trying alternative...")
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
                # print(f"Image index {image_index} not found in camera file")
                return None
            
            # Extract K matrix
            k_matrix = np.array(cam_params["cam_K"]).reshape(3, 3)
            return k_matrix
            
        except Exception as e:
            # print(f"Error loading camera parameters: {e}")
            return None

    def load_and_display_image(self, image_path):
        """Load and display RGB image without masks"""
        try:
            # Load image at original size
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                raise ValueError("Could not load image")
            
            # Reset detection state when loading fresh image
            self.current_image_with_detections = None
            self.detection_results = None
            self.detection_images = []
            self.detection_polygons = []
            self.detection_camera_ids = []
            self.detection_image_types = []
            self.current_detection_index = 0
            
            # Hide navigation buttons and clear camera label
            self.left_btn.pack_forget()
            self.right_btn.pack_forget()
            self.camera_id_label.config(text="")
            
            # Start at appropriate scale for different image sizes
            height, width = self.current_image.shape[:2]
            if width > 2000:  # Large image
                self.image_scale = 0.5
            elif width > 1000:  # Medium image
                self.image_scale = 0.75
            else:  # Small image
                self.image_scale = 1.0
            
            # print(f"Loaded test image: {width}x{height}, starting scale: {self.image_scale}")
            
            # Display image
            self.update_image_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")

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
        
        # Priority order: detection results (including correspondence), then original image
        if self.current_image_with_detections is not None:
            display_image = self.current_image_with_detections
        else:
            display_image = self.current_image
        
        # Resize image
        height, width = display_image.shape[:2]
        new_width = int(width * self.image_scale)
        new_height = int(height * self.image_scale)
        
        resized_image = cv2.resize(display_image, (new_width, new_height))
        
        # Convert to RGB and then to PhotoImage
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        if self.image_id:
            self.image_canvas.delete(self.image_id)
        
        self.image_id = self.image_canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))

    


    def create_and_display_pointcloud(self, depth_path, rgb_path, k_matrix, depth_scale):
        """Create and display 3D point cloud using Open3D"""
        try:
            # print(f"Creating point cloud from depth: {depth_path}")
            
            # Check depth image properties first
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                raise ValueError("Could not load depth image")
            
            # print(f"Depth image shape: {depth_image.shape}")
            # print(f"Depth image dtype: {depth_image.dtype}")
            # print(f"Depth min/max: {depth_image.min()}/{depth_image.max()}")
            
            # Create point cloud using the provided function
            point_cloud_original, colors_original = create_pointcloud_with_colors(
                depth_path=depth_path,
                rgb_path=rgb_path,
                k_matrix=k_matrix,
                depth_scale=depth_scale,
                use_gpu=False,
            )
            
            # print(f"Original point cloud created with {len(point_cloud_original)} points")
            
            if len(point_cloud_original) > 0:
                # Filter out invalid points (inf, nan, zero depth)
                valid_mask = np.isfinite(point_cloud_original).all(axis=1)
                valid_mask &= (point_cloud_original[:, 2] > 0)  # Positive Z values
                
                if np.sum(valid_mask) > 0:
                    valid_points = point_cloud_original[valid_mask]
                    valid_colors = colors_original[valid_mask]
                    
                    # print(f"Filtered {len(valid_points)} valid points from {len(point_cloud_original)} total")
                    
                    # Create Open3D point cloud
                    self.current_point_cloud = o3d.geometry.PointCloud()
                    self.current_point_cloud.points = o3d.utility.Vector3dVector(valid_points)
                    self.current_point_cloud.colors = o3d.utility.Vector3dVector(valid_colors / 255.0)
                    
                    # # print(f"Open3D point cloud created with {len(self.current_point_cloud.points)} points")
                    
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
                # print("No valid points found - check depth scale and image format")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create point cloud:\n{str(e)}")
            # print(f"Point cloud error: {e}")
            import traceback
            traceback.print_exc()
            
            # Update status on error
            self.pcl_status_label.config(
                text="Error loading point cloud",
                foreground="red"
            )
            self.current_point_cloud = None