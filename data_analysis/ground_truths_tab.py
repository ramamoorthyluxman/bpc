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
from PIL import Image, ImageTk, ImageDraw, ImageFont
from collections import Counter
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'helpers')))
from train_new_dataset import TrainNewDataset

class GroundTruthsTab:
    def __init__(self, notebook):
        self.notebook = notebook
        
        # Create the tab frame
        self.frame = ttk.Frame(notebook)
        self.notebook.add(self.frame, text="Visualize Ground Truths")
        
        # Initialize paths
        self.dataset_path = ""
        self.models_path = ""
        self.current_csv_path = ""
        
        # Initialize variables for image handling
        self.current_image = None
        self.current_image_with_masks = None
        self.image_scale = 0.2  # Start at 20% for large images
        self.image_id = None
        self.current_point_cloud = None
        self.current_masks_data = None
        self.current_pose_data = None

        # Initialize variable for depth checkbox
        self.use_depth_var = tk.BooleanVar(value=True)  # Default to True
        
        # Setup the tab content
        self.setup_tab()
    
    def setup_tab(self):
        # Frame to hold buttons side by side
        buttons_frame = ttk.Frame(self.frame)
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
        paths_frame = ttk.Frame(self.frame)
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
            self.frame,
            text="Ready to process dataset",
            foreground="blue"
        )
        self.status_label.pack(anchor='nw', padx=10, pady=10)
        
        # CSV Display Area
        self.setup_csv_display()
        
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
    
    def setup_csv_display(self):
        """Setup the CSV display table with scrollbars"""
        # Label for CSV display
        csv_label = ttk.Label(self.frame, text="Dataset CSV Content:")
        csv_label.pack(anchor='nw', padx=10, pady=(20, 5))
        
        # Frame for the table and scrollbars - fixed height, full width
        table_frame = ttk.Frame(self.frame)
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
            self.frame,
            text="Load Image + PCL",
            command=self.load_image_pcl_clicked
        )
        self.load_button.pack(anchor='nw', padx=10, pady=10)
        
        # Setup image and 3D display area
        self.setup_display_area()
    
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
        existing_csvs = [f for f in os.listdir(meta_data_dir) if f.startswith('master_dataset') and f.endswith('.csv')]

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
        dialog = tk.Toplevel()
        dialog.title("Dataset CSV Exists")
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
                
                # Update training summary
                self.update_training_summary()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file:\n{str(e)}")
            self.status_label.config(text="✗ Error loading CSV", foreground="red")
    
    def analyze_csv_data(self):
        """Analyze CSV data to extract class information"""
        try:
            if not self.csv_tree.get_children():
                return None
            
            headers = [col for col in self.csv_tree['columns']]
            
            # Find object_id column
            object_id_col = None
            for i, header in enumerate(headers):
                if 'object_id' in header.lower():
                    object_id_col = i
                    break
            
            if object_id_col is None:
                return {
                    'error': 'No object_id column found in CSV',
                    'total_images': len(self.csv_tree.get_children()),
                    'unique_classes': [],
                    'class_counts': {}
                }
            
            # Collect all object_ids
            object_ids = []
            total_images = 0
            
            for item in self.csv_tree.get_children():
                values = self.csv_tree.item(item, 'values')
                if object_id_col < len(values) and values[object_id_col]:
                    object_ids.append(values[object_id_col])
                total_images += 1
            
            # Count occurrences
            class_counts = Counter(object_ids)
            unique_classes = list(class_counts.keys())
            
            return {
                'total_images': total_images,
                'unique_classes': unique_classes,
                'num_unique_classes': len(unique_classes),
                'class_counts': dict(class_counts),
                'error': None
            }
            
        except Exception as e:
            return {
                'error': f'Error analyzing CSV data: {str(e)}',
                'total_images': 0,
                'unique_classes': [],
                'class_counts': {}
            }
    
    def update_training_summary(self):
        """Update the training summary text box"""
        analysis = self.analyze_csv_data()
        
        if analysis is None:
            summary_text = "No CSV data loaded"
        elif analysis['error']:
            summary_text = f"Error: {analysis['error']}\nTotal rows: {analysis['total_images']}"
        else:
            summary_text = f"Dataset Summary:\n"
            summary_text += f"═══════════════════\n"
            summary_text += f"Total Images: {analysis['total_images']}\n"
            summary_text += f"Unique Classes: {analysis['num_unique_classes']}\n\n"
            
            summary_text += f"Class Distribution:\n"
            summary_text += f"──────────────────\n"
            
            if analysis['unique_classes']:
                # Sort by count (descending)
                sorted_classes = sorted(analysis['class_counts'].items(), key=lambda x: x[1], reverse=True)
                for class_name, count in sorted_classes:
                    percentage = (count / analysis['total_images']) * 100
                    summary_text += f"{class_name}: {count} images ({percentage:.1f}%)\n"
            else:
                summary_text += "No classes found\n"
        
        # Update the text widget
        self.training_summary.delete(1.0, tk.END)
        self.training_summary.insert(tk.END, summary_text)
    
    def setup_display_area(self):
        """Setup the image and 3D point cloud display area"""
        # Frame for image and 3D viewer side by side
        display_frame = ttk.Frame(self.frame)
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
        
        # Container for image canvas
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
        
        # Right side - Container for 3D Point Cloud and Training sections
        right_container = ttk.Frame(display_frame)
        right_container.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # 3D Point Cloud section
        pcl_frame = ttk.LabelFrame(right_container, text="3D Point Cloud")
        pcl_frame.pack(fill='both', expand=True, pady=(0, 5))
        
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
        
        # Training section
        training_frame = ttk.LabelFrame(right_container, text="Training")
        training_frame.pack(fill='both', expand=True, pady=(5, 0))
        
        # Button frame for training
        training_button_frame = ttk.Frame(training_frame)
        training_button_frame.pack(pady=10)

        # Container for button and checkbox
        controls_frame = ttk.Frame(training_button_frame)
        controls_frame.pack()
        
        # Train Mask R-CNN button
        self.train_maskrcnn_button = ttk.Button(
            controls_frame,
            text="Train Mask R-CNN",
            command=self.train_maskrcnn_clicked
        )
        self.train_maskrcnn_button.pack(side='left', padx=(0, 10))  # Packed first, on the left

        # Use Depth checkbox
        self.use_depth_checkbox = ttk.Checkbutton(
            controls_frame,
            text="Use Depth Images",
            variable=self.use_depth_var,
            onvalue=True,
            offvalue=False
        )
        self.use_depth_checkbox.pack(side='left')
        
        # Training summary text area with scrollbar
        summary_frame = ttk.Frame(training_frame)
        summary_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Text widget with scrollbar
        self.training_summary = tk.Text(
            summary_frame,
            wrap=tk.WORD,
            height=12,
            width=40,
            font=("Consolas", 9)
        )
        
        training_scroll = ttk.Scrollbar(summary_frame, orient='vertical', command=self.training_summary.yview)
        self.training_summary.configure(yscrollcommand=training_scroll.set)
        
        # Grid layout for text and scrollbar
        self.training_summary.grid(row=0, column=0, sticky='nsew')
        training_scroll.grid(row=0, column=1, sticky='ns')
        
        # Configure grid weights
        summary_frame.grid_rowconfigure(0, weight=1)
        summary_frame.grid_columnconfigure(0, weight=1)
        
        # Initialize training summary
        self.training_summary.insert(tk.END, "Load CSV data to see training summary")
    
    def train_maskrcnn_clicked(self):
        """Handle Train Mask R-CNN button click"""
        # Placeholder for training functionality
        if not self.current_csv_path:
            messagebox.showwarning("No Data", "Please load CSV data first")
            return
        
        analysis = self.analyze_csv_data()

        if analysis and analysis['error']:
            messagebox.showerror("Error", f"Cannot analyze dataset for training:\n{analysis['error']}")
            return
        
        try:

            # Get the checkbox value
            use_depth = self.use_depth_var.get()

            trainer = TrainNewDataset(
                dataset_csv_path=self.current_csv_path,
                use_depth=True
            )
        
            if analysis and not analysis['error']:
                message = f"Training initiated with:\n"
                message += f"• Dataset: {os.path.basename(self.current_csv_path)}\n"
                message += f"• {analysis['total_images']} images\n"
                message += f"• {analysis['num_unique_classes']} classes\n"
                message += f"• Classes: {', '.join(analysis['unique_classes'][:5])}"  # Show first 5 classes
                if len(analysis['unique_classes']) > 5:
                    message += f"... and {len(analysis['unique_classes']) - 5} more"
            
                messagebox.showinfo("Training Started", message)
            
            # Optional: Update status label
            self.status_label.config(
                text="Training Mask R-CNN in progress...", 
                foreground="orange"
            )

            # Optional: Disable the train button to prevent multiple instances
            self.train_maskrcnn_button.config(state='disable')            
            self.use_depth_checkbox.config(state='normal')

            trainer.train()

        except Exception as e:
            messagebox.showerror("Training Error", f"Failed to initiate training:\n{str(e)}")
            self.status_label.config(
                text="Training failed to start", 
                foreground="red"
            )
            # Re-enable button on error
            self.train_maskrcnn_button.config(state='normal')

        self.train_maskrcnn_button.config(state='normal')


    
    def extract_mask_data_from_row(self, row_values):
        """Extract mask data from CSV row"""
        try:
            headers = [col for col in self.csv_tree['columns']]
            
            # Look for polygon or bbox data in headers
            for i, header in enumerate(headers):
                header_lower = header.lower()
                
                # Check for polygon coordinates
                if any(keyword in header_lower for keyword in ['polygon', 'poly', 'mask']):
                    if i < len(row_values) and row_values[i]:
                        try:
                            # Try to parse polygon coordinates
                            poly_str = str(row_values[i])
                            if '[' in poly_str:
                                import ast
                                poly_coords = ast.literal_eval(poly_str)
                                if isinstance(poly_coords, list) and len(poly_coords) > 2:
                                    return [{
                                        'object_id': 'GT_Object',
                                        'polygon': poly_coords
                                    }]
                        except:
                            pass
                
                # Check for bounding box coordinates
                elif 'bbox' in header_lower and i < len(row_values) and row_values[i]:
                    try:
                        bbox_str = str(row_values[i]).strip('[]')
                        bbox_coords = [float(x.strip()) for x in bbox_str.split(',')]
                        if len(bbox_coords) == 4:
                            x1, y1, x2, y2 = bbox_coords
                            return [{
                                'object_id': 'GT_Object',
                                'polygon': [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                            }]
                    except:
                        pass
            
            # Generate sample mask if no data found
            if self.current_image is not None:
                height, width = self.current_image.shape[:2]
                return [{
                    'object_id': 'Sample_Object',
                    'polygon': [[width*0.3, height*0.3], [width*0.7, height*0.3], [width*0.7, height*0.7], [width*0.3, height*0.7]]
                }]
            
            return []
            
        except Exception as e:
            print(f"Error extracting mask data: {e}")
            return []
    
    def extract_pose_data_from_row(self, row_values):
        """Extract pose data from CSV row"""
        try:
            headers = [col for col in self.csv_tree['columns']]
            header_map = {header.lower(): i for i, header in enumerate(headers)}
            
            print(f"Available headers: {list(header_map.keys())}")
            print(f"Row has {len(row_values)} values")
            
            # Look for rotation matrix elements (r11-r33) and translation (tx,ty,tz)
            rotation_elements = {}
            translation_elements = {}
            
            # Find rotation matrix elements
            for i in range(1, 4):
                for j in range(1, 4):
                    key = f'r{i}{j}'
                    if key in header_map:
                        col_idx = header_map[key]
                        if col_idx < len(row_values) and row_values[col_idx]:
                            try:
                                rotation_elements[key] = float(row_values[col_idx])
                                print(f"Found {key} = {rotation_elements[key]}")
                            except ValueError:
                                print(f"Could not convert {key} value: {row_values[col_idx]}")
                                pass
            
            # Find translation elements
            for axis, key in [('x', 'tx'), ('y', 'ty'), ('z', 'tz')]:
                if key in header_map:
                    col_idx = header_map[key]
                    if col_idx < len(row_values) and row_values[col_idx]:
                        try:
                            translation_elements[key] = float(row_values[col_idx])
                            print(f"Found {key} = {translation_elements[key]}")
                        except ValueError:
                            print(f"Could not convert {key} value: {row_values[col_idx]}")
                            pass
            
            print(f"Found {len(rotation_elements)} rotation elements, {len(translation_elements)} translation elements")
            
            # If we found pose data, create transformation matrix
            if len(rotation_elements) == 9 and len(translation_elements) == 3:
                rotation_matrix = np.array([
                    [rotation_elements['r11'], rotation_elements['r12'], rotation_elements['r13']],
                    [rotation_elements['r21'], rotation_elements['r22'], rotation_elements['r23']],
                    [rotation_elements['r31'], rotation_elements['r32'], rotation_elements['r33']]
                ])
                
                translation_vector = np.array([
                    translation_elements['tx'],
                    translation_elements['ty'],
                    translation_elements['tz']
                ])
                
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :3] = rotation_matrix
                transformation_matrix[:3, 3] = translation_vector
                
                print("Successfully created transformation matrix from CSV data")
                return [{
                    'object_id': 'GT_Object',
                    'transformation_matrix': transformation_matrix
                }]
            
            # Generate sample pose if no data found (visible position)
            print("No complete pose data found in CSV, creating sample pose")
            
            # Create a sample pose that's visible relative to point cloud
            if self.current_point_cloud is not None:
                points = np.asarray(self.current_point_cloud.points)
                if len(points) > 0:
                    center = np.mean(points, axis=0)
                    print(f"Point cloud center: {center}")
                    
                    transformation_matrix = np.eye(4)
                    # Position sample pose near the center of the point cloud
                    transformation_matrix[:3, 3] = center + [0.1, 0.1, 0.1]
                else:
                    transformation_matrix = np.eye(4)
                    transformation_matrix[:3, 3] = [0.2, 0.2, 0.2]
            else:
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, 3] = [0.2, 0.2, 0.2]
            
            return [{
                'object_id': 'Sample_Object',
                'transformation_matrix': transformation_matrix
            }]
            
        except Exception as e:
            print(f"Error extracting pose data: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def draw_masks_on_image(self, image, mask_data):
        """Draw polygon masks on the image"""
        try:
            # Convert BGR to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            # Try to load a font for text labels
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            for mask in mask_data:
                obj_id = mask.get('object_id', 'Object')
                polygon = mask.get('polygon', [])
                
                if polygon and len(polygon) > 2:
                    # Convert polygon points to tuple format for PIL
                    poly_points = [(int(pt[0]), int(pt[1])) for pt in polygon]
                    
                    # Draw semi-transparent polygon
                    color = (255, 0, 0, 100)  # Red with transparency
                    outline_color = (255, 0, 0, 255)  # Solid red outline
                    
                    draw.polygon(poly_points, fill=color, outline=outline_color, width=2)
                    
                    # Draw object ID label
                    if poly_points:
                        label_x = min(pt[0] for pt in poly_points)
                        label_y = min(pt[1] for pt in poly_points) - 20
                        label_x = max(0, label_x)
                        label_y = max(0, label_y)
                        
                        # Draw text background
                        text_bbox = draw.textbbox((label_x, label_y), str(obj_id), font=font)
                        draw.rectangle(text_bbox, fill=(255, 255, 255, 200), outline=(0, 0, 0, 255))
                        
                        # Draw text
                        draw.text((label_x, label_y), str(obj_id), fill=(0, 0, 0, 255), font=font)
            
            # Convert back to BGR for OpenCV
            result_array = np.array(pil_image)
            result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
            
            return result_bgr
            
        except Exception as e:
            print(f"Error drawing masks: {e}")
            return image
    
    def calculate_appropriate_axis_size(self):
        """Calculate appropriate axis size based on point cloud dimensions"""
        if self.current_point_cloud is None:
            print("No point cloud available, using default axis size")
            return 0.1
        
        try:
            points = np.asarray(self.current_point_cloud.points)
            if len(points) == 0:
                print("Point cloud is empty, using default axis size")
                return 0.1
            
            # Calculate bounding box dimensions
            min_coords = points.min(axis=0)
            max_coords = points.max(axis=0)
            dimensions = max_coords - min_coords
            
            # Use 10% of the largest dimension as axis size (increased from 5%)
            max_dimension = np.max(dimensions)
            axis_size = max_dimension * 0.1
            
            # Ensure minimum and maximum sizes
            axis_size = max(0.05, min(axis_size, 1.0))  # Increased minimum from 0.02 to 0.05
            
            print(f"Point cloud dimensions: {dimensions}")
            print(f"Max dimension: {max_dimension}")
            print(f"Calculated axis size: {axis_size}")
            
            return axis_size
            
        except Exception as e:
            print(f"Error calculating axis size: {e}")
            return 0.1
    
    def load_and_display_image(self, image_path):
        """Load and display RGB image with masks automatically"""
        try:
            # Load image at original size
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                raise ValueError("Could not load image")
            
            # Start at appropriate scale for different image sizes
            height, width = self.current_image.shape[:2]
            if width > 2000:  # Large image
                self.image_scale = 0.3
            elif width > 1000:  # Medium image
                self.image_scale = 0.5
            else:  # Small image
                self.image_scale = 1.0
            
            print(f"Loaded image: {width}x{height}, starting scale: {self.image_scale}")
            
            # Automatically load and show masks
            selected_items = self.csv_tree.selection()
            if selected_items:
                item = selected_items[0]
                row_values = self.csv_tree.item(item, 'values')
                
                # Extract and draw masks
                mask_data = self.extract_mask_data_from_row(row_values)
                if mask_data:
                    self.current_image_with_masks = self.draw_masks_on_image(self.current_image.copy(), mask_data)
                    self.current_masks_data = mask_data
                    print(f"Loaded {len(mask_data)} masks on image")
            
            # Display image (with masks if available)
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
        
        # Use masked image if available, otherwise use original
        display_image = self.current_image_with_masks if self.current_image_with_masks is not None else self.current_image
        
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
    
    def open_3d_viewer(self):
        """Open Open3D visualization window with automatic GT pose display"""
        if self.current_point_cloud is None:
            messagebox.showwarning("No Data", "Please load image and point cloud data first")
            return
        
        try:
            # Get point cloud bounds for reference
            points = np.asarray(self.current_point_cloud.points)
            pc_min = points.min(axis=0)
            pc_max = points.max(axis=0)
            pc_center = (pc_min + pc_max) / 2
            pc_size = np.max(pc_max - pc_min)
            print("GT pc_size: ", pc_size)
            
            print(f"Point cloud bounds: min={pc_min}, max={pc_max}")
            print(f"Point cloud center: {pc_center}")
            print(f"Point cloud size: {pc_size}")
            
            # Start with the point cloud
            geometries = [self.current_point_cloud]
            
            # Make axes much larger - use 20% of point cloud size
            axis_size = pc_size * 0.05
            print(f"Using axis size: {axis_size}")
            
            # Always add a coordinate frame at point cloud center for reference
            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
            origin_frame.translate([0, 0, 0])  # Center at origin
            geometries.append(origin_frame)
            print(f"Added origin coordinate frame at point cloud center: {[0, 0, 0]}")
            
            # Automatically load and add GT poses
            selected_items = self.csv_tree.selection()
            if selected_items:
                item = selected_items[0]
                row_values = self.csv_tree.item(item, 'values')
                
                print("Extracting pose data from selected row...")
                pose_data = self.extract_pose_data_from_row(row_values)
                
                if pose_data:
                    self.current_pose_data = pose_data
                    print(f"Found {len(pose_data)} poses")
                    
                    for i, pose in enumerate(pose_data):
                        transformation_matrix = pose['transformation_matrix']
                        obj_id = pose['object_id']
                        pose_position = transformation_matrix[:3, 3]
                        
                        print(f"Pose {i} transformation matrix:")
                        print(transformation_matrix)
                        print(f"Pose position: {pose_position}")
                        
                        # Always create the original pose too (even if far away)
                        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
                        print("GT transformation matrix: ", transformation_matrix)
                        coord_frame.transform(transformation_matrix)
                        center = np.asarray(coord_frame.get_center())
                        print(f"Frame {i} center: {center}") 
                        geometries.append(coord_frame)
                        
                        # Add red sphere marker at original pose location
                        marker_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size*0.01)
                        marker_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red marker
                        marker_sphere.translate(pose_position)
                        geometries.append(marker_sphere)
                        
                        print(f"Added ORIGINAL pose marker at: {pose_position}")
                        
                else:
                    print("No pose data found")
            else:
                print("No row selected")
            
            print(f"Total geometries to display: {len(geometries)}")
            for i, geom in enumerate(geometries):
                print(f"Geometry {i}: {type(geom)}")
            
            # Use the draw_geometries function
            print("Opening visualization...")
            o3d.visualization.draw_geometries(
                geometries,
                window_name="Point Cloud with GT Poses",
                width=1200,
                height=800,
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
            depth_scale = row_values[45] if len(row_values) > 45 else "1000"
            
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
                use_gpu=True,
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
            root = self.notebook.nametowidget(self.notebook.winfo_toplevel())
            root.after(0, self.dataset_processing_complete, result, output_csv_path)
            
        except Exception as e:
            # Handle unexpected errors
            error_result = {
                'success': False,
                'message': f"Unexpected error: {str(e)}",
                'total_rows': 0
            }
            root = self.notebook.nametowidget(self.notebook.winfo_toplevel())
            root.after(0, self.dataset_processing_complete, error_result, output_csv_path)
    
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