import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image, ImageTk

class AnnotationViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Annotation Viewer")
        self.root.geometry("1200x800")
        
        # Variables to store file paths
        self.annotation_file = None
        self.image_file = None
        self.annotations = None
        self.image = None
        
        # Create the GUI
        self.create_widgets()
    
    def create_widgets(self):
        # Create frames
        self.control_frame = tk.Frame(self.root, padx=10, pady=10)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.display_frame = tk.Frame(self.root)
        self.display_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Control buttons
        self.load_json_btn = tk.Button(self.control_frame, text="Load Annotation JSON", command=self.load_annotation)
        self.load_json_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.load_img_btn = tk.Button(self.control_frame, text="Load Image", command=self.load_image)
        self.load_img_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.display_btn = tk.Button(self.control_frame, text="Display Annotations", command=self.display_annotations)
        self.display_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Status labels
        self.json_status = tk.Label(self.control_frame, text="No annotation file loaded", width=30, anchor='w')
        self.json_status.grid(row=1, column=0, columnspan=2, sticky='w', padx=5)
        
        self.img_status = tk.Label(self.control_frame, text="No image file loaded", width=30, anchor='w')
        self.img_status.grid(row=2, column=0, columnspan=2, sticky='w', padx=5)
        
        # Color selector
        self.color_var = tk.StringVar(value="red")
        self.color_label = tk.Label(self.control_frame, text="Polygon Color:")
        self.color_label.grid(row=0, column=3, padx=5, pady=5)
        
        colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
        self.color_menu = tk.OptionMenu(self.control_frame, self.color_var, *colors)
        self.color_menu.grid(row=0, column=4, padx=5, pady=5)
        
        # Zoom control
        self.zoom_label = tk.Label(self.control_frame, text="Zoom:")
        self.zoom_label.grid(row=0, column=5, padx=5, pady=5)
        
        self.zoom_in_btn = tk.Button(self.control_frame, text="+", command=self.zoom_in, width=2)
        self.zoom_in_btn.grid(row=0, column=6, padx=2, pady=5)
        
        self.zoom_out_btn = tk.Button(self.control_frame, text="-", command=self.zoom_out, width=2)
        self.zoom_out_btn.grid(row=0, column=7, padx=2, pady=5)
        
        self.zoom_reset_btn = tk.Button(self.control_frame, text="Reset View", command=self.reset_view)
        self.zoom_reset_btn.grid(row=0, column=8, padx=5, pady=5)
        
        # Create the initial matplotlib figure
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.plot = self.figure.add_subplot(111)
        
        # Create canvas and toolbar for matplotlib
        self.canvas = FigureCanvasTkAgg(self.figure, self.display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar (includes zoom and pan)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.display_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure the plot
        self.plot.set_title("Image with Annotations")
        self.plot.set_axis_off()
        
        # Connect mouse events for enhanced interaction
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Variables for panning
        self.pan_active = False
        self.pan_start_x = None
        self.pan_start_y = None
        
    def on_scroll(self, event):
        """Handle scroll events for zooming"""
        # Only zoom if we have an image loaded
        if not hasattr(self, 'img_obj') or self.img_obj is None:
            return
            
        # Get current axis limits
        x_min, x_max = self.plot.get_xlim()
        y_min, y_max = self.plot.get_ylim()
        
        # Calculate zoom factor
        base_scale = 1.1
        if event.button == 'up':
            # Zoom in
            scale_factor = 1 / base_scale
        else:
            # Zoom out
            scale_factor = base_scale
            
        # Calculate new limits (zoom around cursor position)
        x_range = (x_max - x_min) * scale_factor
        y_range = (y_max - y_min) * scale_factor
        
        # Only continue if we're not zooming out too far
        if hasattr(self, 'img_width') and hasattr(self, 'img_height'):
            # Don't zoom out further than 20% of original size
            if x_range > self.img_width * 5 or y_range > self.img_height * 5:
                if scale_factor > 1:  # Only limit zoom out, not zoom in
                    return
        
        # Calculate new center point (weighted by cursor position)
        if event.xdata is not None and event.ydata is not None:
            # Shift the center point towards the cursor
            center_x = event.xdata
            center_y = event.ydata
            
            # Apply new limits
            self.plot.set_xlim(center_x - x_range/2, center_x + x_range/2)
            self.plot.set_ylim(center_y - y_range/2, center_y + y_range/2)
            self.canvas.draw_idle()
    
    def on_press(self, event):
        """Handle mouse button press for panning"""
        if event.button == 3:  # Right mouse button
            self.pan_active = True
            self.pan_start_x = event.x
            self.pan_start_y = event.y
    
    def on_release(self, event):
        """Handle mouse button release"""
        if event.button == 3:  # Right mouse button
            self.pan_active = False
    
    def on_motion(self, event):
        """Handle mouse motion for panning"""
        if self.pan_active and hasattr(self, 'img_obj'):
            # Calculate movement in data coordinates
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            
            # Convert from screen to data coordinates
            x_min, x_max = self.plot.get_xlim()
            y_min, y_max = self.plot.get_ylim()
            
            # Figure out the scale factor
            canvas_width = self.canvas.get_tk_widget().winfo_width()
            canvas_height = self.canvas.get_tk_widget().winfo_height()
            
            # Adjust based on the canvas size
            x_scale = (x_max - x_min) / canvas_width
            y_scale = (y_max - y_min) / canvas_height
            
            # Move in the opposite direction of mouse movement
            new_x_min = x_min - dx * x_scale
            new_x_max = x_max - dx * x_scale
            new_y_min = y_min + dy * y_scale  # Inverted y-axis
            new_y_max = y_max + dy * y_scale  # Inverted y-axis
            
            # Apply new limits
            self.plot.set_xlim(new_x_min, new_x_max)
            self.plot.set_ylim(new_y_min, new_y_max)
            self.canvas.draw_idle()
            
            # Update start position for next movement
            self.pan_start_x = event.x
            self.pan_start_y = event.y
    
    def zoom_in(self):
        """Zoom in button handler"""
        if hasattr(self, 'img_obj'):
            x_min, x_max = self.plot.get_xlim()
            y_min, y_max = self.plot.get_ylim()
            
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            
            # Zoom in by 20%
            x_range = (x_max - x_min) * 0.8
            y_range = (y_max - y_min) * 0.8
            
            self.plot.set_xlim(center_x - x_range/2, center_x + x_range/2)
            self.plot.set_ylim(center_y - y_range/2, center_y + y_range/2)
            self.canvas.draw_idle()
    
    def zoom_out(self):
        """Zoom out button handler"""
        if hasattr(self, 'img_obj'):
            x_min, x_max = self.plot.get_xlim()
            y_min, y_max = self.plot.get_ylim()
            
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            
            # Zoom out by 20%
            x_range = (x_max - x_min) * 1.2
            y_range = (y_max - y_min) * 1.2
            
            # Don't zoom out further than 20% of original size
            if hasattr(self, 'img_width') and hasattr(self, 'img_height'):
                if x_range > self.img_width * 5 or y_range > self.img_height * 5:
                    return
            
            self.plot.set_xlim(center_x - x_range/2, center_x + x_range/2)
            self.plot.set_ylim(center_y - y_range/2, center_y + y_range/2)
            self.canvas.draw_idle()
    
    def reset_view(self):
        """Reset view to original"""
        if hasattr(self, 'img_width') and hasattr(self, 'img_height'):
            self.plot.set_xlim(0, self.img_width)
            self.plot.set_ylim(self.img_height, 0)  # Invert y-axis for correct orientation
            self.canvas.draw_idle()
    
    def load_annotation(self):
        """Load and parse the annotation JSON file"""
        file_path = filedialog.askopenfilename(
            title="Select Annotation JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r') as f:
                self.annotations = json.load(f)
            
            self.annotation_file = file_path
            filename = os.path.basename(file_path)
            self.json_status.config(text=f"Loaded: {filename}")
            
            # If the JSON contains an image path, suggest it
            if 'image_path' in self.annotations:
                img_path = self.annotations['image_path']
                json_dir = os.path.dirname(file_path)
                suggested_img_path = os.path.join(json_dir, img_path)
                
                if os.path.exists(suggested_img_path):
                    self.image_file = suggested_img_path
                    self.img_status.config(text=f"Suggested: {os.path.basename(suggested_img_path)}")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load annotation file: {str(e)}")
    
    def load_image(self):
        """Load the image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.image_file = file_path
            filename = os.path.basename(file_path)
            self.img_status.config(text=f"Loaded: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_annotations(self):
        """Display the image with annotation polygons"""
        if not self.annotation_file:
            messagebox.showwarning("Warning", "Please load an annotation file first")
            return
        
        if not self.image_file:
            messagebox.showwarning("Warning", "Please load an image file first")
            return
        
        try:
            # Clear the plot
            self.plot.clear()
            
            # Load and display the image
            img = plt.imread(self.image_file)
            self.img_obj = self.plot.imshow(img)
            
            # Store image dimensions for reference
            self.img_height, self.img_width = img.shape[:2]
            
            # Get all masks from the annotations
            if 'masks' in self.annotations:
                masks = self.annotations['masks']
                color = self.color_var.get()
                
                # Draw each polygon
                for i, mask in enumerate(masks):
                    label = mask.get('label', f"Object {i}")
                    points = np.array(mask['points'])
                    
                    # Create a polygon patch
                    polygon = Polygon(points, 
                                     fill=False, 
                                     edgecolor=color, 
                                     linewidth=2, 
                                     label=label)
                    self.plot.add_patch(polygon)
                    
                    # Add a label at the center of the polygon
                    centroid = np.mean(points, axis=0)
                    self.plot.text(centroid[0], centroid[1], label, 
                                 color='white', fontsize=9, 
                                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
            
            # Set the title and axis with image dimensions
            self.plot.set_title(f"Image with Annotations ({self.img_width}x{self.img_height})")
            self.plot.set_xlim(0, self.img_width)
            self.plot.set_ylim(self.img_height, 0)  # Invert y-axis for correct orientation
            self.plot.set_axis_off()
            
            # Update the status bar with zoom info
            self.root.title(f"Annotation Viewer - {os.path.basename(self.image_file)} ({self.img_width}x{self.img_height})")
            
            # Update the canvas
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display annotations: {str(e)}")
            raise e  # Re-raise for debugging

def main():
    root = tk.Tk()
    app = AnnotationViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()