import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
        
        # Create the initial matplotlib figure
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure the plot
        self.plot.set_title("Image with Annotations")
        self.plot.set_axis_off()
        
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
            self.plot.imshow(img)
            
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
            img_height, img_width = img.shape[:2]
            self.plot.set_title(f"Image with Annotations ({img_width}x{img_height})")
            self.plot.set_xlim(0, img_width)
            self.plot.set_ylim(img_height, 0)  # Invert y-axis for correct orientation
            self.plot.set_axis_off()
            
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