#!/usr/bin/env python3
# SAM Fine-tuned Inference Script

import os
import cv2
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import argparse
from tqdm import tqdm
from pathlib import Path
import random

def load_finetuned_sam(
    model_path,
    model_type="vit_h",
    original_checkpoint=None,
    device=None
):
    """
    Load a fine-tuned SAM model
    
    Args:
        model_path: Path to fine-tuned model weights
        model_type: SAM model type (vit_b, vit_l, vit_h)
        original_checkpoint: Path to original SAM checkpoint (if None, will attempt to download)
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model on {device}...")
    
    # Set default checkpoint paths based on model type if not provided
    if original_checkpoint is None:
        checkpoint_paths = {
            "vit_h": "models/sam_vit_h_4b8939.pth",
            "vit_l": "models/sam_vit_l_0b3195.pth",
            "vit_b": "models/sam_vit_b_01ec64.pth"
        }
        original_checkpoint = checkpoint_paths[model_type]
    
    # Download the original model if it doesn't exist
    if not os.path.exists(original_checkpoint):
        os.makedirs(os.path.dirname(original_checkpoint), exist_ok=True)
        print(f"Original checkpoint not found at {original_checkpoint}")
        print("Downloading original SAM model...")
        
        model_urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        }
        
        import requests
        response = requests.get(model_urls[model_type], stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        
        with open(original_checkpoint, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        
        progress_bar.close()
        print(f"Downloaded original model to {original_checkpoint}")
    
    # First load the base model
    sam = sam_model_registry[model_type](checkpoint=original_checkpoint)
    
    # Load fine-tuned weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle PyTorch Lightning checkpoints
        if 'state_dict' in state_dict:
            # Remove "sam." prefix from state_dict keys
            state_dict = {k.replace('sam.', ''): v for k, v in state_dict['state_dict'].items() 
                          if k.startswith('sam.')}
        
        # Load weights
        sam.load_state_dict(state_dict, strict=False)
        print(f"Loaded fine-tuned weights from {model_path}")
    else:
        print(f"Warning: Fine-tuned model not found at {model_path}")
        print("Using original SAM model instead")
    
    sam.to(device)
    return sam

def predict_segmentation(
    model,
    image_path,
    classes_path,
    output_dir=None,
    point_coords=None,
    point_labels=None,
    box=None,
    multimask_output=False,
    show_points=True,
    show_boxes=True,
    show_class_labels=True,
    conf_threshold=0.7
):
    """
    Predict segmentation for an image using fine-tuned SAM
    
    Args:
        model: Loaded SAM model
        image_path: Path to input image
        classes_path: Path to classes.json file
        output_dir: Directory to save results (if None, will show interactively)
        point_coords: Point coordinates for prompt (if None, will use grid)
        point_labels: Point labels for prompt (if None, will use all foreground)
        box: Box prompt coordinates [x1, y1, x2, y2]
        multimask_output: Whether to output multiple masks per prompt
        show_points: Whether to show prompt points in visualization
        show_boxes: Whether to show prompt boxes in visualization
        show_class_labels: Whether to show class labels in visualization
        conf_threshold: Confidence threshold for mask prediction
    
    Returns:
        Combined mask with class IDs and list of class names
    """
    # Load the classes
    with open(classes_path, 'r') as f:
        classes = json.load(f)
    
    # Create predictor
    predictor = SamPredictor(model)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set image in predictor
    predictor.set_image(image)
    
    # If no points provided, use automatic grid
    if point_coords is None and box is None:
        # Create a grid of points
        h, w = image.shape[:2]
        num_points = 8  # points per side
        
        x = np.linspace(0, w-1, num_points)
        y = np.linspace(0, h-1, num_points)
        
        grid_x, grid_y = np.meshgrid(x, y)
        point_coords = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
        
        # Use all grid points as foreground prompts
        point_labels = np.ones(len(point_coords))
    
    # Get masks
    if box is not None:
        masks, scores, logits = predictor.predict(
            box=box,
            multimask_output=multimask_output
        )
    else:
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output
        )
    
    # Visualize results
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    
    # Generate different colors for each class
    num_classes = len(classes) + 1  # +1 for background
    color_map = {}
    
    # Set a seed for reproducible colors
    random.seed(42)
    for i in range(num_classes):
        if i == 0:  # Background
            color_map[i] = [0.1, 0.1, 0.1, 0.0]  # Transparent
        else:
            # Generate vibrant, distinct colors
            h = i / num_classes
            s = 0.8
            v = 0.9
            
            # Convert HSV to RGB
            h_i = int(h * 6)
            f = h * 6 - h_i
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)
            
            if h_i == 0:
                r, g, b = v, t, p
            elif h_i == 1:
                r, g, b = q, v, p
            elif h_i == 2:
                r, g, b = p, v, t
            elif h_i == 3:
                r, g, b = p, q, v
            elif h_i == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q
            
            color_map[i] = [r, g, b, 0.5]  # 50% transparency
    
    # Combine all masks
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
    
    for i, (mask, score, logit) in enumerate(zip(masks, scores, logits)):
        # Only show masks with good scores
        if score < conf_threshold:
            continue
        
        # Get class prediction (highest activation)
        class_id = logit.argmax().item()
        
        # Update combined mask for this region
        combined_mask[mask] = class_id
    
    # Visualize the combined mask
    class_labels_shown = set()
    
    for class_id in range(1, num_classes):  # Skip background
        mask = combined_mask == class_id
        if not np.any(mask):
            continue
        
        color = color_map[class_id]
        mask_image = np.ones((mask.shape[0], mask.shape[1], 4))
        mask_image[:, :, :3] = color[:3]
        mask_image[:, :, 3] = mask * color[3]  # Apply transparency
        
        plt.imshow(mask_image)
        
        # Get the class name
        if class_id - 1 < len(classes):
            class_name = classes[class_id - 1]
        else:
            class_name = f"Class {class_id}"
        
        # Add a label with class name
        if show_class_labels and class_id not in class_labels_shown:
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                center_y = np.median(y_indices)
                center_x = np.median(x_indices)
                
                plt.annotate(
                    class_name,
                    xy=(center_x, center_y),
                    color='white',
                    weight='bold',
                    fontsize=12,
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle="round,pad=0.3", fc=color[:3], alpha=0.7)
                )
                
                class_labels_shown.add(class_id)
    
    # Show prompt points if requested
    if show_points and point_coords is not None:
        if point_labels is not None:
            # Separate foreground/background points
            fg_points = point_coords[point_labels == 1]
            bg_points = point_coords[point_labels == 0]
            
            if len(fg_points) > 0:
                plt.scatter(fg_points[:, 0], fg_points[:, 1], color='green', marker='*', 
                           s=200, edgecolor='white', linewidth=1.25)
            
            if len(bg_points) > 0:
                plt.scatter(bg_points[:, 0], bg_points[:, 1], color='red', marker='*', 
                           s=200, edgecolor='white', linewidth=1.25)
        else:
            plt.scatter(point_coords[:, 0], point_coords[:, 1], color='green', marker='*', 
                       s=200, edgecolor='white', linewidth=1.25)
    
    # Show box if requested
    if show_boxes and box is not None:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                             edgecolor='green', facecolor='none')
        plt.gca().add_patch(rect)
    
    plt.axis('off')
    
    # Get image filename for title
    img_name = os.path.basename(image_path)
    plt.title(f"Segmentation: {img_name}", fontsize=16)
    
    # Save or display the result
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"segmented_{base_name}")
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"Saved segmentation to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return combined_mask, classes

def batch_predict(
    model,
    image_dir,
    classes_path,
    output_dir,
    conf_threshold=0.7,
    image_types=('.jpg', '.jpeg', '.png')
):
    """
    Predict segmentations for all images in a directory
    
    Args:
        model: Loaded SAM model
        image_dir: Directory containing input images
        classes_path: Path to classes.json file
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for mask prediction
        image_types: Tuple of valid image extensions
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images in directory
    image_files = []
    for ext in image_types:
        image_files.extend(list(Path(image_dir).glob(f"*{ext}")))
    
    if not image_files:
        print(f"No images found in {image_dir} with extensions {image_types}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            predict_segmentation(
                model=model,
                image_path=str(image_path),
                classes_path=classes_path,
                output_dir=output_dir,
                conf_threshold=conf_threshold
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def predict_with_click(
    model,
    image_path,
    classes_path,
    output_dir=None
):
    """
    Interactive prediction with click prompts
    
    Args:
        model: Loaded SAM model
        image_path: Path to input image
        classes_path: Path to classes.json file
        output_dir: Directory to save results
    """
    # Load the classes
    with open(classes_path, 'r') as f:
        classes = json.load(f)
    
    # Create predictor
    predictor = SamPredictor(model)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set image in predictor
    predictor.set_image(image)
    
    # Create interactive figure
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.axis('off')
    plt.title("Click on objects to segment (close window when done)")
    
    # List to store points and labels
    points = []
    labels = []
    
    # Click handler
    def onclick(event):
        if event.button == 1:  # Left click (foreground)
            points.append([event.xdata, event.ydata])
            labels.append(1)
            plt.plot(event.xdata, event.ydata, 'go', markersize=10)
        elif event.button == 3:  # Right click (background)
            points.append([event.xdata, event.ydata])
            labels.append(0)
            plt.plot(event.xdata, event.ydata, 'ro', markersize=10)
        
        if len(points) > 0:
            # Convert points and labels to numpy arrays
            input_points = np.array(points)
            input_labels = np.array(labels)
            
            # Get masks
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            
            # Show the best mask
            best_idx = scores.argmax()
            best_mask = masks[best_idx]
            
            # Get class prediction
            class_id = logits[best_idx].argmax().item()
            
            if class_id - 1 < len(classes):
                class_name = classes[class_id - 1]
            else:
                class_name = f"Class {class_id}"
            
            # Update plot
            plt.clf()
            plt.imshow(image)
            
            # Show mask overlay
            color = np.array([30/255, 144/255, 255/255, 0.6])  # Light blue, semi-transparent
            h, w = best_mask.shape
            mask_image = best_mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            plt.imshow(mask_image)
            
            # Show points
            for p, l in zip(points, labels):
                if l == 1:
                    plt.plot(p[0], p[1], 'go', markersize=10)
                else:
                    plt.plot(p[0], p[1], 'ro', markersize=10)
            
            plt.title(f"Predicted: {class_name} (Score: {scores[best_idx]:.3f})")
            plt.axis('off')
            plt.draw()
    
    # Connect click handler
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    
    # Show plot (this will block until window is closed)
    plt.show()
    
    # If no points were selected, return
    if not points:
        print("No points selected.")
        return None, classes
    
    # After window is closed, do final prediction with all points
    input_points = np.array(points)
    input_labels = np.array(labels)
    
    # Predict final segmentation
    return predict_segmentation(
        model=model,
        image_path=image_path,
        classes_path=classes_path,
        output_dir=output_dir,
        point_coords=input_points,
        point_labels=input_labels,
        show_points=True,
        show_class_labels=True
    )

def export_masks(
    masks,
    classes,
    output_dir,
    image_name,
    export_format='png'
):
    """
    Export segmentation masks for further processing
    
    Args:
        masks: Combined mask with class IDs
        classes: List of class names
        output_dir: Directory to save exported masks
        image_name: Base name of the image
        export_format: Format to export masks ('png', 'json', or 'both')
    """
    # Create output directory
    masks_dir = os.path.join(output_dir, 'exported_masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(image_name))[0]
    
    # Export as PNG (one file per class)
    if export_format in ['png', 'both']:
        for i, class_name in enumerate(classes):
            class_id = i + 1
            
            # Create binary mask for this class
            binary_mask = (masks == class_id).astype(np.uint8) * 255
            
            if np.any(binary_mask):
                # Save mask
                mask_path = os.path.join(masks_dir, f"{base_name}_{class_name}.png")
                cv2.imwrite(mask_path, binary_mask)
    
    # Export as JSON with polygon format
    if export_format in ['json', 'both']:
        import json
        from skimage import measure
        
        polygons = {}
        
        for i, class_name in enumerate(classes):
            class_id = i + 1
            
            # Create binary mask for this class
            binary_mask = (masks == class_id).astype(np.uint8)
            
            if np.any(binary_mask):
                # Find contours
                contours = measure.find_contours(binary_mask, 0.5)
                
                # Convert to polygons
                class_polygons = []
                for contour in contours:
                    # Simplify contour to reduce points
                    if len(contour) > 100:
                        from scipy.interpolate import splprep, splev
                        x, y = contour.T
                        tck, u = splprep([x, y], s=0, per=1)
                        u_new = np.linspace(u.min(), u.max(), 100)
                        x_new, y_new = splev(u_new, tck)
                        contour = np.column_stack([x_new, y_new])
                    
                    # Convert to list of points
                    polygon = []
                    for point in contour:
                        polygon.append([float(point[1]), float(point[0])])
                    
                    class_polygons.append(polygon)
                
                if class_polygons:
                    polygons[class_name] = class_polygons
        
        if polygons:
            # Save as JSON
            json_path = os.path.join(masks_dir, f"{base_name}_polygons.json")
            with open(json_path, 'w') as f:
                json.dump(polygons, f)
    
    print(f"Exported masks to {masks_dir}")

def main():
    parser = argparse.ArgumentParser(description="SAM Fine-tuned Inference")
    
    # Required arguments
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--classes", type=str, required=True, help="Path to classes.json file")
    
    # Optional arguments
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--model-type", type=str, default="vit_h", 
                      choices=["vit_b", "vit_l", "vit_h"], help="SAM model type")
    parser.add_argument("--original-checkpoint", type=str, help="Path to original SAM checkpoint")
    parser.add_argument("--confidence", type=float, default=0.7, 
                      help="Confidence threshold for mask prediction")
    
    # Batch processing
    parser.add_argument("--image-dir", type=str, help="Directory containing input images for batch processing")
    
    # Interactive mode
    parser.add_argument("--interactive", action="store_true", 
                      help="Enable interactive mode with click prompts")
    
    # Export options
    parser.add_argument("--export", type=str, choices=["png", "json", "both"], 
                      help="Export masks for further processing")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.image and not args.image_dir:
        parser.error("Either --image, --image-dir, or --interactive must be specified")
    
    # Load model
    model = load_finetuned_sam(
        model_path=args.model,
        model_type=args.model_type,
        original_checkpoint=args.original_checkpoint
    )
    
    # Process based on mode
    if args.interactive:
        # Interactive mode with click prompts
        if args.image:
            masks, classes = predict_with_click(
                model=model,
                image_path=args.image,
                classes_path=args.classes,
                output_dir=args.output
            )
            
            # Export masks if requested
            if args.export and masks is not None:
                export_masks(
                    masks=masks,
                    classes=classes,
                    output_dir=args.output,
                    image_name=args.image,
                    export_format=args.export
                )
        else:
            parser.error("--image must be specified with --interactive")
    
    elif args.image_dir:
        # Batch processing
        batch_predict(
            model=model,
            image_dir=args.image_dir,
            classes_path=args.classes,
            output_dir=args.output,
            conf_threshold=args.confidence
        )
    
    else:
        # Single image processing
        masks, classes = predict_segmentation(
            model=model,
            image_path=args.image,
            classes_path=args.classes,
            output_dir=args.output,
            conf_threshold=args.confidence
        )
        
        # Export masks if requested
        if args.export:
            export_masks(
                masks=masks,
                classes=classes,
                output_dir=args.output,
                image_name=args.image,
                export_format=args.export
            )

if __name__ == "__main__":
    main()