import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
from PIL import Image
import torchvision.transforms as T
import glob
import time
import cv2
import json

from model import get_model_instance_segmentation


def load_model(model_path, num_classes):
    """
    Load a saved model from disk
    
    Args:
        model_path (str): Path to the saved model file
        num_classes (int): Number of classes in the model
        
    Returns:
        model: Loaded PyTorch model
    """
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


def load_image(image_path):
    """
    Load an image and convert to tensor
    
    Args:
        image_path: Path to image file
        
    Returns:
        image_tensor: Normalized image tensor [C, H, W]
        original_image: PIL Image for visualization
    """
    original_image = Image.open(image_path).convert("RGB")
    
    # Convert to tensor and normalize
    transform = T.Compose([
        T.ToTensor(),
    ])
    
    image_tensor = transform(original_image)
    return image_tensor, original_image


def get_prediction(model, image_tensor, device, threshold=0.3):
    """
    Get model prediction on an image
    
    Args:
        model: PyTorch model
        image_tensor: Input image tensor [C, H, W]
        device: Device to run on
        threshold: Confidence threshold
        
    Returns:
        output: Filtered model output
    """
    # Move to device and add batch dimension
    img = image_tensor.to(device)[None]
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(img)[0]
    
    # Filter by threshold
    keep = output['scores'] > threshold
    filtered_output = {
        'boxes': output['boxes'][keep],
        'labels': output['labels'][keep],
        'scores': output['scores'][keep],
        'masks': output['masks'][keep]
    }
    
    return filtered_output


def visualize_prediction(image, output, categories, output_path=None, show_masks=True, show_boxes=True):
    """
    Visualize model prediction on an image
    
    Args:
        image: PIL Image
        output: Model output dictionary
        categories: List of category names
        output_path: Path to save the visualization
        show_masks: Whether to show segmentation masks
        show_boxes: Whether to show bounding boxes
    """
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(16, 10))
    ax.imshow(image_np)
    
    # Get prediction components
    boxes = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    masks = output['masks'].cpu().numpy() if 'masks' in output else None
    
    # Create colors for each class
    colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))
    
    # Draw each detection
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        # Get color for class
        color = colors[label % len(colors)]
        
        # Class name
        category_name = categories[label] if label < len(categories) else f"Class {label}"
        label_text = f"{category_name}: {score:.2f}"
        
        # Draw bounding box
        if show_boxes:
            x1, y1, x2, y2 = box.astype(int)
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        
        # Draw mask
        if show_masks and masks is not None:
            mask = masks[i, 0]
            mask_binary = mask > 0.5
            
            # Create colored mask overlay
            colored_mask = np.zeros_like(image_np)
            color_tuple = tuple(int(c * 255) for c in color[:3])  # Convert to RGB tuple
            
            for c in range(3):
                colored_mask[:, :, c] = np.where(mask_binary, color_tuple[c], 0)
            
            # Add the mask with alpha blending
            ax.imshow(np.where(mask_binary[:, :, None], colored_mask, 0), alpha=0.5)
        
        # Add label text
        if show_boxes:
            x1, y1, _, _ = box.astype(int)
            text = ax.text(x1, y1-10, label_text, fontsize=12, color='white', 
                         bbox=dict(facecolor=color, alpha=0.7, pad=2))
    
    plt.axis('off')
    
    # Save or display figure
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def save_rgb_annotated_image(original_image, output, categories, output_path):
    """
    Save annotations directly on the RGB image
    
    Args:
        original_image: PIL Image
        output: Model output dictionary
        categories: List of category names
        output_path: Path to save the annotated RGB image
    """
    # Convert PIL Image to OpenCV format (RGB to BGR)
    image_cv = np.array(original_image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    # Get prediction components
    boxes = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    masks = output['masks'].cpu().numpy() if 'masks' in output else None
    
    # Create colors for each class
    colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))
    
    # Apply masks to the original image
    if masks is not None:
        # Create a copy for the mask overlay
        mask_overlay = image_cv.copy()
        
        for i, (mask, label) in enumerate(zip(masks, labels)):
            # Get color for class
            color = colors[label % len(colors)]
            color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))  # RGB to BGR
            
            # Create binary mask
            mask_binary = (mask[0] > 0.5).astype(np.uint8)
            
            # Apply colored mask
            mask_overlay[mask_binary == 1] = cv2.addWeighted(
                mask_overlay[mask_binary == 1], 
                0.5,  # Alpha for original image
                np.full_like(mask_overlay[mask_binary == 1], color_bgr),
                0.5,  # Alpha for color
                0
            )
        
        # Blend mask overlay with original
        image_cv = mask_overlay
    
    # Draw bounding boxes and labels
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        # Get color for class
        color = colors[label % len(colors)]
        color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))  # RGB to BGR
        
        # Draw bounding box
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Class name and score
        category_name = categories[label] if label < len(categories) else f"Class {label}"
        label_text = f"{category_name}: {score:.2f}"
        
        # Add label text with background
        # Get text size
        text_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # Create a rectangle for text background
        cv2.rectangle(image_cv, (x1, y1-text_size[1]-10), (x1+text_size[0], y1), color_bgr, -1)
        # Add text
        cv2.putText(image_cv, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save the annotated image
    cv2.imwrite(output_path, image_cv)
    print(f"Saved RGB annotated image to {output_path}")


def mask_to_points(mask):
    """
    Convert binary mask to list of contour points
    
    Args:
        mask: Binary mask as numpy array (2D)
        
    Returns:
        points: List of [x, y] contour points
    """
    # Ensure mask is binary
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour (if multiple are found)
    if not contours:
        return []
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Convert to list of points
    points = [[int(point[0][0]), int(point[0][1])] for point in largest_contour]
    
    return points


def save_annotations(image_path, height, width, output, categories, output_path):
    """
    Save predictions as annotations in JSON format
    
    Args:
        image_path: Path to the input image
        height: Image height
        width: Image width
        output: Model output dictionary
        categories: List of category names
        output_path: Path to save the annotation JSON
    """
    # Get prediction components
    boxes = output['boxes'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    masks = output['masks'].cpu().numpy() if 'masks' in output else None
    
    # Prepare annotation data
    annotation_data = {
        "image_path": image_path,
        "height": height,
        "width": width,
        "masks": []
    }
    
    # Create mask directory if needed
    mask_dir = os.path.join(os.path.dirname(output_path), "masks")
    os.makedirs(mask_dir, exist_ok=True)
    
    # Process each mask
    if masks is not None:
        base_name = os.path.basename(image_path).split('.')[0]
        
        for i, (mask, box, label) in enumerate(zip(masks, boxes, labels)):
            # Get category name
            category_name = categories[label] if label < len(categories) else f"Class_{label}"
            
            # Convert mask to points (polygon)
            points = mask_to_points(mask[0])
            
            if not points:
                continue
            
            # Save mask image if needed
            mask_path = f"masks/{base_name}_{i}.png"
            full_mask_path = os.path.join(os.path.dirname(output_path), mask_path)
            mask_img = (mask[0] > 0.5).astype(np.uint8) * 255
            cv2.imwrite(full_mask_path, mask_img)
            
            # Calculate bounding box center
            x1, y1, x2, y2 = box.astype(int)
            bbox_center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            
            # Calculate geometric center (centroid) of polygon
            if points:
                points_array = np.array(points)
                centroid_x = np.mean(points_array[:, 0])
                centroid_y = np.mean(points_array[:, 1])
                geometric_center = [int(centroid_x), int(centroid_y)]
            else:
                geometric_center = bbox_center  # Fallback if no polygon points
            
            # Add to annotation data
            mask_data = {
                "label": category_name,
                "points": points,
                "mask_path": mask_path,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "bbox_center": bbox_center,
                "geometric_center": geometric_center
            }
            annotation_data["masks"].append(mask_data)
    
    # Save annotation data as JSON
    with open(output_path, 'w') as f:
        json.dump(annotation_data, f, indent=2)
    
    print(f"Saved annotation to {output_path}")


def process_image(image_path, model, categories, output_dir, device, threshold=0.5, show=False, save_json=True):
    """
    Process a single image with the model
    
    Args:
        image_path: Path to image file
        model: PyTorch model
        categories: List of category names
        output_dir: Directory to save results
        device: Device to run on
        threshold: Confidence threshold
        show: Whether to display results interactively
        save_json: Whether to save annotation in JSON format
    """
    print(f"Processing image: {os.path.basename(image_path)}")
    
    # Load image
    img_tensor, original_image = load_image(image_path)
    
    # Get prediction
    output = get_prediction(model, img_tensor, device, threshold)
    
    # Prepare output paths
    if output_dir:
        base_name = os.path.basename(image_path)
        visualization_path = os.path.join(output_dir, f"pred_{base_name}")
        rgb_annotated_path = os.path.join(output_dir, f"rgb_annotated_{base_name}")
        json_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_annotation.json")
    else:
        visualization_path = None
        rgb_annotated_path = None
        json_path = None
    
    # Visualize prediction
    visualize_prediction(original_image, output, categories, visualization_path)
    
    # Save annotated RGB image
    if rgb_annotated_path:
        save_rgb_annotated_image(original_image, output, categories, rgb_annotated_path)
    
    # Save annotation as JSON
    if save_json and json_path:
        width, height = original_image.size
        save_annotations(image_path, height, width, output, categories, json_path)
    
    # Display if requested
    if show and visualization_path:
        img = Image.open(visualization_path)
        img.show()
    
    # Print detection summary
    num_detections = len(output['boxes'])
    print(f"Found {num_detections} objects above threshold {threshold}:")
    
    for i in range(num_detections):
        label = output['labels'][i].item()
        score = output['scores'][i].item()
        category_name = categories[label] if label < len(categories) else f"Class {label}"
        print(f"  {category_name}: {score:.3f}")
    
    print("")
    
    return num_detections


def process_video(video_path, model, categories, output_dir, device, threshold=0.5, fps=None, 
                 show_progress=True, skip_frames=0, save_json=True):
    """
    Process video with the model
    
    Args:
        video_path: Path to video file
        model: PyTorch model
        categories: List of category names
        output_dir: Directory to save results
        device: Device to run on
        threshold: Confidence threshold
        fps: Output frames per second (default: same as input)
        show_progress: Whether to show progress bar
        skip_frames: Number of frames to skip between processing
        save_json: Whether to save annotation in JSON format
    
    Returns:
        output_path: Path to output video
    """
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    output_fps = fps if fps else input_fps
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Frames: {total_frames}")
    print(f"  FPS: {input_fps}")
    
    # Create output videos
    output_path = os.path.join(output_dir, f"pred_{os.path.basename(video_path)}")
    output_rgb_path = os.path.join(output_dir, f"rgb_annotated_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for better quality
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    out_rgb = cv2.VideoWriter(output_rgb_path, fourcc, output_fps, (width, height))
    
    # Create directory for frame annotations if saving JSON
    frame_dir = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_frames")
    if save_json:
        os.makedirs(frame_dir, exist_ok=True)
    
    # Process frames
    frame_count = 0
    processed_count = 0
    
    if show_progress:
        from tqdm import tqdm
        pbar = tqdm(total=total_frames)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for faster processing
        if frame_count % (skip_frames + 1) != 0:
            if show_progress:
                pbar.update(1)
            continue
        
        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(frame_rgb)
        
        # Get prediction
        img_tensor = T.ToTensor()(image)
        output = get_prediction(model, img_tensor, device, threshold)
        
        # Save annotation as JSON
        if save_json:
            frame_name = f"frame_{frame_count:06d}"
            json_path = os.path.join(frame_dir, f"{frame_name}_annotation.json")
            frame_path = f"{os.path.basename(frame_dir)}/{frame_name}.jpg"
            
            # Save frame image
            frame_img_path = os.path.join(output_dir, frame_path)
            os.makedirs(os.path.dirname(frame_img_path), exist_ok=True)
            cv2.imwrite(frame_img_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            # Save annotation
            save_annotations(frame_path, height, width, output, categories, json_path)
        
        # Create figure for visualization with masks
        fig, ax = plt.subplots(1, figsize=(width/100, height/100), dpi=100)
        ax.imshow(image)
        
        # Get prediction components
        boxes = output['boxes'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        masks = output['masks'].cpu().numpy() if 'masks' in output else None
        
        # Create colors for each class
        colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))
        
        # Draw each detection
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            # Get color for class
            color = colors[label % len(colors)]
            
            # Class name
            category_name = categories[label] if label < len(categories) else f"Class {label}"
            label_text = f"{category_name}: {score:.2f}"
            
            # Draw bounding box
            x1, y1, x2, y2 = box.astype(int)
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Draw mask
            if masks is not None:
                mask = masks[i, 0]
                mask_binary = mask > 0.5
                
                # Create colored mask overlay
                colored_mask = np.zeros_like(frame_rgb)
                color_tuple = tuple(int(c * 255) for c in color[:3])  # Convert to RGB tuple
                
                for c in range(3):
                    colored_mask[:, :, c] = np.where(mask_binary, color_tuple[c], 0)
                
                # Add the mask with alpha blending
                ax.imshow(np.where(mask_binary[:, :, None], colored_mask, 0), alpha=0.5)
            
            # Add label text
            text = ax.text(x1, y1-10, label_text, fontsize=8, color='white', 
                         bbox=dict(facecolor=color, alpha=0.7, pad=2))
        
        # Hide axes and set tight layout
        ax.axis('off')
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Convert matplotlib figure to image
        fig.canvas.draw()
        frame_with_detections = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame_with_detections = frame_with_detections.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Convert back to BGR for OpenCV
        frame_with_detections = cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR)
        
        # Resize to original dimensions
        frame_with_detections = cv2.resize(frame_with_detections, (width, height))
        
        # Write to masked output video
        out.write(frame_with_detections)
        
        # Create annotated RGB frame and save to video
        frame_copy = frame.copy()  # Work with original BGR frame
        
        # Apply masks to the original image
        if masks is not None:
            for i, (mask, label) in enumerate(zip(masks, labels)):
                # Get color for class
                color = colors[label % len(colors)]
                color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))  # RGB to BGR
                
                # Create binary mask
                mask_binary = (mask[0] > 0.5).astype(np.uint8)
                
                # Apply colored mask with alpha blending
                for c in range(3):  # Apply to each color channel
                    frame_copy[:, :, c] = np.where(
                        mask_binary == 1,
                        frame_copy[:, :, c] * 0.5 + color_bgr[c] * 0.5,
                        frame_copy[:, :, c]
                    )
        
        # Draw bounding boxes and labels
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            # Get color for class
            color = colors[label % len(colors)]
            color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))  # RGB to BGR
            
            # Draw bounding box
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color_bgr, 2)
            
            # Class name and score
            category_name = categories[label] if label < len(categories) else f"Class {label}"
            label_text = f"{category_name}: {score:.2f}"
            
            # Add label text with background
            text_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame_copy, (x1, y1-text_size[1]-10), (x1+text_size[0], y1), color_bgr, -1)
            cv2.putText(frame_copy, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Write to RGB annotated video
        out_rgb.write(frame_copy)
        
        # Clean up
        plt.close(fig)
        
        processed_count += 1
        
        if show_progress:
            pbar.update(1)
    
    # Clean up
    cap.release()
    out.release()
    out_rgb.release()
    
    if show_progress:
        pbar.close()
    
    print(f"Video processing complete:")
    print(f"  Processed {processed_count}/{total_frames} frames")
    print(f"  Masked output saved to {output_path}")
    print(f"  RGB annotated output saved to {output_rgb_path}")
    
    return output_path, output_rgb_path


def main(args):
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() and not args.cpu else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load categories
    categories = ['background']
    if args.categories_file:
        try:
            with open(args.categories_file, 'r') as f:
                categories = ['background'] + [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(categories)-1} categories from {args.categories_file}")
        except Exception as e:
            print(f"Error loading categories: {e}")
            print("Using default 'background' category only")
    elif args.categories:
        categories = ['background'] + args.categories.split(',')
        print(f"Using {len(categories)-1} categories from command line")
    else:
        print("Warning: No categories provided. Only 'background' will be used.")
    
    # Load model
    num_classes = len(categories)
    print(f"Loading model from {args.model_path} with {num_classes} classes")
    
    try:
        model = load_model(args.model_path, num_classes)
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Check input type and process
    if args.input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Process video
        process_video(args.input_path, model, categories, args.output_dir, device, 
                    args.threshold, args.fps, skip_frames=args.skip_frames, save_json=not args.no_json)
    
    elif os.path.isdir(args.input_path):
        # Process all images in directory
        image_paths = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            image_paths.extend(glob.glob(os.path.join(args.input_path, f"*.{ext}")))
            image_paths.extend(glob.glob(os.path.join(args.input_path, f"*.{ext.upper()}")))
        
        print(f"Found {len(image_paths)} images in {args.input_path}")
        
        total_detections = 0
        start_time = time.time()
        
        for image_path in image_paths:
            total_detections += process_image(
                image_path, model, categories, args.output_dir, 
                device, args.threshold, args.show, save_json=not args.no_json
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\nSummary:")
        print(f"Processed {len(image_paths)} images in {total_time:.2f} seconds")
        print(f"Average time per image: {total_time/max(1, len(image_paths)):.2f} seconds")
        print(f"Total objects detected: {total_detections}")
        
    else:
        # Process single image
        process_image(
            args.input_path, model, categories, args.output_dir, 
            device, args.threshold, args.show, save_json=not args.no_json
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Mask R-CNN on images or video")
    parser.add_argument('input_path', help='path to image, directory of images, or video file')
    parser.add_argument('--model-path', required=True, help='path to model checkpoint')
    parser.add_argument('--output-dir', default='./output', help='directory to save results')
    parser.add_argument('--categories', default='', help='comma-separated list of category names')
    parser.add_argument('--categories-file', default='', help='path to file with category names (one per line)')
    parser.add_argument('--threshold', type=float, default=0.3, help='detection confidence threshold')
    parser.add_argument('--show', action='store_true', help='display images after processing')
    parser.add_argument('--cpu', action='store_true', help='force CPU usage')
    parser.add_argument('--fps', type=float, default=None, help='output video fps (default: same as input)')
    parser.add_argument('--skip-frames', type=int, default=0, help='number of frames to skip (for faster video processing)')
    parser.add_argument('--no-json', action='store_true', help='do not save annotation as JSON')
    
    args = parser.parse_args()
    main(args)