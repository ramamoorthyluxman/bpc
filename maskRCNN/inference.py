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


def get_prediction(model, image_tensor, device, threshold=0.5):
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


def process_image(image_path, model, categories, output_dir, device, threshold=0.5, show=False):
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
    """
    print(f"Processing image: {os.path.basename(image_path)}")
    
    # Load image
    img_tensor, original_image = load_image(image_path)
    
    # Get prediction
    output = get_prediction(model, img_tensor, device, threshold)
    
    # Prepare output path
    if output_dir:
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"pred_{base_name}")
    else:
        output_path = None
    
    # Visualize prediction
    visualize_prediction(original_image, output, categories, output_path)
    
    # Display if requested
    if show and output_path:
        img = Image.open(output_path)
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
                 show_progress=True, skip_frames=0):
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
    
    # Create output video
    output_path = os.path.join(output_dir, f"pred_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for better quality
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
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
        
        # Create figure for visualization
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
        
        # Write to output video
        out.write(frame_with_detections)
        
        # Clean up
        plt.close(fig)
        
        processed_count += 1
        
        if show_progress:
            pbar.update(1)
    
    # Clean up
    cap.release()
    out.release()
    
    if show_progress:
        pbar.close()
    
    print(f"Video processing complete:")
    print(f"  Processed {processed_count}/{total_frames} frames")
    print(f"  Output saved to {output_path}")
    
    return output_path


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
                    args.threshold, args.fps, skip_frames=args.skip_frames)
    
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
                device, args.threshold, args.show
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
            device, args.threshold, args.show
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Mask R-CNN on images or video")
    parser.add_argument('input_path', help='path to image, directory of images, or video file')
    parser.add_argument('--model-path', required=True, help='path to model checkpoint')
    parser.add_argument('--output-dir', default='./output', help='directory to save results')
    parser.add_argument('--categories', default='', help='comma-separated list of category names')
    parser.add_argument('--categories-file', default='', help='path to file with category names (one per line)')
    parser.add_argument('--threshold', type=float, default=0.5, help='detection confidence threshold')
    parser.add_argument('--show', action='store_true', help='display images after processing')
    parser.add_argument('--cpu', action='store_true', help='force CPU usage')
    parser.add_argument('--fps', type=float, default=None, help='output video fps (default: same as input)')
    parser.add_argument('--skip-frames', type=int, default=0, help='number of frames to skip (for faster video processing)')
    
    args = parser.parse_args()
    main(args)
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


def get_prediction(model, image_tensor, device, threshold=0.5):
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
        color_tuple = tuple(c * 255 for c in color[:3])
        
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
            mask_color = np.array(color)
            mask_binary = mask > 0.5
            
            # Create colored mask overlay
            masked_image = image_np.copy()
            for c in range(3):  # RGB channels
                masked_image[:, :, c] = np.where(mask_binary, 
                                               masked_image[:, :, c] * 0.5 + color_tuple[c] * 0.5, 
                                               masked_image[:, :, c])
            
            # Blend with original image
            alpha = 0.5
            ax.imshow(np.where(mask_binary[:, :, None], masked_image, image_np), alpha=alpha)
        
        # Add label text
        if show_boxes:
            x1, y1, _, _ = box.astype(int)
            text = ax.text(x1, y1-10, label_text, fontsize=12, color='white')
            text.set_path_effects([patheffects.withStroke(linewidth=2, foreground='black')])
    
    # Remove axes
    ax.axis('off')
    
    # Save or display figure
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def process_image(image_path, model, categories, output_dir, device, threshold=0.5, show=False):
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
    """
    # Load image
    img_tensor, original_image = load_image(image_path)
    
    # Get prediction
    output = get_prediction(model, img_tensor, device, threshold)
    
    # Prepare output path
    if output_dir:
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"pred_{base_name}")
    else:
        output_path = None
    
    # Visualize prediction
    visualize_prediction(original_image, output, categories, output_path)
    
    # Display if requested
    if show and output_path:
        img = Image.open(output_path)
        img.show()
    
    # Print detection summary
    num_detections = len(output['boxes'])
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Found {num_detections} objects above threshold {threshold}:")
    
    for i in range(num_detections):
        label = output['labels'][i].item()
        score = output['scores'][i].item()
        category_name = categories[label] if label < len(categories) else f"Class {label}"
        print(f"  {category_name}: {score:.3f}")
    
    print("")
    
    return num_detections


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
    
    # Process images
    if os.path.isdir(args.input_path):
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
                device, args.threshold, args.show
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\nSummary:")
        print(f"Processed {len(image_paths)} images in {total_time:.2f} seconds")
        print(f"Average time per image: {total_time/len(image_paths):.2f} seconds")
        print(f"Total objects detected: {total_detections}")
        
    else:
        # Process single image
        process_image(
            args.input_path, model, categories, args.output_dir, 
            device, args.threshold, args.show
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Mask R-CNN on images")
    parser.add_argument('input_path', help='path to image or directory of images')
    parser.add_argument('--model-path', required=True, help='path to model checkpoint')
    parser.add_argument('--output-dir', default='./output', help='directory to save results')
    parser.add_argument('--categories', default='', help='comma-separated list of category names')
    parser.add_argument('--categories-file', default='', help='path to file with category names (one per line)')
    parser.add_argument('--threshold', type=float, default=0.5, help='detection confidence threshold')
    parser.add_argument('--show', action='store_true', help='display images after processing')
    parser.add_argument('--cpu', action='store_true', help='force CPU usage')
    
    args = parser.parse_args()
    main(args)
