"""
Inference script for using the fine-tuned SAM model.
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
from typing import List, Tuple, Dict, Union, Optional

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
try:
    from model import SAMForFineTuning, create_sam_model
except ImportError:
    print("Error importing model module. Make sure model.py is in the current directory.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned SAM model")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to fine-tuned SAM checkpoint")
    parser.add_argument("--sam_checkpoint", type=str, default=None,
                        help="Path to original SAM checkpoint (if not included in the fine-tuned checkpoint)")
    parser.add_argument("--model_type", type=str, default="vit_h", 
                        choices=["vit_h", "vit_l", "vit_b"],
                        help="SAM model type")
    
    # Input parameters
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--points", type=str, default=None,
                        help="Points to use as prompts, format: 'x1,y1;x2,y2;...'")
    parser.add_argument("--annotation_path", type=str, default=None,
                        help="Path to annotation JSON file (will extract points from annotations)")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./predictions",
                        help="Output directory for predictions")
    parser.add_argument("--save_masks", action="store_true",
                        help="Save binary masks as separate files")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha value for mask overlay")
    
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Threshold for binary mask")
    
    return parser.parse_args()

def prepare_image(image_path: str, target_size: int = 1024) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], np.ndarray]:
    """
    Load and prepare image for SAM.
    
    Args:
        image_path: Path to image file
        target_size: Target size for the image (SAM requires 1024x1024)
        
    Returns:
        original_image: Original image
        processed_image: Processed image in the format expected by SAM
        original_size: Original image size (H, W)
        transform_matrix: Transformation matrix for converting between original and processed coordinates
    """
    # Load image
    original_image = np.array(Image.open(image_path).convert("RGB"))
    h, w = original_image.shape[:2]
    original_size = (h, w)
    
    # Preprocess image to exact square size (1024x1024)
    # Calculate scale to fit within target_size
    scale = min(target_size / h, target_size / w)
    
    # Calculate new dimensions
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create square image with padding
    square_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Calculate padding
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    
    # Place resized image in center
    square_img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    # Convert to tensor format
    processed_image = torch.from_numpy(square_img.transpose(2, 0, 1)).float() / 255.0
    
    # Create transformation matrix for points (from original to processed)
    transform_matrix = np.array([
        [scale, 0, pad_w],
        [0, scale, pad_h],
        [0, 0, 1]  # Add homogeneous coordinate row for proper transformation
    ])
    
    return original_image, processed_image, original_size, transform_matrix

def parse_points(points_str: str) -> np.ndarray:
    """
    Parse points from string format.
    
    Args:
        points_str: Points in format "x1,y1;x2,y2;..."
        
    Returns:
        points: Array of points [N, 2]
    """
    points = []
    for point_pair in points_str.split(';'):
        if not point_pair:
            continue
        x, y = map(float, point_pair.split(','))
        points.append([x, y])
    
    return np.array(points)

def extract_points_from_annotation(annotation_path: str, strategy: str = "center") -> List[np.ndarray]:
    """
    Extract points from annotation file.
    
    Args:
        annotation_path: Path to annotation JSON file
        strategy: Strategy to extract points ("center", "random", "bbox")
        
    Returns:
        points: List of points for each shape
    """
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    points_list = []
    
    for shape in annotation["shapes"]:
        if shape["shape_type"] != "polygon":
            continue
        
        polygon_points = shape["points"]
        if not polygon_points:
            continue
        
        # Convert to numpy array
        polygon = np.array(polygon_points)
        
        if strategy == "center":
            # Use center of polygon
            center = np.mean(polygon, axis=0)
            points_list.append(np.array([center]))
        elif strategy == "random":
            # Use a random point from the polygon
            random_idx = np.random.randint(0, len(polygon))
            points_list.append(np.array([polygon[random_idx]]))
        elif strategy == "bbox":
            # Use corners of bounding box
            x_min, y_min = np.min(polygon, axis=0)
            x_max, y_max = np.max(polygon, axis=0)
            points_list.append(np.array([[x_min, y_min]]))
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
    
    return points_list

def transform_point(point: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Transform point using the transformation matrix.
    
    Args:
        point: Point coordinates [x, y]
        transform_matrix: 3x3 transformation matrix
        
    Returns:
        Transformed point [x, y]
    """
    # Convert to homogeneous coordinates
    point_h = np.array([point[0], point[1], 1])
    
    # Apply transformation
    transformed = transform_matrix @ point_h
    
    return transformed[:2]

def predict_masks(
    model: torch.nn.Module, 
    image: torch.Tensor, 
    points: List[np.ndarray],
    original_size: Tuple[int, int],
    transform_matrix: np.ndarray,
    device: torch.device
) -> Tuple[np.ndarray, List[float]]:
    """
    Predict masks for image using points as prompts.
    
    Args:
        model: SAM model
        image: Image tensor [3, H, W]
        points: List of points to use as prompts (in original image coordinates)
        original_size: Original image size (H, W)
        transform_matrix: Transformation matrix to transform points
        device: Device to use
        
    Returns:
        masks: Predicted masks [N, 1, H, W]
        scores: Confidence scores for each mask
    """
    model.eval()
    
    # Prepare inputs
    batch_image = image.unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # Transform points to match the processed image
    transformed_points = []
    for point_set in points:
        t_points = np.array([transform_point(p, transform_matrix) for p in point_set])
        transformed_points.append(torch.from_numpy(t_points).float().to(device))
    
    # Predict masks
    with torch.no_grad():
        pred_masks, metrics = model(batch_image, transformed_points, [original_size])
    
    # Convert to numpy
    masks = pred_masks.cpu().numpy()  # [N, 1, H, W]
    scores = metrics["iou"] if isinstance(metrics, dict) and "iou" in metrics else [0.0] * len(masks)
    
    return masks, scores

def visualize_predictions(
    image: np.ndarray,
    masks: np.ndarray,
    points: List[np.ndarray],
    scores: List[float],
    transform_matrix: np.ndarray,
    output_path: str,
    threshold: float = 0.0,
    alpha: float = 0.5
):
    """
    Visualize predicted masks overlaid on the original image with bounding boxes.
    
    Args:
        image: Original image (H, W, 3)
        masks: Predicted masks [N, 1, H_mask, W_mask]
        points: Points used as prompts (in original image coordinates)
        scores: Confidence scores for each mask
        transform_matrix: Transformation matrix (original to processed)
        output_path: Path to save visualization
        threshold: Threshold for binary mask
        alpha: Alpha value for mask overlay
    """
    # Create figure with a single axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Show original image
    ax.imshow(image)
    
    # Process each mask
    for i, mask in enumerate(masks):
        # Get binary mask and resize to original image dimensions
        binary_mask = (mask[0] > threshold).astype(np.uint8)
        binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
        
        if np.sum(binary_mask) == 0:  # Skip empty masks
            continue
            
        # Find contours to get bounding box
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get color for this mask (using a colormap)
        color = plt.cm.tab10(i % 10)
        color_rgb = (np.array(color[:3]) * 255).astype(np.uint8)
        
        # Create RGBA mask for overlay
        colored_mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        colored_mask[binary_mask > 0] = [color_rgb[0], color_rgb[1], color_rgb[2], int(alpha * 255)]
        ax.imshow(colored_mask)
        
        # Draw bounding box
        for contour in contours:
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            rect = plt.Rectangle((x, y), w_rect, h_rect, 
                                fill=False, color=color, linewidth=2)
            ax.add_patch(rect)
        
        # Draw points
        if i < len(points):
            for p in points[i]:
                ax.scatter(p[0], p[1], c=[color], s=80, marker="*", edgecolors='white', linewidths=1)
        
        # Add label with score
        score = scores[i] if i < len(scores) else 0.0
        if len(contours) > 0:
            x, y, _, _ = cv2.boundingRect(contours[0])
            ax.text(x, y - 5, f"Obj {i+1}: {score:.2f}", 
                   color='white', fontsize=12, bbox=dict(facecolor=color, alpha=0.7))
    
    ax.set_title("Segmentation Results with Bounding Boxes")
    ax.axis("off")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return output_path

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare image
    image_file = os.path.basename(args.image_path)
    image_name = os.path.splitext(image_file)[0]
    
    print(f"Processing image: {image_file}")
    original_image, processed_image, original_size, transform_matrix = prepare_image(
        args.image_path, target_size=1024
    )
    
    # Get points
    if args.annotation_path:
        print(f"Extracting points from annotation: {args.annotation_path}")
        points = extract_points_from_annotation(args.annotation_path, strategy="center")
    elif args.points:
        print(f"Using provided points: {args.points}")
        points = [parse_points(args.points)]
    else:
        # If no points are provided, use the center of the image
        h, w = original_size
        center_point = np.array([[w/2, h/2]])
        points = [center_point]
        print(f"No points provided, using center point: {center_point}")
    
    # Set up device
    device = torch.device(args.device)
    
    # Try loading the fine-tuned model directly
    print(f"Loading model from checkpoint: {args.checkpoint}")
    try:
        # Load with weights_only=False since we're loading our own trusted checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        
        # If the checkpoint has the model state dict nested
        if "model_state_dict" in checkpoint:
            # First initialize the model
            if args.sam_checkpoint:
                # Initialize with original SAM checkpoint
                model = create_sam_model(
                    checkpoint_path=args.sam_checkpoint,
                    model_type=args.model_type,
                    device=args.device
                )
            else:
                # Try to initialize with empty model
                print("Warning: No original SAM checkpoint provided. Using default initialization.")
                model = create_sam_model(
                    checkpoint_path=None,
                    model_type=args.model_type,
                    device=args.device
                )
            
            # Then load the fine-tuned weights
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # If checkpoint is the model state dict directly
            model = create_sam_model(
                checkpoint_path=args.sam_checkpoint or None,
                model_type=args.model_type,
                device=args.device
            )
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        print("Trying to load as a standard SAM checkpoint...")
        try:
            # For original SAM checkpoint
            from segment_anything import sam_model_registry
            model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
            model.to(device)
        except Exception as e2:
            print(f"Failed to load checkpoint as standard SAM model: {e2}")
            print("\nSuggestion: Try running with the original SAM checkpoint:")
            print("python predict.py --checkpoint /path/to/sam_vit_h_4b8939.pth --model_type vit_h --image_path your_image.jpg")
            sys.exit(1)
    
    # Move model to device
    model.to(device)
    
    # Predict masks
    print("Predicting masks...")
    masks, scores = predict_masks(
        model, processed_image, points, original_size, transform_matrix, device
    )
    
    # Visualize predictions
    print("Visualizing predictions...")
    output_path = os.path.join(args.output_dir, f"{image_name}_prediction.png")
    visualize_predictions(
        original_image, masks, points, scores, transform_matrix,
        output_path, args.threshold, args.alpha
    )
    
    # Save binary masks if requested
    if args.save_masks:
        print("Saving binary masks...")
        for i, mask in enumerate(masks):
            mask_binary = (mask[0] > args.threshold).astype(np.uint8) * 255
            mask_path = os.path.join(args.output_dir, f"{image_name}_mask_{i+1}.png")
            cv2.imwrite(mask_path, mask_binary)
    
    print(f"Predictions saved to: {args.output_dir}")
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    main()