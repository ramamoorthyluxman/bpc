#!/usr/bin/env python3
"""
Run inference with a trained segmentation model on images.

This script:
1. Loads a trained Mask R-CNN model
2. Performs inference on an image or directory of images
3. Visualizes and saves the results with segmentation masks

Usage:
    python inference_fixed.py --model_path path/to/model.pth --input path/to/image.jpg --output path/to/output
"""

import os
import json
import argparse
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb
import torchvision.transforms as T

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--class_mapping', help='Path to class mapping JSON file')
    parser.add_argument('--input', required=True, help='Path to input image or directory')
    parser.add_argument('--output', required=True, help='Path to output directory')
    parser.add_argument('--threshold', type=float, default=0.7, help='Detection threshold')
    parser.add_argument('--device', default='cuda', help='Device to run inference on (cuda or cpu)')
    return parser.parse_args()

def get_model(num_classes):
    """
    Create a Mask R-CNN model with the specified number of classes.
    
    Args:
        num_classes (int): Number of classes (including background)
        
    Returns:
        model: Mask R-CNN model
    """
    # Load Mask R-CNN model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,
        progress=True
    )
    
    # Replace the pre-trained box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model

def load_model(model_path, device, class_mapping_path=None):
    """
    Load the trained model from a checkpoint.
    
    Args:
        model_path (str): Path to model checkpoint
        device (str): Device to load model on ('cuda' or 'cpu')
        class_mapping_path (str): Path to class mapping JSON file
        
    Returns:
        model: Loaded model
        class_names: List of class names
    """
    # Load the checkpoint - with weights_only=False to support older PyTorch models
    try:
        # First try to load with weights_only=False for compatibility
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("Loaded model with weights_only=False")
    except (TypeError, ValueError) as e:
        # If the above fails (older PyTorch versions), try the default
        checkpoint = torch.load(model_path, map_location=device)
        print("Loaded model with default settings")
    
    # Load class mapping if provided, or try to find it in the same directory
    if class_mapping_path:
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
    else:
        # Try to find class mapping in the same directory as the model
        model_dir = os.path.dirname(model_path)
        potential_path = os.path.join(model_dir, 'class_mapping.json')
        
        if os.path.exists(potential_path):
            with open(potential_path, 'r') as f:
                class_mapping = json.load(f)
        else:
            # If no class mapping file is found, create a default one
            print("No class mapping file found. Using default classes from COCO dataset.")
            # Default classes from COCO dataset
            default_classes = [
                'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
                'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
            ]
            class_mapping = {i: cls for i, cls in enumerate(default_classes)}
    
    # Convert string indices back to integers if they are strings
    if any(isinstance(k, str) for k in class_mapping.keys()):
        class_mapping = {int(k): v for k, v in class_mapping.items()}
    
    # Create class names list with background as first class
    num_classes = len(class_mapping) + 1  # +1 for background
    class_names = ['background'] + [class_mapping[i] for i in range(len(class_mapping))]
    
    # Create model
    model = get_model(num_classes)
    
    # Extract model state dict from checkpoint if it's a dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Classes: {class_names}")
    
    return model, class_names

def get_prediction(model, image, threshold, device):
    """
    Get prediction from the model.
    
    Args:
        model: Trained model
        image: PIL Image
        threshold: Detection threshold
        device: Device to run inference on
        
    Returns:
        boxes: Detected bounding boxes
        labels: Detected class labels
        scores: Confidence scores
        masks: Segmentation masks
    """
    # Transform image for the model
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Get predictions with score > threshold
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    masks = prediction[0]['masks'].cpu().numpy()
    
    # Filter by threshold
    keep = scores > threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    masks = masks[keep]
    
    return boxes, labels, scores, masks

def visualize_prediction(image, boxes, labels, scores, masks, class_names, save_path=None):
    """
    Visualize prediction with bounding boxes, labels and masks.
    
    Args:
        image: PIL Image
        boxes: Detected bounding boxes
        labels: Detected class labels
        scores: Confidence scores
        masks: Segmentation masks
        class_names: List of class names
        save_path: Path to save the visualization
        
    Returns:
        None
    """
    # Convert PIL image to numpy array
    image_np = np.array(image)
    height, width = image_np.shape[:2]
    
    # Create figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    # Show original image with bounding boxes and labels
    ax[0].imshow(image_np)
    ax[0].set_title('Detections')
    
    # Create color map for classes
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
    # Draw each box and label
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        # Get class name and color
        class_name = class_names[label]
        color = colors[label][:3]
        
        # Create rectangle patch
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax[0].add_patch(rect)
        
        # Add label with score
        ax[0].text(
            x1, y1-5, f'{class_name}: {score:.2f}',
            color='white', fontsize=10,
            bbox=dict(facecolor=color, alpha=0.8, boxstyle='round,pad=0.2')
        )
    
    ax[0].axis('off')
    
    # Show masks
    # Create a color overlay of all masks
    mask_overlay = np.zeros((height, width, 4), dtype=np.float32)
    
    # Draw each mask with its class color
    for i, (mask, label) in enumerate(zip(masks, labels)):
        # Get mask and color
        mask = mask[0, :, :]  # First channel
        mask = mask > 0.5  # Threshold
        color = colors[label]
        
        # Add to overlay with transparency
        for c in range(3):
            mask_overlay[:, :, c] += mask * color[c] * 0.7
        
        # Update alpha channel for overlap
        mask_overlay[:, :, 3] += mask * 0.3
    
    # Clip values to valid range
    mask_overlay = np.clip(mask_overlay, 0, 1)
    
    # Show mask overlay on a copy of the image
    image_with_masks = image_np.copy().astype(np.float32) / 255.0
    
    # Apply mask overlay using alpha blending
    for c in range(3):
        image_with_masks[:, :, c] = (
            image_with_masks[:, :, c] * (1 - mask_overlay[:, :, 3]) +
            mask_overlay[:, :, c] * mask_overlay[:, :, 3]
        )
    
    # Show image with masks
    ax[1].imshow(image_with_masks)
    ax[1].set_title('Segmentation Masks')
    ax[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def process_image(image_path, model, class_names, threshold, device, output_dir):
    """
    Process a single image and save the results.
    
    Args:
        image_path (str): Path to the input image
        model: Trained model
        class_names: List of class names
        threshold: Detection threshold
        device: Device to run inference on
        output_dir: Directory to save results
        
    Returns:
        None
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Get predictions
        boxes, labels, scores, masks = get_prediction(model, image, threshold, device)
        
        # Create output filename
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_result.png")
        
        # Visualize and save results
        visualize_prediction(image, boxes, labels, scores, masks, class_names, output_path)
        
        # Also save detection results as JSON for later use
        result_data = {
            'image_path': image_path,
            'detections': []
        }
        
        for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
            result_data['detections'].append({
                'box': box.tolist(),
                'label': int(label),
                'class_name': class_names[label],
                'score': float(score)
            })
        
        json_path = os.path.join(output_dir, f"{name}_result.json")
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        # Save mask images separately
        for i, (mask, label) in enumerate(zip(masks, labels)):
            mask_img = (mask[0] > 0.5).astype(np.uint8) * 255
            mask_path = os.path.join(output_dir, f"{name}_mask_{i}_{class_names[label]}.png")
            Image.fromarray(mask_img).save(mask_path)
        
        print(f"Processed {image_path} -> {output_path}")
        print(f"Found {len(boxes)} objects")
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, class_names = load_model(args.model_path, device, args.class_mapping)
    
    # Process input (file or directory)
    if os.path.isfile(args.input):
        # Process single image
        process_image(args.input, model, class_names, args.threshold, device, args.output)
    elif os.path.isdir(args.input):
        # Process all images in directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if any(f.lower().endswith(ext) for ext in image_extensions)
        ]
        
        if not image_files:
            print(f"No images found in {args.input}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        for image_path in image_files:
            process_image(image_path, model, class_names, args.threshold, device, args.output)
    else:
        print(f"Input {args.input} not found")

if __name__ == "__main__":
    main()