import os
import argparse
import torch
import numpy as np
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
    """
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


def load_images_batch(image_paths):
    """
    Load a batch of images and convert to tensors
    """
    image_tensors = []
    image_sizes = []  # Store sizes for JSON instead of full PIL images
    valid_paths = []
    
    transform = T.Compose([T.ToTensor()])
    
    for image_path in image_paths:
        try:
            original_image = Image.open(image_path).convert("RGB")
            image_tensor = transform(original_image)
            
            image_tensors.append(image_tensor)
            image_sizes.append(original_image.size)  # (width, height)
            valid_paths.append(image_path)
        except Exception as e:
            #print(f"Error loading {image_path}: {e}")
            continue
    
    return image_tensors, image_sizes, valid_paths


def get_batch_predictions_optimized(model, image_tensors, device, threshold=0.3):
    """
    Get model predictions with minimal GPU-CPU transfers
    """
    # Move tensors to device
    imgs = [img.to(device, non_blocking=True) for img in image_tensors]
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(imgs)
    
    # Filter by threshold ON GPU
    filtered_outputs = []
    for output in outputs:
        keep = output['scores'] > threshold
        
        if keep.sum() > 0:
            filtered_output = {
                'boxes': output['boxes'][keep],
                'labels': output['labels'][keep],
                'scores': output['scores'][keep],
                'masks': output['masks'][keep]
            }
        else:
            # Empty tensors but keep them on GPU
            filtered_output = {
                'boxes': torch.empty(0, 4, device=device),
                'labels': torch.empty(0, dtype=torch.long, device=device),
                'scores': torch.empty(0, device=device),
                'masks': torch.empty(0, 1, 1, 1, device=device)
            }
        
        filtered_outputs.append(filtered_output)
    
    return filtered_outputs


def extract_mask_contours_fast(mask_np):
    """
    Fast contour extraction for JSON annotations only
    """
    mask_binary = (mask_np > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Get largest contour and simplify it
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify contour to reduce JSON size (optional - remove if you want full precision)
    epsilon = 0.002 * cv2.arcLength(largest_contour, True)
    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to list format
    points = [[int(point[0][0]), int(point[0][1])] for point in simplified_contour]
    
    return points


def process_single_result_json_only(output, image_path, image_size, categories, output_dir):
    """
    Process a single result - JSON annotation only (no visualization)
    """
    width, height = image_size
    
    # Early exit for empty results
    if len(output['boxes']) == 0:
        json_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_annotation.json")
        annotation_data = {
            "image_path": image_path,
            "height": height,
            "width": width,
            "masks": []
        }
        with open(json_path, 'w') as f:
            json.dump(annotation_data, f)
        return 0
    
    # Transfer to CPU only once
    boxes_np = output['boxes'].cpu().numpy()
    labels_np = output['labels'].cpu().numpy()
    scores_np = output['scores'].cpu().numpy()
    masks_np = output['masks'].cpu().numpy() if len(output['masks']) > 0 else None
    
    # Prepare annotation data
    annotation_data = {
        "image_path": image_path,
        "height": height,
        "width": width,
        "masks": []
    }
    
    # Process masks for JSON
    if masks_np is not None and len(masks_np) > 0:
        base_name = os.path.basename(image_path).split('.')[0]
        
        # Create masks directory for individual mask files
        mask_dir = os.path.join(output_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        
        for i, (mask, box, label, score) in enumerate(zip(masks_np, boxes_np, labels_np, scores_np)):
            category_name = categories[label] if label < len(categories) else f"Class_{label}"
            
            # Extract contour points
            points = extract_mask_contours_fast(mask[0])
            
            if not points:
                continue
            
            # Save individual mask file (binary PNG)
            mask_path = f"masks/{base_name}_{i}.png"
            full_mask_path = os.path.join(output_dir, mask_path)
            mask_binary = (mask[0] > 0.5).astype(np.uint8) * 255
            cv2.imwrite(full_mask_path, mask_binary)
            
            # Calculate centers
            x1, y1, x2, y2 = box.astype(int)
            bbox_center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            
            # Calculate geometric center from points
            if points:
                points_array = np.array(points)
                geometric_center = [int(np.mean(points_array[:, 0])), int(np.mean(points_array[:, 1]))]
            else:
                geometric_center = bbox_center
            
            # Add to annotation data
            mask_data = {
                "label": category_name,
                "confidence": float(score),  # Add confidence score
                "points": points,
                "mask_path": mask_path,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "bbox_center": bbox_center,
                "geometric_center": geometric_center,
                "area": int(cv2.contourArea(np.array(points).reshape(-1, 1, 2))) if points else 0
            }
            annotation_data["masks"].append(mask_data)
    
    # Save JSON annotation
    json_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_annotation.json")
    with open(json_path, 'w') as f:
        json.dump(annotation_data, f, indent=2)
    
    return len(boxes_np)


def process_batch_json_only(image_paths, model, categories, output_dir, device, threshold=0.5):
    """
    Process a batch of images - JSON annotations only
    """
    #print(f"Processing batch of {len(image_paths)} images...")
    
    # Load batch of images (now returns sizes instead of full PIL images)
    image_tensors, image_sizes, valid_paths = load_images_batch(image_paths)
    
    if not image_tensors:
        #print("No valid images in batch")
        return 0
    
    # Get batch predictions (tensors stay on GPU)
    start_time = time.time()
    outputs = get_batch_predictions_optimized(model, image_tensors, device, threshold)
    inference_time = time.time() - start_time
    
    #print(f"Batch inference time: {inference_time:.2f} seconds ({inference_time/len(image_tensors):.2f} per image)")
    
    # Process each result - JSON only
    total_detections = 0
    postprocess_start = time.time()
    
    for output, image_path, image_size in zip(outputs, valid_paths, image_sizes):
        detections = process_single_result_json_only(
            output, image_path, image_size, categories, output_dir
        )
        total_detections += detections
        
        base_name = os.path.basename(image_path)
        #print(f"  {base_name}: {detections} objects detected")
    
    postprocess_time = time.time() - postprocess_start
    #print(f"JSON generation time: {postprocess_time:.2f} seconds ({postprocess_time/len(image_tensors):.2f} per image)")
    
    # Clear GPU cache after batch
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return total_detections


def infer_json_only(
    image_path,
    model_path,
    output_path,
    category_txt_path,
    confidence_threshold=0.8,
    batch_size=4
):
    """
    JSON-only inference function (no visualization)
    """
    # Set device and enable optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f"Using device: {device}")
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Load categories
    categories = ['background']
    try:
        with open(category_txt_path, 'r') as f:
            categories = ['background'] + [line.strip() for line in f if line.strip()]
        #print(f"Loaded {len(categories)-1} categories from {category_txt_path}")
    except Exception as e:
        print(f"Error loading categories: {e}")
        print("Using default 'background' category only")
    
    # Load model
    num_classes = len(categories)
    #print(f"Loading model from {model_path} with {num_classes} classes")

    try:
        model = load_model(model_path, num_classes)
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Collect image paths
    if os.path.isdir(image_path):
        image_paths = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'PNG', 'BMP']:
            image_paths.extend(glob.glob(os.path.join(image_path, f"*.{ext}")))
    else:
        image_paths = [image_path]
    
    #print(f"Found {len(image_paths)} images to process")
    #print(f"Using batch size: {batch_size}")
    #print("Mode: JSON annotations only (no visualization)")
    
    # Process in batches
    total_detections = 0
    total_start_time = time.time()
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        #print(f"\nProcessing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
        
        batch_detections = process_batch_json_only(
            batch_paths, model, categories, output_path, 
            device, confidence_threshold
        )
        
        total_detections += batch_detections
    
    # Summary
    total_time = time.time() - total_start_time
    #print(f"\n{'='*60}")
    #print(f"JSON-ONLY PROCESSING COMPLETE")
    #print(f"{'='*60}")
    #print(f"Total images processed: {len(image_paths)}")
    #print(f"Total processing time: {total_time:.2f} seconds")
    #print(f"Average time per image: {total_time/len(image_paths):.2f} seconds")
    #print(f"Total objects detected: {total_detections}")
    #print(f"Throughput: {len(image_paths)/total_time:.2f} images/second")
    #print(f"{'='*60}")


def infer(
    image_path,
    model_path,
    output_path,
    category_txt_path,
    confidence_threshold=0.8
):
    """
    Wrapper function to maintain compatibility with existing config.py
    """
    infer_json_only(
        image_path=image_path,
        model_path=model_path,
        output_path=output_path,
        category_txt_path=category_txt_path,
        confidence_threshold=confidence_threshold,
        batch_size=8  # Increased default since no visualization overhead
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSON-Only Mask R-CNN Inference")
    parser.add_argument('input_path', help='path to image or directory of images')
    parser.add_argument('--model-path', required=True, help='path to model checkpoint')
    parser.add_argument('--output-dir', default='./output', help='directory to save results')
    parser.add_argument('--categories-file', required=True, help='path to categories file')
    parser.add_argument('--threshold', type=float, default=0.5, help='detection confidence threshold')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for processing')
    
    args = parser.parse_args()
    
    infer_json_only(
        image_path=args.input_path,
        model_path=args.model_path,
        output_path=args.output_dir,
        category_txt_path=args.categories_file,
        confidence_threshold=args.threshold,
        batch_size=args.batch_size
    )