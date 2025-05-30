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
    """
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


def load_images_batch(image_paths):
    """
    Load a batch of images and convert to tensors
    """
    image_tensors = []
    original_images = []
    valid_paths = []
    
    transform = T.Compose([T.ToTensor()])
    
    for image_path in image_paths:
        try:
            original_image = Image.open(image_path).convert("RGB")
            image_tensor = transform(original_image)
            
            image_tensors.append(image_tensor)
            original_images.append(original_image)
            valid_paths.append(image_path)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            continue
    
    return image_tensors, original_images, valid_paths


def get_batch_predictions_optimized(model, image_tensors, device, threshold=0.3):
    """
    Get model predictions with minimal GPU-CPU transfers
    
    Key optimization: Keep everything on GPU until final processing
    """
    # Move tensors to device
    imgs = [img.to(device, non_blocking=True) for img in image_tensors]  # non_blocking for async transfer
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(imgs)
    
    # Filter by threshold ON GPU (this is the key optimization)
    filtered_outputs = []
    for output in outputs:
        # Keep filtering on GPU
        keep = output['scores'] > threshold
        
        # Only create filtered output if we have detections
        if keep.sum() > 0:
            filtered_output = {
                'boxes': output['boxes'][keep],      # Still on GPU
                'labels': output['labels'][keep],    # Still on GPU
                'scores': output['scores'][keep],    # Still on GPU
                'masks': output['masks'][keep]       # Still on GPU
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


def process_single_result_optimized(output, image_path, original_image, categories, output_dir, save_json=True, show_centers=True, visualize_and_save=True):
    """
    Process a single result with optimized GPU-CPU transfers
    
    Modified to ALWAYS return JSON data regardless of visualize_and_save setting
    Returns: (num_detections, json_data) - json_data is always provided
    """
    # Early exit for empty results (no GPU transfer needed)
    if len(output['boxes']) == 0:
        base_name = os.path.basename(image_path)
        width, height = original_image.size
        
        # Create empty annotation data
        annotation_data = {
            "image_path": image_path,
            "height": height,
            "width": width,
            "masks": []
        }
        
        if visualize_and_save:
            # Save empty annotated image
            rgb_annotated_path = os.path.join(output_dir, f"rgb_annotated_{base_name}")
            image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(rgb_annotated_path, image_cv)
            
            if save_json:
                json_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_annotation.json")
                with open(json_path, 'w') as f:
                    json.dump(annotation_data, f)
        
        # Always return JSON data
        return 0, annotation_data
    
    # NOW we transfer to CPU (only when we have detections to process)
    # Transfer in one go to minimize overhead
    cpu_output = {}
    cpu_output['boxes'] = output['boxes'].cpu()
    cpu_output['labels'] = output['labels'].cpu()  
    cpu_output['scores'] = output['scores'].cpu()
    
    # For masks, only transfer if we need them
    if 'masks' in output and len(output['masks']) > 0:
        cpu_output['masks'] = output['masks'].cpu()
    else:
        cpu_output['masks'] = None
    
    # Convert to numpy only when needed for OpenCV operations
    boxes_np = cpu_output['boxes'].numpy()
    labels_np = cpu_output['labels'].numpy()
    scores_np = cpu_output['scores'].numpy()
    masks_np = cpu_output['masks'].numpy() if cpu_output['masks'] is not None else None
    
    base_name = os.path.basename(image_path)
    width, height = original_image.size
    
    # Always generate JSON data
    annotation_data = generate_annotations_data(image_path, height, width, boxes_np, labels_np, masks_np, categories)
    
    if visualize_and_save:
        # Save annotated RGB image
        rgb_annotated_path = os.path.join(output_dir, f"rgb_annotated_{base_name}")
        save_rgb_annotated_image_optimized(original_image, boxes_np, labels_np, scores_np, masks_np, 
                                         categories, rgb_annotated_path, show_centers)
        
        # Save JSON annotation to file
        if save_json:
            json_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_annotation.json")
            save_annotations_optimized(image_path, height, width, boxes_np, labels_np, masks_np, 
                                     categories, json_path)
    
    # Always return JSON data
    return len(boxes_np), annotation_data


def generate_annotations_data(image_path, height, width, boxes_np, labels_np, masks_np, categories):
    """
    Generate annotation data without saving to disk
    """
    annotation_data = {
        "image_path": image_path,
        "height": height,
        "width": width,
        "masks": []
    }
    
    if masks_np is not None and len(masks_np) > 0:
        for i, (mask, box, label) in enumerate(zip(masks_np, boxes_np, labels_np)):
            category_name = categories[label] if label < len(categories) else f"Class_{label}"
            
            # Quick contour extraction
            mask_binary = (mask[0] > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                points = [[int(point[0][0]), int(point[0][1])] for point in largest_contour]
                
                # Calculate centers
                x1, y1, x2, y2 = box.astype(int)
                bbox_center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                
                if points:
                    points_array = np.array(points)
                    geometric_center = [int(np.mean(points_array[:, 0])), int(np.mean(points_array[:, 1]))]
                else:
                    geometric_center = bbox_center
                
                mask_data = {
                    "label": category_name,
                    "points": points,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "bbox_center": bbox_center,
                    "geometric_center": geometric_center
                }
                annotation_data["masks"].append(mask_data)
    
    return annotation_data


def save_rgb_annotated_image_optimized(original_image, boxes_np, labels_np, scores_np, masks_np, 
                                     categories, output_path, show_centers=True):
    """
    Optimized RGB annotation saving - takes numpy arrays directly
    """
    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    # Pre-compute colors once
    colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))
    
    # Apply masks to the original image
    if masks_np is not None and len(masks_np) > 0:
        mask_overlay = image_cv.copy()
        
        for i, (mask, label) in enumerate(zip(masks_np, labels_np)):
            color = colors[label % len(colors)]
            color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
            
            mask_binary = (mask[0] > 0.5).astype(np.uint8)
            
            # Vectorized mask application
            mask_indices = mask_binary == 1
            if np.any(mask_indices):
                mask_overlay[mask_indices] = cv2.addWeighted(
                    mask_overlay[mask_indices], 
                    0.5,
                    np.full_like(mask_overlay[mask_indices], color_bgr),
                    0.5,
                    0
                )
                
                # Add geometric center if requested
                if show_centers:
                    y_coords, x_coords = np.where(mask_binary)
                    if len(x_coords) > 0:
                        centroid_x = int(np.mean(x_coords))
                        centroid_y = int(np.mean(y_coords))
                        
                        cv2.circle(mask_overlay, (centroid_x, centroid_y), 8, (255, 255, 255), -1)
                        cv2.circle(mask_overlay, (centroid_x, centroid_y), 6, color_bgr, -1)
        
        image_cv = mask_overlay
    
    # Draw bounding boxes and labels
    for i, (box, label, score) in enumerate(zip(boxes_np, labels_np, scores_np)):
        color = colors[label % len(colors)]
        color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
        
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color_bgr, 2)
        
        category_name = categories[label] if label < len(categories) else f"Class {label}"
        label_text = f"{category_name}: {score:.2f}"
        
        text_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image_cv, (x1, y1-text_size[1]-10), (x1+text_size[0], y1), color_bgr, -1)
        cv2.putText(image_cv, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, image_cv)


def save_annotations_optimized(image_path, height, width, boxes_np, labels_np, masks_np, categories, output_path):
    """
    Optimized annotation saving - takes numpy arrays directly
    """
    annotation_data = {
        "image_path": image_path,
        "height": height,
        "width": width,
        "masks": []
    }
    
    if masks_np is not None and len(masks_np) > 0:
        mask_dir = os.path.join(os.path.dirname(output_path), "masks")
        os.makedirs(mask_dir, exist_ok=True)
        
        base_name = os.path.basename(image_path).split('.')[0]
        
        for i, (mask, box, label) in enumerate(zip(masks_np, boxes_np, labels_np)):
            category_name = categories[label] if label < len(categories) else f"Class_{label}"
            
            # Quick contour extraction
            mask_binary = (mask[0] > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                points = [[int(point[0][0]), int(point[0][1])] for point in largest_contour]
                
                # Save mask
                mask_path = f"masks/{base_name}_{i}.png"
                full_mask_path = os.path.join(os.path.dirname(output_path), mask_path)
                cv2.imwrite(full_mask_path, mask_binary * 255)
                
                # Calculate centers
                x1, y1, x2, y2 = box.astype(int)
                bbox_center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                
                if points:
                    points_array = np.array(points)
                    geometric_center = [int(np.mean(points_array[:, 0])), int(np.mean(points_array[:, 1]))]
                else:
                    geometric_center = bbox_center
                
                mask_data = {
                    "label": category_name,
                    "points": points,
                    "mask_path": mask_path,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "bbox_center": bbox_center,
                    "geometric_center": geometric_center
                }
                annotation_data["masks"].append(mask_data)
    
    with open(output_path, 'w') as f:
        json.dump(annotation_data, f, indent=2)


def process_batch_optimized(image_paths, model, categories, output_dir, device, threshold=0.5, save_json=True, show_centers=True, visualize_and_save=True):
    """
    Process a batch of images with optimized GPU transfers
    
    Modified to ALWAYS return JSON data regardless of visualize_and_save setting
    Returns: (total_detections, batch_json_data) - batch_json_data is always provided
    """
    #print(f"Processing batch of {len(image_paths)} images...")
    
    # Load batch of images
    image_tensors, original_images, valid_paths = load_images_batch(image_paths)
    
    if not image_tensors:
        #print("No valid images in batch")
        return 0, []
    
    # Get batch predictions (tensors stay on GPU)
    start_time = time.time()
    outputs = get_batch_predictions_optimized(model, image_tensors, device, threshold)
    inference_time = time.time() - start_time
    
    #print(f"Batch inference time: {inference_time:.2f} seconds ({inference_time/len(image_tensors):.2f} per image)")
    
    # Process each result with optimized transfers
    total_detections = 0
    batch_json_data = []  # Always collect JSON data
    postprocess_start = time.time()
    
    for output, image_path, original_image in zip(outputs, valid_paths, original_images):
        detections, json_data = process_single_result_optimized(
            output, image_path, original_image, categories, output_dir, save_json, show_centers, visualize_and_save
        )
        total_detections += detections
        
        # Always collect JSON data
        if json_data is not None:
            batch_json_data.append(json_data)
        
        base_name = os.path.basename(image_path)
        #print(f"  {base_name}: {detections} objects detected")
    
    postprocess_time = time.time() - postprocess_start
    #print(f"Post-processing time: {postprocess_time:.2f} seconds ({postprocess_time/len(image_tensors):.2f} per image)")
    
    # Clear GPU cache after batch
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return total_detections, batch_json_data


def infer(
    image_path,
    model_path,
    output_path,
    category_txt_path,
    confidence_threshold=0.8,
    batch_size=4,
    visualize_and_save=True
):
    """
    GPU-optimized inference function
    
    Modified to ALWAYS return JSON data regardless of visualize_and_save setting
    
    Args:
        visualize_and_save (bool): If True, creates and saves visualizations and files. 
                                  If False, only returns JSON data without saving visualizations.
    
    Returns:
        dict: Always returns dict with all JSON annotation data regardless of visualize_and_save setting.
    """
    # Set device and enable some basic optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")
    
    if device.type == 'cuda':
        # Enable some basic CUDA optimizations
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
        #print(f"Error loading model: {e}")
        return None
    
    # Create output directory only if visualize_and_save is True
    if visualize_and_save:
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
    #print(f"Visualize and save: {visualize_and_save}")
    
    # Process in batches
    total_detections = 0
    all_json_data = []  # Always collect JSON data
    total_start_time = time.time()
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        #print(f"\nProcessing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
        
        batch_detections, batch_json_data = process_batch_optimized(
            batch_paths, model, categories, output_path, 
            device, confidence_threshold, save_json=True, show_centers=True, visualize_and_save=visualize_and_save
        )
        
        total_detections += batch_detections
        
        # Always collect JSON data
        if batch_json_data is not None:
            all_json_data.extend(batch_json_data)
    
    # Summary
    total_time = time.time() - total_start_time
    #print(f"\n{'='*60}")
    #print(f"GPU-OPTIMIZED PROCESSING COMPLETE")
    #print(f"{'='*60}")
    #print(f"Total images processed: {len(image_paths)}")
    #print(f"Total processing time: {total_time:.2f} seconds")
    #print(f"Average time per image: {total_time/len(image_paths):.2f} seconds")
    #print(f"Total objects detected: {total_detections}")
    #print(f"{'='*60}")
    
    # Always return JSON data
    return {
        "total_images": len(image_paths),
        "total_detections": total_detections,
        "processing_time": total_time,
        "results": all_json_data
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-Optimized Mask R-CNN Inference")
    parser.add_argument('input_path', help='path to image or directory of images')
    parser.add_argument('--model-path', required=True, help='path to model checkpoint')
    parser.add_argument('--output-dir', default='./output', help='directory to save results')
    parser.add_argument('--categories-file', required=True, help='path to categories file')
    parser.add_argument('--threshold', type=float, default=0.5, help='detection confidence threshold')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size for processing')
    parser.add_argument('--no-visualize', action='store_true', help='skip visualization and return only JSON data')
    
    args = parser.parse_args()
    
    result = infer(
        image_path=args.input_path,
        model_path=args.model_path,
        output_path=args.output_dir,
        category_txt_path=args.categories_file,
        confidence_threshold=args.threshold,
        batch_size=args.batch_size,
        visualize_and_save=not args.no_visualize
    )
    
    # Result always contains JSON data now
    if result is not None:
        print(f"\nReturned JSON data with {len(result['results'])} image results")
        # You could save this to a single JSON file or process it further as needed
        # Example: 
        # with open('inference_results.json', 'w') as f:
        #     json.dump(result, f, indent=2)