import cv2
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import ast
import re
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

print("Using GMS with polygon-constrained matching and consensus-based transformation (DEBUG MODE)")

def extract_camera_type_from_path(image_path: str) -> str:
    """
    Extract camera type from image path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Camera type (e.g., 'cam1', 'photoneo', etc.)
    """
    # Convert to Path object for easier manipulation
    path_obj = Path(image_path)
    filename = path_obj.stem  # Get filename without extension
    
    # Common camera type patterns
    camera_patterns = [
        r'_cam(\d+)',           # _cam1, _cam2, etc.
        r'_photoneo',           # _photoneo
        r'_kinect',             # _kinect
        r'_realsense',          # _realsense
        r'_zed',                # _zed
        r'_([a-zA-Z]+\d*)',     # any pattern like _camera1, _sensor2, etc.
    ]
    
    for pattern in camera_patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            if 'cam' in pattern:
                return f"cam{match.group(1)}"  # Return cam1, cam2, etc.
            elif 'photoneo' in pattern:
                return "photoneo"
            elif 'kinect' in pattern:
                return "kinect"
            elif 'realsense' in pattern:
                return "realsense"
            elif 'zed' in pattern:
                return "zed"
            else:
                return match.group(1)  # Return the captured group
    
    # Fallback: try to extract from directory structure
    # Look for camera info in parent directories
    parts = path_obj.parts
    for part in reversed(parts):
        if 'cam' in part.lower():
            cam_match = re.search(r'cam(\d+)', part, re.IGNORECASE)
            if cam_match:
                return f"cam{cam_match.group(1)}"
        elif 'photoneo' in part.lower():
            return "photoneo"
        elif 'kinect' in part.lower():
            return "kinect"
        elif 'realsense' in part.lower():
            return "realsense"
        elif 'zed' in part.lower():
            return "zed"
    
    # Default fallback
    print(f"Warning: Could not extract camera type from path: {image_path}")
    return "unknown"

def object_matching_and_pose_estimation(json_file_path: str, csv_file_path: str, output_dir: str = "output") -> List[Dict]:
    """
    Match detected objects with dataset images and estimate pose.
    Uses GMS with polygon-constrained matching and consensus-based transformation.
    
    Args:
        json_file_path (str): Path to JSON file containing object detection results
        csv_file_path (str): Path to CSV file containing dataset with ground truth
        output_dir (str): Directory to save visualization images (default: "output")
    
    Returns:
        List[Dict]: Results containing matches, correspondences, and pose information
        
    Matching Method:
        1. Extract GMS features ONLY from within polygon regions (not bounding boxes)
        2. Filter keypoints to be strictly within polygon boundaries
        3. Apply GMS filtering for ultra-robust correspondences
        4. Use consensus-based transformation for maximum robustness
        - Smart approach: only match actual object regions, ignore background
        - Robust transformation: try multiple subsets and pick best consensus
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created: {output_path.absolute()}")
    
    # Load JSON data
    with open(json_file_path, 'r') as f:
        detection_data = json.load(f)
    
    # Load CSV data
    dataset_df = pd.read_csv(csv_file_path)
    
    # Extract image info from JSON
    image_path = detection_data['image_path']
    image_height = detection_data['height']
    image_width = detection_data['width']
    detected_objects = detection_data['masks']
    
    # Extract camera type from the main image path
    main_camera_type = extract_camera_type_from_path(image_path)
    print(f"Detected camera type: {main_camera_type}")
    
    # Load the main image
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    main_image = cv2.imread(image_path)
    main_image_rgb = cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB)
    
    results = []
    
    print(f"Processing {len(detected_objects)} detected objects using polygon-constrained GMS...")
    
    for obj_idx, detected_obj in enumerate(detected_objects):
        print(f"Processing object {obj_idx + 1}/{len(detected_objects)}: {detected_obj['label']}")
        
        # Extract object ID from label (assuming format like "obj_000018")
        object_id = int(detected_obj['label'].split('_')[-1])
        
        # Create polygon mask for detected object
        detected_polygon = detected_obj['points']
        detected_mask = create_mask_from_polygon(detected_polygon, image_height, image_width)
        detected_masked_image = apply_mask_to_image(main_image_rgb, detected_mask)
        
        # Find matching objects in dataset
        matching_entries = dataset_df[
            (dataset_df['object_id'] == object_id) & 
            (dataset_df['camera_type'] == main_camera_type)
        ]
        
        if len(matching_entries) == 0:
            print(f"No matching entries found for object {object_id} with camera type '{main_camera_type}'")
            continue
        
        best_match = None
        best_similarity = -1
        best_correspondences = None
        best_dataset_masked_image = None
        
        print(f"  Found {len(matching_entries)} potential matches in dataset for camera type '{main_camera_type}'")
        
        # Compare with each matching dataset entry
        for idx, (_, dataset_entry) in enumerate(matching_entries.iterrows()):
            try:
                print(f"    Comparing with dataset entry {idx + 1}/{len(matching_entries)} (Camera: {dataset_entry['camera_type']})")
                
                # Load dataset image
                dataset_image_path = dataset_entry['image_path']
                if not Path(dataset_image_path).exists():
                    print(f"    Dataset image not found: {dataset_image_path}")
                    continue
                
                dataset_image = cv2.imread(dataset_image_path)
                dataset_image_rgb = cv2.cvtColor(dataset_image, cv2.COLOR_BGR2RGB)
                
                # Parse dataset polygon mask
                polygon_str = dataset_entry['polygon_mask']
                dataset_polygon = parse_polygon_string(polygon_str)
                
                if not dataset_polygon:
                    print(f"    Could not parse polygon: {polygon_str}")
                    continue
                
                # Create polygon mask for dataset object
                dataset_mask = create_mask_from_polygon(dataset_polygon, 
                                                      int(dataset_entry['image_height']), 
                                                      int(dataset_entry['image_width']))
                dataset_masked_image = apply_mask_to_image(dataset_image_rgb, dataset_mask)
                
                # Calculate similarity
                similarity = calculate_similarity(detected_masked_image, dataset_masked_image)
                
                # Find polygon-constrained feature correspondences using GMS
                correspondences = find_polygon_constrained_correspondences(
                    main_image_rgb, dataset_image_rgb,
                    detected_polygon, dataset_polygon,
                    min_matches=3  # Lowered minimum threshold
                )
                
                # Skip if insufficient matches
                if len(correspondences['matches']) < 3:
                    print(f"    Skipping: only {len(correspondences['matches'])} polygon-constrained matches (minimum 3 required)")
                    continue
                
                print(f"    Similarity: {similarity:.4f}, Polygon-constrained matches: {len(correspondences['matches'])}")
                
                # Combined score: prioritize polygon-constrained match quality
                if len(correspondences['matches']) >= 3:
                    match_quality_score = min(len(correspondences['matches']) / 15.0, 1.0)  # Normalize to 15 matches
                    combined_score = 0.6 * similarity + 0.4 * match_quality_score
                else:
                    combined_score = similarity * 0.3
                
                if combined_score > best_similarity:
                    best_similarity = combined_score
                    best_match = dataset_entry
                    best_correspondences = correspondences
                    best_dataset_masked_image = dataset_masked_image
                    
            except Exception as e:
                print(f"    Error processing dataset entry: {e}")
                continue
        
        if best_match is not None:
            # Extract rotation and translation from best match
            rotation_matrix = np.array([
                [best_match['r11'], best_match['r12'], best_match['r13']],
                [best_match['r21'], best_match['r22'], best_match['r23']],
                [best_match['r31'], best_match['r32'], best_match['r33']]
            ])
            
            translation_vector = np.array([best_match['tx'], best_match['ty'], best_match['tz']])
            
            result = {
                'object_index': obj_idx,
                'detected_object': detected_obj,
                'detected_polygon': detected_polygon,
                'best_match': best_match,
                'similarity_score': best_similarity,
                'correspondences': best_correspondences,
                'rotation_matrix': rotation_matrix,
                'translation_vector': translation_vector,
                'detected_masked_image': detected_masked_image,
                'matched_dataset_image': best_dataset_masked_image,
                'camera_type': main_camera_type
            }
            
            results.append(result)
            
            # Save individual masked images
            save_masked_images(result, obj_idx, output_path)
            
            # Visualize results
            visualize_match_result(result, obj_idx, output_path)
            
            # Create transformation visualization
            visualize_transformation(result, obj_idx, output_path)
            
            print(f"  Best match found with polygon-constrained score: {best_similarity:.4f}")
            print(f"    Camera type: {main_camera_type}")
            print(f"    Polygon-constrained matches: {len(best_correspondences['matches'])}")
            
        else:
            print(f"  No suitable match found for object {object_id} with camera type '{main_camera_type}'")
    
    # Create summary visualization
    if results:
        create_summary_visualization(results, output_path)
    
    return results

def parse_polygon_string(polygon_str: str) -> List[List[int]]:
    """Parse polygon string from CSV file."""
    try:
        polygon_str = polygon_str.strip()
        
        if polygon_str.startswith('[[') and polygon_str.endswith(']]'):
            return ast.literal_eval(polygon_str)
        
        if '], [' in polygon_str:
            return ast.literal_eval(polygon_str)
        
        polygon_str = re.sub(r'\s+', ' ', polygon_str)
        polygon_str = polygon_str.replace(' ]', ']').replace('[ ', '[')
        
        return ast.literal_eval(polygon_str)
        
    except Exception as e:
        print(f"Error parsing polygon string: {polygon_str}, Error: {e}")
        return []

def create_mask_from_polygon(polygon_points: List[List[int]], height: int, width: int) -> np.ndarray:
    """Create a binary mask from polygon points."""
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon_points = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon_points], 255)
    return mask

def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to image, zeroing out pixels outside the mask."""
    masked_image = image.copy()
    mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
    masked_image = (masked_image * mask_3d).astype(np.uint8)
    return masked_image

def calculate_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate similarity between two images using SSIM."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    target_size = min(300, max(h1, w1), max(h2, w2))
    
    aspect1 = w1 / h1
    aspect2 = w2 / h2
    
    if aspect1 > 1:
        target_w1, target_h1 = target_size, int(target_size / aspect1)
    else:
        target_w1, target_h1 = int(target_size * aspect1), target_size
        
    if aspect2 > 1:
        target_w2, target_h2 = target_size, int(target_size / aspect2)
    else:
        target_w2, target_h2 = int(target_size * aspect2), target_size
    
    img1_resized = cv2.resize(img1, (target_w1, target_h1))
    img2_resized = cv2.resize(img2, (target_w2, target_h2))
    
    max_h = max(target_h1, target_h2)
    max_w = max(target_w1, target_w2)
    
    img1_padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    img2_padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    
    img1_padded[:target_h1, :target_w1] = img1_resized
    img2_padded[:target_h2, :target_w2] = img2_resized
    
    gray1 = cv2.cvtColor(img1_padded, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_padded, cv2.COLOR_RGB2GRAY)
    
    similarity = ssim(gray1, gray2)
    
    return similarity

def find_polygon_constrained_correspondences(img1: np.ndarray, img2: np.ndarray, 
                                           polygon1: List[List[int]], polygon2: List[List[int]], 
                                           min_matches: int = 5) -> Dict:
    """
    Find GMS correspondences constrained to polygon regions with debugging and fallbacks.
    """
    print(f"    Polygon-constrained GMS matching for images {img1.shape} and {img2.shape}")
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1.copy()
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2.copy()
    
    # Create SIFT detector with more lenient parameters
    try:
        sift = cv2.SIFT_create(
            nfeatures=3000,           # More features
            nOctaveLayers=4,          # More octave layers
            contrastThreshold=0.02,   # Lower threshold for more features
            edgeThreshold=6,          # Lower edge threshold
            sigma=1.6
        )
    except AttributeError:
        sift = cv2.xfeatures2d.SIFT_create(
            nfeatures=3000,
            contrastThreshold=0.02,
            edgeThreshold=6
        )
    
    # Detect keypoints and compute descriptors
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)
    
    print(f"    SIFT found {len(kp1)} and {len(kp2)} total keypoints")
    
    if desc1 is None or desc2 is None:
        print(f"    No SIFT descriptors found!")
        return {'keypoints1': [], 'keypoints2': [], 'matches': []}
    
    # Debug: Check polygon validity
    print(f"    Polygon 1 has {len(polygon1)} points: {polygon1[:3]}..." if len(polygon1) > 3 else f"    Polygon 1: {polygon1}")
    print(f"    Polygon 2 has {len(polygon2)} points: {polygon2[:3]}..." if len(polygon2) > 3 else f"    Polygon 2: {polygon2}")
    
    # Try polygon-constrained filtering first
    filtered_kp1, filtered_desc1, valid_indices1 = filter_keypoints_by_polygon(kp1, desc1, polygon1, debug=True)
    filtered_kp2, filtered_desc2, valid_indices2 = filter_keypoints_by_polygon(kp2, desc2, polygon2, debug=True)
    
    print(f"    After polygon filtering: {len(filtered_kp1)} and {len(filtered_kp2)} keypoints")
    
    # Fallback 1: If polygon filtering is too strict, use expanded polygons
    if len(filtered_kp1) < 10 or len(filtered_kp2) < 10:
        print(f"    Polygon filtering too strict, trying expanded polygons...")
        expanded_poly1 = expand_polygon(polygon1, expansion=10)
        expanded_poly2 = expand_polygon(polygon2, expansion=10)
        
        filtered_kp1, filtered_desc1, valid_indices1 = filter_keypoints_by_polygon(kp1, desc1, expanded_poly1, debug=True)
        filtered_kp2, filtered_desc2, valid_indices2 = filter_keypoints_by_polygon(kp2, desc2, expanded_poly2, debug=True)
        
        print(f"    After expanded polygon filtering: {len(filtered_kp1)} and {len(filtered_kp2)} keypoints")
    
    # Fallback 2: If still too few, use bounding box approach
    if len(filtered_kp1) < 5 or len(filtered_kp2) < 5:
        print(f"    Still too few keypoints, falling back to bounding box approach...")
        bbox1 = get_polygon_bbox(polygon1)
        bbox2 = get_polygon_bbox(polygon2)
        
        filtered_kp1, filtered_desc1 = filter_keypoints_by_bbox(kp1, desc1, bbox1)
        filtered_kp2, filtered_desc2 = filter_keypoints_by_bbox(kp2, desc2, bbox2)
        
        print(f"    After bbox fallback: {len(filtered_kp1)} and {len(filtered_kp2)} keypoints")
    
    if filtered_desc1 is None or filtered_desc2 is None or len(filtered_desc1) < 3 or len(filtered_desc2) < 3:
        print(f"    Insufficient features after all fallbacks")
        return {'keypoints1': [], 'keypoints2': [], 'matches': []}
    
    # Match features with more lenient parameters
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(filtered_desc1, filtered_desc2, k=2)
    
    print(f"    BruteForce matcher found {len(matches)} potential match pairs")
    
    # Apply more lenient Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:  # More lenient ratio
                good_matches.append(m)
        elif len(match_pair) == 1:  # Handle case with only one match
            good_matches.append(match_pair[0])
    
    print(f"    After Lowe's ratio test: {len(good_matches)} initial matches")
    
    if len(good_matches) < 3:  # Lower minimum threshold
        print(f"    Insufficient initial matches for further processing")
        return {'keypoints1': filtered_kp1, 'keypoints2': filtered_kp2, 'matches': good_matches}
    
    # Apply simplified GMS filtering (less aggressive)
    if len(good_matches) >= 10:
        gms_matches = apply_simplified_gms_filter(good_matches, filtered_kp1, filtered_kp2)
        print(f"    After simplified GMS filtering: {len(gms_matches)} matches")
    else:
        gms_matches = good_matches
        print(f"    Skipping GMS (too few matches), using {len(gms_matches)} direct matches")
    
    # Sort and limit matches
    gms_matches = sorted(gms_matches, key=lambda x: x.distance)
    max_matches = min(30, len(gms_matches))
    final_matches = gms_matches[:max_matches]
    
    print(f"    Final result: {len(final_matches)} polygon-constrained matches")
    
    return {
        'keypoints1': filtered_kp1,
        'keypoints2': filtered_kp2,
        'matches': final_matches
    }

def filter_keypoints_by_polygon(keypoints, descriptors, polygon_points, debug=False):
    """Filter keypoints to only include those inside the polygon with debugging."""
    if descriptors is None:
        return [], None, []
    
    # Create polygon path for point-in-polygon test
    polygon_array = np.array(polygon_points, dtype=np.int32)
    
    valid_kp = []
    valid_desc = []
    valid_indices = []
    
    total_tested = 0
    inside_count = 0
    
    for i, kp in enumerate(keypoints):
        total_tested += 1
        point = (int(kp.pt[0]), int(kp.pt[1]))
        
        # Use cv2.pointPolygonTest to check if point is inside polygon
        # Use measureDist=False for faster computation (just inside/outside/on-edge)
        result = cv2.pointPolygonTest(polygon_array, point, False)
        
        if result >= 0:  # Inside or on boundary (>= 0 means inside/on-edge, < 0 means outside)
            inside_count += 1
            valid_kp.append(kp)
            valid_desc.append(descriptors[i])
            valid_indices.append(i)
    
    if debug:
        print(f"      Polygon filtering: {inside_count}/{total_tested} keypoints inside polygon")
        if inside_count == 0:
            # Debug polygon bounds
            poly_array = np.array(polygon_points)
            x_min, x_max = np.min(poly_array[:, 0]), np.max(poly_array[:, 0])
            y_min, y_max = np.min(poly_array[:, 1]), np.max(poly_array[:, 1])
            print(f"      Polygon bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
            
            # Debug some keypoint positions
            if len(keypoints) > 0:
                sample_kp = keypoints[:min(5, len(keypoints))]
                sample_points = [(int(kp.pt[0]), int(kp.pt[1])) for kp in sample_kp]
                print(f"      Sample keypoint positions: {sample_points}")
    
    if len(valid_desc) > 0:
        valid_desc_array = np.array(valid_desc)
    else:
        valid_desc_array = None
    
    return valid_kp, valid_desc_array, valid_indices

def expand_polygon(polygon_points, expansion=10):
    """Expand polygon outward by a given number of pixels."""
    if not polygon_points:
        return polygon_points
    
    # Convert to numpy array
    points = np.array(polygon_points, dtype=np.float32)
    
    # Calculate centroid
    centroid = np.mean(points, axis=0)
    
    # Expand each point away from centroid
    expanded_points = []
    for point in points:
        direction = point - centroid
        if np.linalg.norm(direction) > 0:
            direction_normalized = direction / np.linalg.norm(direction)
            expanded_point = point + direction_normalized * expansion
            expanded_points.append([int(expanded_point[0]), int(expanded_point[1])])
        else:
            expanded_points.append([int(point[0]), int(point[1])])
    
    return expanded_points

def get_polygon_bbox(polygon_points):
    """Get bounding box from polygon points."""
    if not polygon_points:
        return [0, 0, 100, 100]  # Default bbox
    
    points = np.array(polygon_points)
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
    
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_min, y_min, width, height]

def filter_keypoints_by_bbox(keypoints, descriptors, bbox):
    """Filter keypoints to only include those inside the bounding box."""
    if descriptors is None:
        return [], None
    
    x, y, w, h = bbox
    x_min, y_min = x, y
    x_max, y_max = x + w, y + h
    
    valid_kp = []
    valid_desc = []
    
    for i, kp in enumerate(keypoints):
        px, py = kp.pt
        if x_min <= px <= x_max and y_min <= py <= y_max:
            valid_kp.append(kp)
            valid_desc.append(descriptors[i])
    
    if len(valid_desc) > 0:
        valid_desc_array = np.array(valid_desc)
    else:
        valid_desc_array = None
    
    return valid_kp, valid_desc_array

def apply_simplified_gms_filter(matches, kp1, kp2, grid_size=25, threshold_factor=4.0):
    """Apply simplified GMS filtering - less aggressive than full GMS."""
    if len(matches) < 10:
        return matches
    
    # Calculate motion vectors
    motions = []
    for match in matches:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        motion = (pt2[0] - pt1[0], pt2[1] - pt1[1])
        motions.append(motion)
    
    if len(motions) < 5:
        return matches
    
    # Calculate global motion statistics
    motions_array = np.array(motions)
    median_motion = np.median(motions_array, axis=0)
    motion_std = np.std(motions_array, axis=0)
    
    # Filter by motion consistency (simplified)
    consistent_matches = []
    for i, (match, motion) in enumerate(zip(matches, motions)):
        motion_diff = np.linalg.norm(np.array(motion) - median_motion)
        motion_threshold = threshold_factor * np.linalg.norm(motion_std + 1e-6)
        
        if motion_diff < motion_threshold:
            consistent_matches.append(match)
    
    print(f"      Simplified GMS filtering: {len(consistent_matches)}/{len(matches)} consistent matches")
    
    return consistent_matches

def apply_gms_filter(matches, kp1, kp2, img1_shape, img2_shape, grid_size=20, threshold_factor=6.0):
    """Apply GMS filtering for robust matching."""
    if len(matches) < 15:
        return matches
    
    h1, w1 = img1_shape
    h2, w2 = img2_shape
    
    # Create grids
    grid_rows1 = max(4, h1 // grid_size)
    grid_cols1 = max(4, w1 // grid_size)
    
    # Group matches by grid cells
    grid_matches = {}
    match_motions = []
    
    for match in matches:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        
        grid_row1 = min(int(pt1[1] / grid_size), grid_rows1 - 1)
        grid_col1 = min(int(pt1[0] / grid_size), grid_cols1 - 1)
        grid_key = (grid_row1, grid_col1)
        
        motion = (pt2[0] - pt1[0], pt2[1] - pt1[1])
        match_motions.append(motion)
        
        if grid_key not in grid_matches:
            grid_matches[grid_key] = []
        grid_matches[grid_key].append((match, pt1, pt2, motion))
    
    if len(match_motions) < 10:
        return matches
    
    # Calculate global motion statistics
    motions_array = np.array(match_motions)
    median_motion = np.median(motions_array, axis=0)
    motion_std = np.std(motions_array, axis=0)
    
    # Filter by global motion consistency
    globally_consistent_matches = []
    for i, (match, motion) in enumerate(zip(matches, match_motions)):
        motion_diff = np.linalg.norm(np.array(motion) - median_motion)
        motion_threshold = threshold_factor * np.linalg.norm(motion_std)
        
        if motion_diff < motion_threshold:
            globally_consistent_matches.append(match)
    
    print(f"      Global motion filtering: {len(globally_consistent_matches)}/{len(matches)} matches")
    
    return globally_consistent_matches

def consensus_based_transformation(matches, kp1, kp2, img_shape):
    """
    Smart consensus-based transformation estimation.
    Try different subsets of matches and pick the one with best consensus.
    """
    if len(matches) < 4:  # Lowered minimum requirement
        return None, f"Insufficient matches: {len(matches)}/4"
    
    print(f"      Consensus transformation with {len(matches)} matches")
    
    best_homography = None
    best_consensus_score = 0
    best_info = ""
    
    # Try different subset sizes (adapted for smaller match counts)
    subset_sizes = [min(6, len(matches)), min(8, len(matches)), min(len(matches), 15)]
    subset_sizes = [s for s in subset_sizes if s >= 4]  # Ensure minimum 4 matches
    
    for subset_size in subset_sizes:
        if subset_size < 4:
            continue
            
        print(f"      Trying subset size: {subset_size}")
        
        # Sort matches by quality and take best subset
        sorted_matches = sorted(matches, key=lambda x: x.distance)
        subset_matches = sorted_matches[:subset_size]
        
        # Extract points
        src_pts = np.float32([kp2[m.trainIdx].pt for m in subset_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in subset_matches]).reshape(-1, 1, 2)
        
        try:
            # Compute homography
            homography, mask = cv2.findHomography(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,  # Slightly more lenient
                confidence=0.95,            # More lenient confidence
                maxIters=1000               # Fewer iterations for speed
            )
            
            if homography is None:
                continue
            
            # Test consensus: how many of ALL matches agree with this homography?
            all_src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            all_dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Transform all source points
            transformed_pts = cv2.perspectiveTransform(all_src_pts, homography)
            
            # Calculate reprojection errors for all matches
            errors = np.linalg.norm(transformed_pts.reshape(-1, 2) - all_dst_pts.reshape(-1, 2), axis=1)
            
            # Count inliers (error < 4.0 pixels) - more lenient threshold
            inliers = np.sum(errors < 4.0)
            consensus_score = inliers
            
            print(f"        Subset {subset_size}: {inliers}/{len(matches)} consensus inliers")
            
            if consensus_score > best_consensus_score:
                best_homography = homography
                best_consensus_score = consensus_score
                best_info = f"Consensus transformation SUCCESS\nSubset size: {subset_size}\nConsensus inliers: {inliers}/{len(matches)}\nReprojection threshold: 4.0px"
                
        except Exception as e:
            print(f"        Subset {subset_size}: Error - {str(e)[:50]}")
            continue
    
    if best_homography is not None and best_consensus_score >= 3:  # Lowered minimum consensus
        print(f"      Consensus transformation: SUCCESS with {best_consensus_score} inliers")
        return best_homography, best_info
    else:
        return None, f"Consensus transformation: FAILED\nBest consensus: {best_consensus_score}/3 required"

def visualize_transformation(result: Dict, obj_idx: int, output_dir: Path):
    """Visualize the transformation with consensus-based approach."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Object {obj_idx + 1} Polygon-Constrained Transformation - {result["detected_object"]["label"]} (Camera: {result["camera_type"]})', fontsize=16)
    
    test_image = result['detected_masked_image']
    ref_image = result['matched_dataset_image']
    correspondences = result['correspondences']
    
    # Display test image (target)
    axes[0].imshow(test_image)
    axes[0].set_title('Test Image (Target)\n(Polygon-Constrained Region)')
    axes[0].axis('off')
    
    # Display reference image (source)
    axes[1].imshow(ref_image)
    axes[1].set_title('Reference Image (Source)\n(Polygon-Constrained Match)')
    axes[1].axis('off')
    
    # Compute consensus-based transformation
    transformed_ref_image = None
    transformation_info = "No transformation computed"
    
    if len(correspondences['matches']) >= 4:  # Lowered minimum requirement
        try:
            homography, info = consensus_based_transformation(
                correspondences['matches'], 
                correspondences['keypoints1'], 
                correspondences['keypoints2'],
                test_image.shape[:2]
            )
            
            if homography is not None:
                # Apply transformation
                h_test, w_test = test_image.shape[:2]
                transformed_ref_image = cv2.warpPerspective(ref_image, homography, (w_test, h_test))
                
                # Compute quality metrics
                transformation_quality = compute_transformation_quality(test_image, transformed_ref_image)
                transformation_info = info + f"\nSSIM: {transformation_quality['ssim']:.3f}\nMSE: {transformation_quality['mse']:.1f}"
                
            else:
                transformation_info = info
                
        except Exception as e:
            transformation_info = f"Consensus transformation error: {str(e)[:50]}..."
            print(f"    Consensus transformation error: {e}")
    else:
        transformation_info = f"Insufficient matches: {len(correspondences['matches'])}/4"
    
    # Display result
    if transformed_ref_image is not None:
        axes[2].imshow(transformed_ref_image)
        axes[2].set_title(f'Consensus Transformed Reference\n{transformation_info}')
    else:
        placeholder = np.zeros_like(test_image)
        axes[2].imshow(placeholder)
        axes[2].set_title(f'Consensus Transformation Failed\n{transformation_info}')
        axes[2].text(0.5, 0.5, 'Consensus Transformation\nNot Available', 
                    ha='center', va='center', transform=axes[2].transAxes, 
                    fontsize=14, color='white',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
    
    axes[2].axis('off')
    
    # Add keypoint overlays
    if len(correspondences['matches']) > 0:
        for i, match in enumerate(correspondences['matches'][:15]):  # Show up to 15 points
            pt1 = correspondences['keypoints1'][match.queryIdx].pt
            pt2 = correspondences['keypoints2'][match.trainIdx].pt
            color = plt.cm.viridis(i / max(len(correspondences['matches'][:15]) - 1, 1))[:3]
            
            axes[0].plot(pt1[0], pt1[1], 'o', color=color, markersize=4, 
                        markeredgecolor='white', markeredgewidth=1)
            axes[1].plot(pt2[0], pt2[1], 'o', color=color, markersize=4, 
                        markeredgecolor='white', markeredgewidth=1)
    
    # Add detailed information
    pose_info = f"""POLYGON-CONSTRAINED POSE ESTIMATION:
Detected polygon points: {len(result['detected_polygon'])}
Dataset polygon match: polygon-based

Rotation Matrix:
[{result['rotation_matrix'][0,0]:.3f} {result['rotation_matrix'][0,1]:.3f} {result['rotation_matrix'][0,2]:.3f}]
[{result['rotation_matrix'][1,0]:.3f} {result['rotation_matrix'][1,1]:.3f} {result['rotation_matrix'][1,2]:.3f}]
[{result['rotation_matrix'][2,0]:.3f} {result['rotation_matrix'][2,1]:.3f} {result['rotation_matrix'][2,2]:.3f}]

Translation Vector:
[{result['translation_vector'][0]:.2f}, {result['translation_vector'][1]:.2f}, {result['translation_vector'][2]:.2f}]

SMART CORRESPONDENCES:
Total polygon-constrained matches: {len(correspondences['matches'])}
Keypoints shown: {min(15, len(correspondences['matches']))}

CONSENSUS TRANSFORMATION METHOD:
Masking: Exact Polygon (not bounding box)
Algorithm: Consensus-based transformation
Features: Only features INSIDE polygon regions
Multiple subset testing for best consensus
Purpose: Robust transformation with polygon precision
Note: Full 3D pose available in rotation/translation
Smart: Ignores background, matches only object regions
"""
    
    # Add info text
    fig.text(0.02, 0.02, pose_info, fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8),
             verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    # Save visualization
    output_filename = f"object_{obj_idx + 1:02d}_{result['detected_object']['label']}_polygon_consensus_transformation.png"
    output_filepath = output_dir / output_filename
    plt.savefig(output_filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved polygon-consensus transformation: {output_filepath}")

def compute_transformation_quality(test_image: np.ndarray, transformed_ref_image: np.ndarray) -> Dict[str, float]:
    """Compute quality metrics for the transformation."""
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
    transformed_gray = cv2.cvtColor(transformed_ref_image, cv2.COLOR_RGB2GRAY)
    
    test_mask = (test_gray > 0).astype(np.uint8)
    transformed_mask = (transformed_gray > 0).astype(np.uint8)
    overlap_mask = test_mask & transformed_mask
    
    if np.sum(overlap_mask) > 0:
        ssim_score = ssim(test_gray, transformed_gray, data_range=255)
        overlap_indices = np.where(overlap_mask)
        test_pixels = test_gray[overlap_indices]
        transformed_pixels = transformed_gray[overlap_indices]
        mse_score = np.mean((test_pixels.astype(float) - transformed_pixels.astype(float)) ** 2)
    else:
        ssim_score = 0.0
        mse_score = float('inf')
    
    return {
        'ssim': ssim_score,
        'mse': mse_score,
        'overlap_ratio': np.sum(overlap_mask) / max(np.sum(test_mask), 1)
    }

def save_masked_images(result: Dict, obj_idx: int, output_dir: Path):
    """Save individual masked images for inspection."""
    masked_dir = output_dir / "masked_images"
    masked_dir.mkdir(exist_ok=True)
    
    # Save detected object
    detected_filename = f"object_{obj_idx + 1:02d}_{result['detected_object']['label']}_detected_polygon.png"
    detected_filepath = masked_dir / detected_filename
    plt.figure(figsize=(8, 6))
    plt.imshow(result['detected_masked_image'])
    plt.title(f'Detected Object: {result["detected_object"]["label"]} (Camera: {result["camera_type"]})\nPolygon-Constrained')
    plt.axis('off')
    plt.savefig(detected_filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save matched dataset image
    matched_filename = f"object_{obj_idx + 1:02d}_{result['detected_object']['label']}_matched_polygon.png"
    matched_filepath = masked_dir / matched_filename
    plt.figure(figsize=(8, 6))
    plt.imshow(result['matched_dataset_image'])
    plt.title(f'Matched Dataset Image (Camera: {result["camera_type"]}, Score: {result["similarity_score"]:.3f}, Matches: {len(result["correspondences"]["matches"])})')
    plt.axis('off')
    plt.savefig(matched_filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    Saved polygon-constrained images: {detected_filepath} and {matched_filepath}")

def visualize_match_result(result: Dict, obj_idx: int, output_dir: Path):
    """Visualize the matching result."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Object {obj_idx + 1} Polygon-Constrained Matching Results - {result["detected_object"]["label"]} (Camera: {result["camera_type"]})', fontsize=16)
    
    # Original detected object
    axes[0, 0].imshow(result['detected_masked_image'])
    axes[0, 0].set_title(f'Detected Object (Polygon Masked) - {result["camera_type"]}')
    axes[0, 0].axis('off')
    
    # Best matched dataset image
    axes[0, 1].imshow(result['matched_dataset_image'])
    axes[0, 1].set_title(f'Best Match (Score: {result["similarity_score"]:.3f}) - {result["camera_type"]}')
    axes[0, 1].axis('off')
    
    # Feature correspondences visualization
    correspondences = result['correspondences']
    if len(correspondences['matches']) > 0:
        img1 = result['detected_masked_image']
        img2 = result['matched_dataset_image']
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        target_h = 300
        scale1 = target_h / h1
        scale2 = target_h / h2
        
        img1_resized = cv2.resize(img1, (int(w1 * scale1), target_h))
        img2_resized = cv2.resize(img2, (int(w2 * scale2), target_h))
        
        combined_img = np.hstack([img1_resized, img2_resized])
        axes[1, 0].imshow(combined_img)
        
        offset_x = img1_resized.shape[1]
        num_matches_to_show = min(15, len(correspondences['matches']))
        
        for i, match in enumerate(correspondences['matches'][:num_matches_to_show]):
            pt1 = correspondences['keypoints1'][match.queryIdx].pt
            pt2 = correspondences['keypoints2'][match.trainIdx].pt
            
            x1, y1 = int(pt1[0] * scale1), int(pt1[1] * scale1)
            x2, y2 = int(pt2[0] * scale2 + offset_x), int(pt2[1] * scale2)
            
            color = plt.cm.plasma(i / max(num_matches_to_show - 1, 1))[:3]
            
            axes[1, 0].plot([x1, x2], [y1, y2], color=color, linewidth=2.0, alpha=0.9)
            axes[1, 0].plot(x1, y1, 'o', color=color, markersize=5, markeredgecolor='white', markeredgewidth=1.5)
            axes[1, 0].plot(x2, y2, 's', color=color, markersize=5, markeredgecolor='white', markeredgewidth=1.5)
    else:
        axes[1, 0].text(0.5, 0.5, 'No correspondences found', 
                       ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=14)
    
    axes[1, 0].set_title(f'Polygon-Constrained Correspondences ({len(correspondences["matches"])} matches)')
    axes[1, 0].axis('off')
    
    # Rotation matrix visualization
    rotation_matrix = result['rotation_matrix']
    im = axes[1, 1].imshow(rotation_matrix, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 1].set_title('Rotation Matrix')
    axes[1, 1].set_xticks(range(3))
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_xticklabels(['X', 'Y', 'Z'])
    axes[1, 1].set_yticklabels(['X', 'Y', 'Z'])
    
    for i in range(3):
        for j in range(3):
            axes[1, 1].text(j, i, f'{rotation_matrix[i, j]:.3f}', 
                           ha='center', va='center', fontweight='bold',
                           color='white' if abs(rotation_matrix[i, j]) > 0.5 else 'black')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save the figure
    output_filename = f"object_{obj_idx + 1:02d}_{result['detected_object']['label']}_polygon_match_result.png"
    output_filepath = output_dir / output_filename
    plt.savefig(output_filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved polygon matching visualization: {output_filepath}")

def create_summary_visualization(results: List[Dict], output_dir: Path):
    """Create a summary visualization of all results."""
    n_results = len(results)
    fig, axes = plt.subplots(2, n_results, figsize=(5*n_results, 10))
    if n_results == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Summary: Polygon-Constrained GMS Matching with Consensus Transformation', fontsize=16)
    
    similarities = []
    match_counts = []
    camera_types = []
    
    for i, result in enumerate(results):
        # Top row: detected objects
        axes[0, i].imshow(result['detected_masked_image'])
        axes[0, i].set_title(f"Object {i+1}\n{result['detected_object']['label']}\n({result['camera_type']})")
        axes[0, i].axis('off')
        
        # Bottom row: matched objects
        axes[1, i].imshow(result['matched_dataset_image'])
        similarity = result['similarity_score']
        n_matches = len(result['correspondences']['matches'])
        axes[1, i].set_title(f"Best Match\nScore: {similarity:.3f}\nMatches: {n_matches}\n({result['camera_type']})")
        axes[1, i].axis('off')
        
        similarities.append(similarity)
        match_counts.append(n_matches)
        camera_types.append(result['camera_type'])
    
    plt.tight_layout()
    
    # Save summary
    summary_filepath = output_dir / "summary_polygon_consensus_matches.png"
    plt.savefig(summary_filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved polygon-consensus summary: {summary_filepath}")
    
    # Statistics
    print("\n" + "="*70)
    print("POLYGON-CONSTRAINED GMS WITH CONSENSUS TRANSFORMATION SUMMARY")
    print("="*70)
    print(f"Total objects processed: {n_results}")
    print(f"Camera types found: {set(camera_types)}")
    print(f"Average score: {np.mean(similarities):.4f}")
    print(f"Average polygon-constrained matches: {np.mean(match_counts):.1f}")
    print(f"Method: Polygon-constrained GMS + Consensus transformation")
    print("="*70)

def test_with_sample_data():
    """Test function with sample data structure."""
    print("Testing polygon-constrained GMS matching with consensus-based transformation...")
    print("Key insight: Only match features INSIDE polygon regions, ignore background")
    print("DEBUG MODE: Extensive logging and multiple fallback strategies")
    
    # This would be called with actual file paths:
    results = object_matching_and_pose_estimation("/home/rama/bpc_ws/bpc/maskRCNN/results/000000_annotation.json", "/home/rama/bpc_ws/bpc/utilities/filtered_dataset.csv")
    
    print("Key features:")
    print("- POLYGON-CONSTRAINED: Features only from INSIDE polygon regions")
    print("- BACKGROUND IGNORED: No background noise in matching")
    print("- EXTENSIVE DEBUGGING: Detailed logging at each step")
    print("- MULTIPLE FALLBACKS:")
    print("  * Expanded polygons if too strict")
    print("  * Bounding box fallback if polygons fail")
    print("  * More lenient SIFT parameters")
    print("  * Lower minimum match requirements")
    print("- CONSENSUS TRANSFORMATION: Multiple subset testing for robustness")
    print("- SMART APPROACH: Simple but focused on actual object regions")
    print("- SIMPLIFIED GMS: Less aggressive filtering")
    print("- MINIMUM MATCHES: 3-4 polygon-constrained correspondences")
    print("- LENIENT PARAMETERS: More forgiving thresholds")

if __name__ == "__main__":
    test_with_sample_data()