import cv2
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import ast
import re
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

print("Using PnP Pose Estimation with Homography RANSAC Outlier Elimination")

def extract_camera_type_from_path(image_path: str) -> str:
    """Extract camera type from image path."""
    path_obj = Path(image_path)
    filename = path_obj.stem
    
    camera_patterns = [
        r'_cam(\d+)', r'_photoneo', r'_kinect', r'_realsense', r'_zed', r'_([a-zA-Z]+\d*)'
    ]
    
    for pattern in camera_patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            if 'cam' in pattern:
                return f"cam{match.group(1)}"
            elif any(cam in pattern for cam in ['photoneo', 'kinect', 'realsense', 'zed']):
                return pattern.replace('_', '').replace('(', '').replace(')', '')
            else:
                return match.group(1)
    
    # Check directory structure
    parts = path_obj.parts
    for part in reversed(parts):
        for pattern in camera_patterns:
            match = re.search(pattern.replace('_', ''), part, re.IGNORECASE)
            if match:
                if 'cam' in pattern:
                    return f"cam{match.group(1)}"
                return match.group(0)
    
    print(f"Warning: Could not extract camera type from path: {image_path}")
    return "unknown"

def get_camera_intrinsics(camera_type: str) -> np.ndarray:
    """
    Get camera intrinsic parameters for PnP pose estimation.
    Adjust these parameters based on your actual camera calibrations.
    """
    camera_params = {
        'cam1': np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]]),
        'cam2': np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]]),
        'photoneo': np.array([[1000.0, 0, 512.0], [0, 1000.0, 384.0], [0, 0, 1.0]]),
        'kinect': np.array([[525.0, 0, 319.5], [0, 525.0, 239.5], [0, 0, 1.0]]),
        'realsense': np.array([[615.0, 0, 320.0], [0, 615.0, 240.0], [0, 0, 1.0]]),
        'default': np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]])
    }
    return camera_params.get(camera_type, camera_params['default'])

def extract_sift_features(image: np.ndarray) -> Tuple[List, np.ndarray]:
    """Extract SIFT features from image with enhanced parameters for masked objects."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply mild Gaussian blur to reduce noise and improve feature detection
    gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
    
    # Create SIFT detector with relaxed parameters for better feature extraction
    sift = cv2.SIFT_create(
        nfeatures=3000,           # Increased from 1500
        nOctaveLayers=5,          # Increased from 4 for better scale coverage
        contrastThreshold=0.02,   # Lowered from 0.03 for more features
        edgeThreshold=12,         # Increased from 8 to reduce edge filtering
        sigma=1.2                 # Reduced for sharper features
    )
    
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # If still too few features, try with even more relaxed parameters
    if len(keypoints) < 50:
        print(f"    Low feature count ({len(keypoints)}), trying relaxed parameters...")
        sift_relaxed = cv2.SIFT_create(
            nfeatures=5000,
            nOctaveLayers=6,
            contrastThreshold=0.01,   # Very low threshold
            edgeThreshold=20,         # Very high edge threshold
            sigma=1.0
        )
        keypoints, descriptors = sift_relaxed.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def match_features_with_ratio_test(desc1: np.ndarray, desc2: np.ndarray, ratio_threshold: float = 0.8):
    """Match SIFT features using relaxed ratio test for masked objects."""
    if desc1 is None or desc2 is None:
        return []
    
    # BruteForce matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Apply Lowe's ratio test with relaxed threshold
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
        elif len(match_pair) == 1:
            # Include single matches if they're reasonably good
            if match_pair[0].distance < 200:  # Reasonable descriptor distance
                good_matches.append(match_pair[0])
    
    # If still too few matches, try even more relaxed approach
    if len(good_matches) < 10:
        print(f"    Low match count ({len(good_matches)}), trying very relaxed matching...")
        # Try with even higher ratio threshold
        relaxed_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.9 * n.distance:  # Very relaxed
                    relaxed_matches.append(m)
            elif len(match_pair) == 1:
                if match_pair[0].distance < 250:
                    relaxed_matches.append(match_pair[0])
        
        if len(relaxed_matches) > len(good_matches):
            good_matches = relaxed_matches
    
    return good_matches

def filter_matches_with_homography_ransac(kp1, kp2, matches, ransac_threshold: float = 3.0) -> Tuple[List, np.ndarray, List]:
    """
    Filter matches using homography estimation with RANSAC.
    This is the core outlier elimination step.
    """
    if len(matches) < 8:  # Minimum points for homography
        return [], None, []
    
    # Extract corresponding points
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    try:
        # Compute homography using RANSAC
        homography, mask = cv2.findHomography(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            confidence=0.995,
            maxIters=5000
        )
        
        if homography is not None and mask is not None:
            # Filter inlier matches
            inlier_matches = [matches[i] for i in range(len(matches)) if mask[i][0] == 1]
            inlier_indices = [i for i in range(len(matches)) if mask[i][0] == 1]
            
            return inlier_matches, homography, inlier_indices
        else:
            return [], None, []
            
    except Exception as e:
        print(f"    Homography RANSAC failed: {e}")
        return [], None, []

def estimate_pose_with_pnp(kp1, kp2, inlier_matches, camera_matrix: np.ndarray) -> Dict:
    """
    Estimate 3D pose using PnP (Perspective-n-Point) algorithm.
    This is the main pose estimation function.
    """
    if len(inlier_matches) < 6:  # Minimum points for reliable PnP
        return {'success': False, 'reason': f'Insufficient inlier matches: {len(inlier_matches)}'}
    
    # Extract 2D image points (detected object)
    image_points = np.array([kp1[m.queryIdx].pt for m in inlier_matches], dtype=np.float32)
    
    # Generate 3D object points (assuming planar object at z=0)
    # Scale reference points to reasonable physical coordinates (e.g., centimeters)
    reference_points_2d = np.array([kp2[m.trainIdx].pt for m in inlier_matches], dtype=np.float32)
    object_points_3d = np.zeros((len(reference_points_2d), 3), dtype=np.float32)
    object_points_3d[:, 0] = (reference_points_2d[:, 0] - 320) / 1000.0  # Convert to meters, center around origin
    object_points_3d[:, 1] = (reference_points_2d[:, 1] - 240) / 1000.0  # Convert to meters, center around origin
    # Z coordinates remain 0 (planar assumption)
    
    # Distortion coefficients (assuming minimal distortion)
    dist_coeffs = np.zeros((4, 1))
    
    try:
        # Solve PnP using RANSAC for additional robustness
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points_3d, image_points, camera_matrix, dist_coeffs,
            iterationsCount=2000,
            reprojectionError=5.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success and inliers is not None and len(inliers) >= 4:
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Calculate reprojection error
            projected_points, _ = cv2.projectPoints(
                object_points_3d[inliers.flatten()], 
                rvec, tvec, camera_matrix, dist_coeffs
            )
            projected_points = projected_points.reshape(-1, 2)
            inlier_image_points = image_points[inliers.flatten()]
            
            reprojection_errors = np.linalg.norm(projected_points - inlier_image_points, axis=1)
            mean_reprojection_error = np.mean(reprojection_errors)
            max_reprojection_error = np.max(reprojection_errors)
            
            return {
                'success': True,
                'rotation_matrix': rotation_matrix,
                'translation_vector': tvec.flatten(),
                'rotation_vector': rvec.flatten(),
                'pnp_inliers': inliers.flatten(),
                'pnp_inlier_count': len(inliers),
                'total_matches': len(inlier_matches),
                'pnp_inlier_ratio': len(inliers) / len(inlier_matches),
                'mean_reprojection_error': mean_reprojection_error,
                'max_reprojection_error': max_reprojection_error,
                'object_points_3d': object_points_3d,
                'image_points': image_points,
                'projected_points': projected_points
            }
        else:
            return {'success': False, 'reason': 'PnP failed or insufficient inliers'}
            
    except Exception as e:
        return {'success': False, 'reason': f'PnP error: {str(e)}'}

def calculate_pose_quality_score(pose_result: Dict, homography_inlier_count: int) -> float:
    """
    Calculate overall pose estimation quality score.
    """
    if not pose_result['success']:
        return 0.0
    
    # Factors contributing to pose quality:
    # 1. PnP inlier ratio (higher is better)
    pnp_quality = pose_result['pnp_inlier_ratio']
    
    # 2. Reprojection error (lower is better)
    reprojection_quality = max(0, 1.0 - pose_result['mean_reprojection_error'] / 10.0)
    
    # 3. Number of matches (more is better, up to a point)
    match_quality = min(homography_inlier_count / 20.0, 1.0)
    
    # 4. PnP inlier count (absolute number matters)
    inlier_count_quality = min(pose_result['pnp_inlier_count'] / 15.0, 1.0)
    
    # Weighted combination
    total_score = (0.3 * pnp_quality + 
                   0.3 * reprojection_quality + 
                   0.2 * match_quality + 
                   0.2 * inlier_count_quality)
    
    return min(total_score, 1.0)

def object_matching_and_pose_estimation(json_file_path: str, csv_file_path: str, output_dir: str = "output") -> List[Dict]:
    """
    Main function: Object matching and pose estimation using PnP with homography RANSAC.
    
    Pipeline:
    1. Extract SIFT features from detected and dataset objects
    2. Match features using ratio test
    3. Filter matches using homography RANSAC (outlier elimination)
    4. Estimate 3D pose using PnP algorithm
    5. Evaluate pose quality and select best match
    
    Args:
        json_file_path: Path to object detection JSON file
        csv_file_path: Path to dataset CSV file
        output_dir: Output directory for visualizations
        
    Returns:
        List of results with pose estimation data
    """
    
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path.absolute()}")
    
    # Load data
    with open(json_file_path, 'r') as f:
        detection_data = json.load(f)
    dataset_df = pd.read_csv(csv_file_path)
    
    # Extract image info
    image_path = detection_data['image_path']
    image_height = detection_data['height']
    image_width = detection_data['width']
    detected_objects = detection_data['masks']
    
    # Get camera info
    main_camera_type = extract_camera_type_from_path(image_path)
    camera_matrix = get_camera_intrinsics(main_camera_type)
    print(f"Camera: {main_camera_type}")
    print(f"Camera matrix:\n{camera_matrix}")
    
    # Load main image
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    main_image = cv2.imread(image_path)
    main_image_rgb = cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB)
    
    results = []
    
    print(f"\nProcessing {len(detected_objects)} detected objects...")
    
    for obj_idx, detected_obj in enumerate(detected_objects):
        print(f"\n--- Object {obj_idx + 1}/{len(detected_objects)}: {detected_obj['label']} ---")
        
        # Extract object ID
        object_id = int(detected_obj['label'].split('_')[-1])
        
        # Create detected object mask and extract masked image
        detected_mask = create_mask_from_polygon(detected_obj['points'], image_height, image_width)
        detected_masked_image = apply_mask_to_image(main_image_rgb, detected_mask)
        
        # Find matching dataset entries (same camera type)
        matching_entries = dataset_df[
            (dataset_df['object_id'] == object_id) & 
            (dataset_df['camera_type'] == main_camera_type)
        ]
        
        if len(matching_entries) == 0:
            print(f"No matching entries for object {object_id} with camera {main_camera_type}")
            continue
        
        print(f"Found {len(matching_entries)} potential matches")
        
        # Extract SIFT features from detected object
        kp1, desc1 = extract_sift_features(detected_masked_image)
        print(f"Detected object features: {len(kp1)} keypoints")
        
        if len(kp1) < 8:
            print("Insufficient features in detected object")
            continue
        
        best_result = None
        best_pose_score = -1
        
        # Compare with each dataset entry
        for idx, (_, dataset_entry) in enumerate(matching_entries.iterrows()):
            print(f"  Dataset entry {idx + 1}/{len(matching_entries)}")
            
            try:
                # Load and process dataset image
                dataset_image_path = dataset_entry['image_path']
                if not Path(dataset_image_path).exists():
                    print(f"    Dataset image not found: {dataset_image_path}")
                    continue
                
                dataset_image = cv2.imread(dataset_image_path)
                dataset_image_rgb = cv2.cvtColor(dataset_image, cv2.COLOR_BGR2RGB)
                
                # Parse dataset polygon and create mask
                polygon_str = dataset_entry['polygon_mask']
                dataset_polygon = parse_polygon_string(polygon_str)
                if not dataset_polygon:
                    print("    Could not parse polygon")
                    continue
                
                dataset_mask = create_mask_from_polygon(
                    dataset_polygon, 
                    int(dataset_entry['image_height']), 
                    int(dataset_entry['image_width'])
                )
                dataset_masked_image = apply_mask_to_image(dataset_image_rgb, dataset_mask)
                
                # Extract SIFT features from dataset object
                kp2, desc2 = extract_sift_features(dataset_masked_image)
                print(f"    Dataset features: {len(kp2)} keypoints")
                
                if len(kp2) < 8:
                    print("    Insufficient features in dataset object")
                    continue
                
                # Step 1: Match features with ratio test
                initial_matches = match_features_with_ratio_test(desc1, desc2)
                print(f"    Initial matches: {len(initial_matches)}")
                
                if len(initial_matches) < 8:
                    print("    Insufficient initial matches")
                    continue
                
                # Step 2: Filter matches using homography RANSAC (outlier elimination)
                inlier_matches, homography, inlier_indices = filter_matches_with_homography_ransac(
                    kp1, kp2, initial_matches
                )
                print(f"    Homography inliers: {len(inlier_matches)}/{len(initial_matches)}")
                
                if len(inlier_matches) < 6:
                    print("    Insufficient homography inliers")
                    continue
                
                # Step 3: Estimate pose using PnP
                pose_result = estimate_pose_with_pnp(kp1, kp2, inlier_matches, camera_matrix)
                
                if pose_result['success']:
                    print(f"    PnP success: {pose_result['pnp_inlier_count']}/{pose_result['total_matches']} inliers")
                    print(f"    Reprojection error: {pose_result['mean_reprojection_error']:.2f} pixels")
                    
                    # Calculate pose quality score
                    pose_score = calculate_pose_quality_score(pose_result, len(inlier_matches))
                    print(f"    Pose quality score: {pose_score:.3f}")
                    
                    if pose_score > best_pose_score:
                        best_pose_score = pose_score
                        best_result = {
                            'dataset_entry': dataset_entry,
                            'dataset_masked_image': dataset_masked_image,
                            'initial_matches': initial_matches,
                            'inlier_matches': inlier_matches,
                            'homography': homography,
                            'pose_result': pose_result,
                            'pose_score': pose_score,
                            'keypoints1': kp1,
                            'keypoints2': kp2
                        }
                else:
                    print(f"    PnP failed: {pose_result['reason']}")
                    
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        # Store result if we found a good match
        if best_result is not None:
            result = {
                'object_index': obj_idx,
                'detected_object': detected_obj,
                'detected_masked_image': detected_masked_image,
                'camera_type': main_camera_type,
                'camera_matrix': camera_matrix,
                **best_result
            }
            
            results.append(result)
            
            # Create visualizations
            save_pose_visualization(result, obj_idx, output_path)
            save_detailed_analysis(result, obj_idx, output_path)
            
            print(f"✓ Best match found - Pose score: {best_pose_score:.3f}")
            
        else:
            print("✗ No suitable match found")
    
    # Create summary
    if results:
        create_pose_summary(results, output_path)
    
    return results

def parse_polygon_string(polygon_str: str) -> List[List[int]]:
    """Parse polygon string from CSV."""
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
        print(f"Error parsing polygon: {e}")
        return []

def create_mask_from_polygon(polygon_points: List[List[int]], height: int, width: int) -> np.ndarray:
    """Create binary mask from polygon points."""
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon_points = np.array(polygon_points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon_points], 255)
    return mask

def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to image."""
    masked_image = image.copy()
    mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
    masked_image = (masked_image * mask_3d).astype(np.uint8)
    return masked_image

def save_pose_visualization(result: Dict, obj_idx: int, output_dir: Path):
    """Save comprehensive pose estimation visualization."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
    
    pose_result = result['pose_result']
    obj_label = result['detected_object']['label']
    
    fig.suptitle(f'PnP Pose Estimation Results - Object {obj_idx + 1}: {obj_label}', fontsize=16, fontweight='bold')
    
    # Row 1: Images and correspondences
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(result['detected_masked_image'])
    ax1.set_title(f'Detected Object\n{len(result["keypoints1"])} SIFT features')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(result['dataset_masked_image'])
    ax2.set_title(f'Dataset Match\n{len(result["keypoints2"])} SIFT features')
    ax2.axis('off')
    
    # Correspondences visualization
    ax3 = fig.add_subplot(gs[0, 2:])
    img1 = result['detected_masked_image']
    img2 = result['dataset_masked_image']
    
    # Create side-by-side visualization
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_h = 200
    
    img1_resized = cv2.resize(img1, (int(w1 * target_h / h1), target_h))
    img2_resized = cv2.resize(img2, (int(w2 * target_h / h2), target_h))
    combined_img = np.hstack([img1_resized, img2_resized])
    
    ax3.imshow(combined_img)
    
    # Draw correspondences (only PnP inliers)
    scale1 = target_h / h1
    scale2 = target_h / h2
    offset_x = img1_resized.shape[1]
    
    pnp_inliers = pose_result['pnp_inliers']
    inlier_matches = result['inlier_matches']
    
    for i, match_idx in enumerate(pnp_inliers[:15]):  # Show up to 15 best inliers
        if match_idx < len(inlier_matches):
            match = inlier_matches[match_idx]
            pt1 = result['keypoints1'][match.queryIdx].pt
            pt2 = result['keypoints2'][match.trainIdx].pt
            
            x1, y1 = int(pt1[0] * scale1), int(pt1[1] * scale1)
            x2, y2 = int(pt2[0] * scale2 + offset_x), int(pt2[1] * scale2)
            
            color = plt.cm.viridis(i / max(len(pnp_inliers[:15]) - 1, 1))[:3]
            ax3.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.8)
            ax3.plot(x1, y1, 'o', color=color, markersize=5, markeredgecolor='white', markeredgewidth=1)
            ax3.plot(x2, y2, 's', color=color, markersize=5, markeredgecolor='white', markeredgewidth=1)
    
    ax3.set_title(f'PnP Inlier Correspondences\n{pose_result["pnp_inlier_count"]}/{pose_result["total_matches"]} inliers')
    ax3.axis('off')
    
    # Row 2: 3D Pose visualization
    ax4 = fig.add_subplot(gs[1, 0], projection='3d')
    
    # Draw 3D coordinate system
    origin = pose_result['translation_vector']
    R = pose_result['rotation_matrix']
    
    axis_length = 0.05
    colors = ['red', 'green', 'blue']
    labels = ['X', 'Y', 'Z']
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        end_point = origin + axis_length * R[:, i]
        ax4.plot([origin[0], end_point[0]], 
                [origin[1], end_point[1]], 
                [origin[2], end_point[2]], 
                color=color, linewidth=4, label=f'{label}-axis')
    
    ax4.scatter(*origin, color='black', s=100, label='Object Origin')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_zlabel('Z (m)')
    ax4.set_title('3D Pose')
    ax4.legend()
    
    # Rotation matrix heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    im = ax5.imshow(pose_result['rotation_matrix'], cmap='RdBu', vmin=-1, vmax=1)
    ax5.set_title('Rotation Matrix')
    ax5.set_xticks(range(3))
    ax5.set_yticks(range(3))
    ax5.set_xticklabels(['X', 'Y', 'Z'])
    ax5.set_yticklabels(['X', 'Y', 'Z'])
    
    for i in range(3):
        for j in range(3):
            text_color = 'white' if abs(pose_result['rotation_matrix'][i, j]) > 0.5 else 'black'
            ax5.text(j, i, f'{pose_result["rotation_matrix"][i, j]:.3f}', 
                    ha='center', va='center', fontweight='bold', color=text_color)
    
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    
    # Reprojection error visualization
    ax6 = fig.add_subplot(gs[1, 2])
    
    if len(pose_result['pnp_inliers']) > 0:
        # Calculate individual reprojection errors
        projected_pts = pose_result['projected_points']
        image_pts = pose_result['image_points'][pose_result['pnp_inliers']]
        errors = np.linalg.norm(projected_pts - image_pts, axis=1)
        
        ax6.hist(errors, bins=min(15, len(errors)), alpha=0.7, color='skyblue', edgecolor='black')
        ax6.axvline(pose_result['mean_reprojection_error'], color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {pose_result["mean_reprojection_error"]:.2f}px')
        ax6.set_xlabel('Reprojection Error (pixels)')
        ax6.set_ylabel('Count')
        ax6.set_title('Reprojection Error Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # Quality metrics
    ax7 = fig.add_subplot(gs[1, 3])
    metrics = ['PnP Inlier\nRatio', 'Reproj Error\n(inverted)', 'Match\nCount', 'Overall\nScore']
    values = [
        pose_result['pnp_inlier_ratio'],
        max(0, 1 - pose_result['mean_reprojection_error'] / 10),
        min(pose_result['pnp_inlier_count'] / 15, 1),
        result['pose_score']
    ]
    
    bars = ax7.bar(metrics, values, color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
    ax7.set_ylim(0, 1)
    ax7.set_ylabel('Quality Score')
    ax7.set_title('Pose Quality Metrics')
    ax7.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Row 3: Detailed information
    ax8 = fig.add_subplot(gs[2, :])
    ax8.axis('off')
    
    info_text = f"""
POSE ESTIMATION PIPELINE RESULTS:

INPUT DATA:
• Object: {obj_label} (ID: {result['detected_object']['label'].split('_')[-1]})
• Camera: {result['camera_type']}
• Detection bbox: {result['detected_object']['bbox']}

FEATURE MATCHING:
• Initial SIFT matches: {len(result['initial_matches'])}
• Homography RANSAC inliers: {len(result['inlier_matches'])} (ratio: {len(result['inlier_matches'])/len(result['initial_matches']):.3f})

PNP POSE ESTIMATION:
• PnP inliers: {pose_result['pnp_inlier_count']}/{pose_result['total_matches']} (ratio: {pose_result['pnp_inlier_ratio']:.3f})
• Mean reprojection error: {pose_result['mean_reprojection_error']:.3f} pixels
• Max reprojection error: {pose_result['max_reprojection_error']:.3f} pixels

3D POSE RESULT:
• Translation: [{pose_result['translation_vector'][0]:.4f}, {pose_result['translation_vector'][1]:.4f}, {pose_result['translation_vector'][2]:.4f}] meters
• Translation magnitude: {np.linalg.norm(pose_result['translation_vector']):.4f} meters

QUALITY ASSESSMENT:
• Overall pose score: {result['pose_score']:.4f} / 1.000
• Status: {"✓ HIGH QUALITY" if result['pose_score'] > 0.7 else "⚠ MEDIUM QUALITY" if result['pose_score'] > 0.4 else "✗ LOW QUALITY"}

ALGORITHM PIPELINE:
1. SIFT feature extraction (1500 features, enhanced parameters)
2. Feature matching with Lowe's ratio test (threshold: 0.65)
3. Homography RANSAC outlier elimination (threshold: 3.0px)
4. PnP pose estimation with RANSAC (threshold: 5.0px)
5. Quality assessment based on multiple metrics
"""
    
    ax8.text(0.02, 0.98, info_text, transform=ax8.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
    
    plt.tight_layout()
    
    # Save visualization
    output_filename = f"object_{obj_idx + 1:02d}_{obj_label}_pnp_pose_estimation.png"
    output_filepath = output_dir / output_filename
    plt.savefig(output_filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved pose visualization: {output_filename}")

def save_detailed_analysis(result: Dict, obj_idx: int, output_dir: Path):
    """Save detailed analysis with reprojection validation."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    pose_result = result['pose_result']
    obj_label = result['detected_object']['label']
    
    fig.suptitle(f'Detailed PnP Analysis - Object {obj_idx + 1}: {obj_label}', fontsize=14, fontweight='bold')
    
    # Original detection with reprojected points
    axes[0, 0].imshow(result['detected_masked_image'])
    
    # Show original keypoints and reprojected points
    if len(pose_result['pnp_inliers']) > 0:
        image_pts = pose_result['image_points']
        projected_pts = pose_result['projected_points']
        
        # Original points (green)
        for i, pt in enumerate(image_pts[pose_result['pnp_inliers']]):
            axes[0, 0].plot(pt[0], pt[1], 'go', markersize=6, markeredgecolor='white', markeredgewidth=1)
        
        # Reprojected points (red)
        for i, pt in enumerate(projected_pts):
            axes[0, 0].plot(pt[0], pt[1], 'rx', markersize=8, markeredgewidth=2)
            
        # Draw error vectors
        for orig_pt, proj_pt in zip(image_pts[pose_result['pnp_inliers']], projected_pts):
            axes[0, 0].plot([orig_pt[0], proj_pt[0]], [orig_pt[1], proj_pt[1]], 
                           'yellow', linewidth=1, alpha=0.7)
    
    axes[0, 0].set_title('Reprojection Validation\nGreen: Original, Red: Reprojected')
    axes[0, 0].axis('off')
    
    # Homography validation
    axes[0, 1].imshow(result['dataset_masked_image'])
    axes[0, 1].set_title(f'Dataset Reference\nHomography inliers: {len(result["inlier_matches"])}')
    axes[0, 1].axis('off')
    
    # Error analysis
    if len(pose_result['pnp_inliers']) > 1:
        axes[1, 0].scatter(range(len(pose_result['pnp_inliers'])), 
                          np.linalg.norm(pose_result['projected_points'] - 
                                       pose_result['image_points'][pose_result['pnp_inliers']], axis=1),
                          c='red', alpha=0.7)
        axes[1, 0].axhline(y=pose_result['mean_reprojection_error'], color='blue', 
                          linestyle='--', label=f'Mean: {pose_result["mean_reprojection_error"]:.2f}px')
        axes[1, 0].set_xlabel('PnP Inlier Index')
        axes[1, 0].set_ylabel('Reprojection Error (pixels)')
        axes[1, 0].set_title('Individual Reprojection Errors')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""DETAILED STATISTICS:

Feature Extraction:
• Detected: {len(result['keypoints1'])} keypoints
• Reference: {len(result['keypoints2'])} keypoints

Matching Pipeline:
• Initial matches: {len(result['initial_matches'])}
• Ratio test threshold: 0.65
• Homography inliers: {len(result['inlier_matches'])}
• Homography inlier ratio: {len(result['inlier_matches'])/len(result['initial_matches']):.3f}

PnP Estimation:
• Input matches: {pose_result['total_matches']}
• PnP inliers: {pose_result['pnp_inlier_count']}
• Final inlier ratio: {pose_result['pnp_inlier_ratio']:.3f}
• Reprojection threshold: 5.0 pixels

Error Analysis:
• Mean error: {pose_result['mean_reprojection_error']:.3f}px
• Max error: {pose_result['max_reprojection_error']:.3f}px
• RMS error: {np.sqrt(np.mean(np.linalg.norm(pose_result['projected_points'] - pose_result['image_points'][pose_result['pnp_inliers']], axis=1)**2)):.3f}px

Quality Score: {result['pose_score']:.4f}
"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
    
    output_filename = f"object_{obj_idx + 1:02d}_{obj_label}_detailed_analysis.png"
    output_filepath = output_dir / output_filename
    plt.savefig(output_filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved detailed analysis: {output_filename}")

def create_pose_summary(results: List[Dict], output_dir: Path):
    """Create summary of all pose estimation results."""
    n_results = len(results)
    fig, axes = plt.subplots(2, n_results, figsize=(5*n_results, 10))
    if n_results == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('PnP Pose Estimation Summary', fontsize=16, fontweight='bold')
    
    pose_scores = []
    reprojection_errors = []
    inlier_counts = []
    
    for i, result in enumerate(results):
        pose_result = result['pose_result']
        
        # Top row: detected objects with keypoints
        axes[0, i].imshow(result['detected_masked_image'])
        
        # Overlay some keypoints
        for j, kp in enumerate(result['keypoints1'][:20]):
            color = plt.cm.plasma(j / 20)[:3]
            axes[0, i].plot(kp.pt[0], kp.pt[1], 'o', color=color, markersize=3)
        
        axes[0, i].set_title(f"Object {i+1}\n{result['detected_object']['label']}")
        axes[0, i].axis('off')
        
        # Bottom row: pose quality visualization
        if pose_result['success']:
            # Show rotation matrix as heatmap
            im = axes[1, i].imshow(pose_result['rotation_matrix'], cmap='RdBu', vmin=-1, vmax=1)
            
            # Add text annotations
            for row in range(3):
                for col in range(3):
                    text_color = 'white' if abs(pose_result['rotation_matrix'][row, col]) > 0.5 else 'black'
                    axes[1, i].text(col, row, f'{pose_result["rotation_matrix"][row, col]:.2f}', 
                                   ha='center', va='center', fontsize=8, fontweight='bold', color=text_color)
            
            axes[1, i].set_title(f'Score: {result["pose_score"]:.3f}\nError: {pose_result["mean_reprojection_error"]:.2f}px')
            axes[1, i].set_xticks(range(3))
            axes[1, i].set_yticks(range(3))
            axes[1, i].set_xticklabels(['X', 'Y', 'Z'])
            axes[1, i].set_yticklabels(['X', 'Y', 'Z'])
            
            pose_scores.append(result['pose_score'])
            reprojection_errors.append(pose_result['mean_reprojection_error'])
            inlier_counts.append(pose_result['pnp_inlier_count'])
        else:
            axes[1, i].text(0.5, 0.5, 'PnP\nFailed', ha='center', va='center',
                           transform=axes[1, i].transAxes, fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
            axes[1, i].set_title('Pose Estimation Failed')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    
    summary_filepath = output_dir / "pnp_pose_summary.png"
    plt.savefig(summary_filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Print and save statistics
    print("\n" + "="*80)
    print("PNP POSE ESTIMATION SUMMARY")
    print("="*80)
    print(f"Algorithm: SIFT + Homography RANSAC + PnP Pose Estimation")
    print(f"Total objects processed: {n_results}")
    
    if pose_scores:
        print(f"Success rate: {len(pose_scores)}/{n_results} ({100*len(pose_scores)/n_results:.1f}%)")
        print(f"Average pose score: {np.mean(pose_scores):.4f}")
        print(f"Average reprojection error: {np.mean(reprojection_errors):.2f} pixels")
        print(f"Average PnP inliers: {np.mean(inlier_counts):.1f}")
        print(f"Best pose score: {max(pose_scores):.4f}")
        print(f"Best reprojection error: {min(reprojection_errors):.2f} pixels")
    
    print("="*80)
    
    # Save detailed statistics
    stats_filepath = output_dir / "pnp_pose_statistics.txt"
    with open(stats_filepath, 'w') as f:
        f.write("PNP POSE ESTIMATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write("ALGORITHM PIPELINE:\n")
        f.write("1. SIFT Feature Extraction (1500 features per image)\n")
        f.write("2. Feature Matching with Lowe's Ratio Test (threshold: 0.65)\n")
        f.write("3. Homography RANSAC Outlier Elimination (threshold: 3.0px)\n")
        f.write("4. PnP Pose Estimation with RANSAC (threshold: 5.0px)\n")
        f.write("5. Multi-criteria Quality Assessment\n")
        f.write("="*80 + "\n")
        f.write(f"Total objects: {n_results}\n")
        
        if pose_scores:
            f.write(f"Success rate: {len(pose_scores)}/{n_results} ({100*len(pose_scores)/n_results:.1f}%)\n")
            f.write(f"Average pose score: {np.mean(pose_scores):.4f}\n")
            f.write(f"Average reprojection error: {np.mean(reprojection_errors):.2f} pixels\n")
            f.write(f"Average PnP inliers: {np.mean(inlier_counts):.1f}\n")
        
        f.write("\nDETAILED RESULTS:\n")
        f.write("-"*50 + "\n")
        for i, result in enumerate(results):
            f.write(f"Object {i+1}: {result['detected_object']['label']}\n")
            if result['pose_result']['success']:
                f.write(f"  Pose score: {result['pose_score']:.4f}\n")
                f.write(f"  Reprojection error: {result['pose_result']['mean_reprojection_error']:.3f}px\n")
                f.write(f"  PnP inliers: {result['pose_result']['pnp_inlier_count']}\n")
                f.write(f"  Translation: {result['pose_result']['translation_vector']}\n")
            else:
                f.write(f"  Status: FAILED - {result['pose_result']['reason']}\n")
            f.write("-"*50 + "\n")
    
    print(f"Saved summary: {summary_filepath}")
    print(f"Saved statistics: {stats_filepath}")

# Example usage
def test_pnp_pose_estimation():
    """Test function for PnP pose estimation."""
    print("PnP Pose Estimation with Homography RANSAC Test")
    print("="*60)
    print("Pipeline:")
    print("1. SIFT feature extraction")
    print("2. Feature matching with ratio test")
    print("3. Homography RANSAC outlier elimination")
    print("4. PnP pose estimation")
    print("5. Quality assessment")
    print("="*60)
    
    # Example call (replace with actual file paths)
    results = object_matching_and_pose_estimation("/home/rama/bpc_ws/bpc/maskRCNN/results/000000_cam1_annotation.json", "/home/rama/bpc_ws/bpc/utilities/filtered_dataset.csv")
    
    return None

if __name__ == "__main__":
    test_pnp_pose_estimation()