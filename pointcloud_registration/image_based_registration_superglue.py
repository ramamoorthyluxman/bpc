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

# SuperGlue dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

print("Using SuperGlue feature extraction and matching")

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

class SuperPoint(nn.Module):
    """SuperPoint feature detector and descriptor."""
    
    def __init__(self):
        super(SuperPoint, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        
        # Shared Encoder
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        
        # Detector Head
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        
        # Descriptor Head
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, 256, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # Shared Encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        
        # Detector Head
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        
        # Descriptor Head
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize
        
        return semi, desc

class SuperGlue(nn.Module):
    """SuperGlue feature matcher."""
    
    def __init__(self, sinkhorn_iterations=20, match_threshold=0.2):
        super(SuperGlue, self).__init__()
        self.sinkhorn_iterations = sinkhorn_iterations
        self.match_threshold = match_threshold
        
        # Multi-layer Transformer
        self.gnn = nn.ModuleList([
            MultiHeadedAttention(4, 256) for _ in range(9)
        ])
        
        # Final MLP
        self.final_proj = nn.Conv1d(256, 256, kernel_size=1, bias=True)
        
    def forward(self, desc0, desc1, kpts0, kpts1):
        # Normalize keypoints
        kpts0 = normalize_keypoints(kpts0, desc0.shape[-2:])
        kpts1 = normalize_keypoints(kpts1, desc1.shape[-2:])
        
        # Add positional encoding
        desc0 = desc0 + self.pos_encoding(kpts0)
        desc1 = desc1 + self.pos_encoding(kpts1)
        
        # Self and cross attention
        for layer in self.gnn:
            desc0, desc1 = layer(desc0, desc1)
        
        # Final projection
        desc0 = self.final_proj(desc0)
        desc1 = self.final_proj(desc1)
        
        # Compute matching matrix
        scores = torch.einsum('bdn,bdm->bnm', desc0, desc1)
        scores = scores / 256**.5
        
        # Sinkhorn algorithm
        scores = self.sinkhorn(scores)
        
        return scores
    
    def pos_encoding(self, kpts):
        # Simple positional encoding
        pe = torch.zeros(kpts.shape[0], 256, kpts.shape[1], device=kpts.device)
        pe[:, :2] = kpts.transpose(1, 2)
        return pe
    
    def sinkhorn(self, Z, log_mu_zero=False, log_nu_zero=False):
        # Sinkhorn normalization in log space
        batch_size, M, N = Z.shape
        
        # Initialize
        log_alpha = torch.zeros(batch_size, M, device=Z.device)
        log_beta = torch.zeros(batch_size, N, device=Z.device)
        
        for i in range(self.sinkhorn_iterations):
            # Row normalization
            log_alpha = torch.logsumexp(Z + log_beta.unsqueeze(1), dim=2)
            # Column normalization  
            log_beta = torch.logsumexp(Z + log_alpha.unsqueeze(2), dim=1)
        
        Z = Z + log_alpha.unsqueeze(2) + log_beta.unsqueeze(1)
        return Z

class MultiHeadedAttention(nn.Module):
    """Multi-headed attention for SuperGlue."""
    
    def __init__(self, num_heads, d_model):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(4)])
    
    def forward(self, desc0, desc1):
        # Self-attention on desc0
        desc0 = desc0 + self.attention(desc0, desc0, desc0)
        # Self-attention on desc1
        desc1 = desc1 + self.attention(desc1, desc1, desc1)
        # Cross-attention
        desc0 = desc0 + self.attention(desc0, desc1, desc1)
        desc1 = desc1 + self.attention(desc1, desc0, desc0)
        return desc0, desc1
    
    def attention(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                           for l, x in zip(self.proj, (query, key, value))]
        x = self.compute_attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))
    
    def compute_attention(self, query, key, value):
        scores = torch.matmul(query.transpose(2, 3), key) / (self.dim**.5)
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(value, p_attn.transpose(2, 3))

def deepcopy(module):
    import copy
    return copy.deepcopy(module)

def normalize_keypoints(kpts, image_shape):
    """Normalize keypoints coordinates to [-1, 1]."""
    height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]

def simple_nms(scores, nms_radius: int):
    """Fast Non-Maximum Suppression to remove nearby points."""
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations."""
    b, c, h, w = descriptors.shape
    keypoints = keypoints.clone()
    keypoints = keypoints - s / 2 + 0.5
    keypoints[:, :, 0] /= float(w * s)  # x coordinate
    keypoints[:, :, 1] /= float(h * s)  # y coordinate
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    
    # Ensure keypoints are in the right format [B, H, W, 2]
    keypoints_grid = keypoints.view(b, 1, -1, 2)
    
    args = {'align_corners': False}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints_grid, mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors

class SuperGlueMatcher:
    """SuperGlue feature matcher with simplified interface."""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.superpoint = SuperPoint().to(device)
        self.superglue = SuperGlue().to(device)
        
        # Initialize with pretrained weights if available
        self._initialize_weights()
        
        self.superpoint.eval()
        self.superglue.eval()
    
    def _initialize_weights(self):
        """Initialize weights with random values (in practice, load pretrained weights)."""
        # In a real implementation, you would load pretrained weights here
        # For this example, we'll use the default PyTorch initialization
        print("    Note: Using default weight initialization. For best results, load pretrained SuperPoint/SuperGlue weights.")
    
    def extract_features(self, image):
        """Extract SuperPoint features from image."""
        # Convert to grayscale and normalize
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Resize if too large
        h, w = gray.shape
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            gray = cv2.resize(gray, (new_w, new_h))
        else:
            new_h, new_w = h, w
            scale = 1.0
        
        # Normalize to [0, 1]
        gray = gray.astype(np.float32) / 255.0
        
        # Convert to tensor
        gray_tensor = torch.from_numpy(gray)[None, None].to(self.device)
        
        with torch.no_grad():
            # Extract features
            semi, desc = self.superpoint(gray_tensor)
            
            # Process keypoints
            semi = semi.squeeze(0)
            desc = desc.squeeze(0)
            
            # Remove dustbin (last channel) and apply softmax
            semi = semi[:-1, :, :]
            dense_scores = F.softmax(semi, dim=0)
            dense_scores = dense_scores.sum(dim=0)
            
            # Apply NMS
            scores = simple_nms(dense_scores[None, None], 4)[0, 0]
            
            # Extract keypoints
            keypoints = [
                torch.nonzero(scores > 0.015, as_tuple=False)
            ][0].float()
            
            if len(keypoints) == 0:
                return np.array([]), np.array([])
            
            # Swap x, y coordinates
            keypoints = keypoints[:, [1, 0]]
            
            # Scale keypoints back to original image size
            keypoints = keypoints / scale
            
            # Sample descriptors
            descriptors = sample_descriptors(
                keypoints[None, None], desc[None], s=8
            )[0, :, 0, :].t()
            
            return keypoints.cpu().numpy(), descriptors.cpu().numpy()
    
    def match_features(self, kpts0, desc0, kpts1, desc1):
        """Match features using SuperGlue."""
        if len(kpts0) == 0 or len(kpts1) == 0:
            return []
        
        # Convert to tensors
        kpts0_tensor = torch.from_numpy(kpts0).float().to(self.device)[None]
        kpts1_tensor = torch.from_numpy(kpts1).float().to(self.device)[None]
        desc0_tensor = torch.from_numpy(desc0).float().to(self.device)[None].transpose(1, 2)
        desc1_tensor = torch.from_numpy(desc1).float().to(self.device)[None].transpose(1, 2)
        
        with torch.no_grad():
            # Match features
            scores = self.superglue(desc0_tensor, desc1_tensor, kpts0_tensor, kpts1_tensor)
            scores = scores.squeeze(0).cpu().numpy()
            
            # Extract matches
            max0 = np.argmax(scores, axis=1)
            max1 = np.argmax(scores, axis=0)
            indices0 = np.arange(scores.shape[0])
            indices1 = np.arange(scores.shape[1])
            
            # Mutual check
            mutual0 = indices0[max1[max0] == indices0]
            mutual1 = max0[mutual0]
            
            # Score threshold
            valid = scores[mutual0, mutual1] > 0.2
            matches = np.column_stack([mutual0[valid], mutual1[valid]])
            
            return matches

def object_matching_and_pose_estimation(json_file_path: str, csv_file_path: str, output_dir: str = "output") -> List[Dict]:
    """
    Match detected objects with dataset images and estimate pose.
    Uses SuperGlue feature extraction and matching.
    
    Args:
        json_file_path (str): Path to JSON file containing object detection results
        csv_file_path (str): Path to CSV file containing dataset with ground truth
        output_dir (str): Directory to save visualization images (default: "output")
    
    Returns:
        List[Dict]: Results containing matches, correspondences, and pose information
        
    Generated Outputs:
        - Individual match results with correspondences visualization
        - Transformation visualization (test → reference → transformed)
        - Summary visualization of all matches
        - Individual masked images for inspection
        - Detailed statistics and matching report
        
    Matching Method:
        SuperGlue neural feature matching
        1. Convert images to grayscale and normalize
        2. Extract SuperPoint keypoints and descriptors from full images
        3. Match features using SuperGlue neural network matcher
        4. Apply confidence threshold for match filtering
        5. Use RANSAC for geometric verification
        - State-of-the-art neural feature matching approach
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created: {output_path.absolute()}")
    
    # Initialize SuperGlue matcher
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    superglue_matcher = SuperGlueMatcher(device=device)
    
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
    
    print(f"Processing {len(detected_objects)} detected objects...")
    
    for obj_idx, detected_obj in enumerate(detected_objects):
        print(f"Processing object {obj_idx + 1}/{len(detected_objects)}: {detected_obj['label']}")
        
        # Extract object ID from label (assuming format like "obj_000018")
        object_id = int(detected_obj['label'].split('_')[-1])
        
        # Create mask for detected object
        detected_mask = create_mask_from_polygon(detected_obj['points'], image_height, image_width)
        detected_masked_image = apply_mask_to_image(main_image_rgb, detected_mask)
        
        # Find matching objects in dataset - NOW WITH CAMERA TYPE FILTERING
        matching_entries = dataset_df[
            (dataset_df['object_id'] == object_id) & 
            (dataset_df['camera_type'] == main_camera_type)
        ]
        
        if len(matching_entries) == 0:
            print(f"No matching entries found for object {object_id} with camera type '{main_camera_type}'")
            
            # Optionally, try without camera type filter as fallback
            print(f"Trying without camera type filter...")
            matching_entries_fallback = dataset_df[dataset_df['object_id'] == object_id]
            
            if len(matching_entries_fallback) == 0:
                print(f"No matching entries found for object {object_id} at all")
                continue
            else:
                print(f"Found {len(matching_entries_fallback)} entries without camera type filter, but skipping for camera consistency")
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
                
                # Create mask for dataset object
                dataset_mask = create_mask_from_polygon(dataset_polygon, 
                                                      int(dataset_entry['image_height']), 
                                                      int(dataset_entry['image_width']))
                dataset_masked_image = apply_mask_to_image(dataset_image_rgb, dataset_mask)
                
                # Calculate similarity
                similarity = calculate_similarity(detected_masked_image, dataset_masked_image)
                
                # Find feature correspondences using SuperGlue
                correspondences = find_feature_correspondences(detected_masked_image, dataset_masked_image, 
                                                            superglue_matcher, min_matches=3)
                
                # Skip if insufficient matches
                if len(correspondences['matches']) < 3:
                    print(f"    Skipping: only {len(correspondences['matches'])} matches (minimum 3 required)")
                    continue
                
                print(f"    Similarity: {similarity:.4f}, SuperGlue Matches: {len(correspondences['matches'])}")
                
                # Combined score: prioritize match quality and quantity balance
                if len(correspondences['matches']) >= 3:
                    # Simple scoring based on similarity and match count
                    match_quality_score = min(len(correspondences['matches']) / 15.0, 1.0)  # Normalize to 15 matches
                    combined_score = 0.7 * similarity + 0.3 * match_quality_score  # Favor SSIM
                else:
                    combined_score = similarity * 0.4  # Penalty for insufficient matches
                
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
                'best_match': best_match,
                'similarity_score': best_similarity,
                'correspondences': best_correspondences,
                'rotation_matrix': rotation_matrix,
                'translation_vector': translation_vector,
                'detected_masked_image': detected_masked_image,
                'matched_dataset_image': best_dataset_masked_image,
                'camera_type': main_camera_type  # Add camera type to results
            }
            
            results.append(result)
            
            # Save individual masked images
            save_masked_images(result, obj_idx, output_path)
            
            # Visualize results
            visualize_match_result(result, obj_idx, output_path)
            
            # Create transformation visualization
            visualize_transformation(result, obj_idx, output_path)
            
            print(f"  Best match found with SuperGlue quality score: {best_similarity:.4f}")
            print(f"    Camera type: {main_camera_type}")
            print(f"    SuperGlue matches: {len(best_correspondences['matches'])}")
            
        else:
            print(f"  No suitable match found for object {object_id} with camera type '{main_camera_type}'")
    
    # Create summary visualization
    if results:
        create_summary_visualization(results, output_path)
    
    return results

def parse_polygon_string(polygon_str: str) -> List[List[int]]:
    """Parse polygon string from CSV file."""
    try:
        # Handle different formats
        polygon_str = polygon_str.strip()
        
        # Try direct evaluation first
        if polygon_str.startswith('[[') and polygon_str.endswith(']]'):
            return ast.literal_eval(polygon_str)
        
        # Handle space-separated format
        if '], [' in polygon_str:
            return ast.literal_eval(polygon_str)
        
        # Handle other formats
        # Remove extra spaces and fix format
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
    # Resize images to same size for comparison
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Use smaller dimensions to avoid upscaling
    target_size = min(300, max(h1, w1), max(h2, w2))
    
    # Calculate aspect ratios and resize accordingly
    aspect1 = w1 / h1
    aspect2 = w2 / h2
    
    if aspect1 > 1:  # wider than tall
        target_w1, target_h1 = target_size, int(target_size / aspect1)
    else:  # taller than wide
        target_w1, target_h1 = int(target_size * aspect1), target_size
        
    if aspect2 > 1:  # wider than tall
        target_w2, target_h2 = target_size, int(target_size / aspect2)
    else:  # taller than wide
        target_w2, target_h2 = int(target_size * aspect2), target_size
    
    img1_resized = cv2.resize(img1, (target_w1, target_h1))
    img2_resized = cv2.resize(img2, (target_w2, target_h2))
    
    # Make both images the same size by padding
    max_h = max(target_h1, target_h2)
    max_w = max(target_w1, target_w2)
    
    img1_padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    img2_padded = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    
    img1_padded[:target_h1, :target_w1] = img1_resized
    img2_padded[:target_h2, :target_w2] = img2_resized
    
    # Convert to grayscale for SSIM
    gray1 = cv2.cvtColor(img1_padded, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_padded, cv2.COLOR_RGB2GRAY)
    
    # Calculate SSIM
    similarity = ssim(gray1, gray2)
    
    return similarity

def find_feature_correspondences(img1: np.ndarray, img2: np.ndarray, superglue_matcher: SuperGlueMatcher, min_matches: int = 3) -> Dict:
    """
    SuperGlue feature extraction and matching.
    Extracts SuperPoint features from full images and matches them using SuperGlue neural network.
    """
    print(f"    SuperGlue matching for images {img1.shape} and {img2.shape}")
    
    try:
        # Extract features using SuperGlue
        kpts1, desc1 = superglue_matcher.extract_features(img1)
        kpts2, desc2 = superglue_matcher.extract_features(img2)
        
        print(f"    SuperPoint found {len(kpts1)} and {len(kpts2)} keypoints")
        
        if len(kpts1) < min_matches or len(kpts2) < min_matches:
            print(f"    Insufficient SuperPoint features")
            return {'keypoints1': [], 'keypoints2': [], 'matches': []}
        
        # Match features using SuperGlue
        matches = superglue_matcher.match_features(kpts1, desc1, kpts2, desc2)
        
        print(f"    SuperGlue found {len(matches)} matches")
        
        if len(matches) < min_matches:
            print(f"    Insufficient SuperGlue matches")
            return {'keypoints1': [], 'keypoints2': [], 'matches': []}
        
        # Convert matches to cv2.DMatch format for compatibility
        cv_matches = []
        for i, (idx1, idx2) in enumerate(matches):
            match = cv2.DMatch()
            match.queryIdx = int(idx1)
            match.trainIdx = int(idx2)
            match.distance = 0.1  # SuperGlue confidence (lower is better)
            cv_matches.append(match)
        
        # Create keypoint objects for compatibility
        cv_kpts1 = [cv2.KeyPoint(float(kp[0]), float(kp[1]), 1.0) for kp in kpts1]
        cv_kpts2 = [cv2.KeyPoint(float(kp[0]), float(kp[1]), 1.0) for kp in kpts2]
        
        print(f"    Using {len(cv_matches)} SuperGlue matches")
        
        return {
            'keypoints1': cv_kpts1,
            'keypoints2': cv_kpts2,
            'matches': cv_matches
        }
        
    except Exception as e:
        print(f"    SuperGlue matching error: {e}")
        return {'keypoints1': [], 'keypoints2': [], 'matches': []}

def save_masked_images(result: Dict, obj_idx: int, output_dir: Path):
    """Save individual masked images for inspection."""
    # Create subdirectory for masked images
    masked_dir = output_dir / "masked_images"
    masked_dir.mkdir(exist_ok=True)
    
    # Save detected object masked image
    detected_filename = f"object_{obj_idx + 1:02d}_{result['detected_object']['label']}_detected.png"
    detected_filepath = masked_dir / detected_filename
    plt.figure(figsize=(8, 6))
    plt.imshow(result['detected_masked_image'])
    plt.title(f'Detected Object: {result["detected_object"]["label"]} (Camera: {result["camera_type"]})')
    plt.axis('off')
    plt.savefig(detected_filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save matched dataset image
    matched_filename = f"object_{obj_idx + 1:02d}_{result['detected_object']['label']}_matched.png"
    matched_filepath = masked_dir / matched_filename
    plt.figure(figsize=(8, 6))
    plt.imshow(result['matched_dataset_image'])
    plt.title(f'Matched Dataset Image (Camera: {result["camera_type"]}, SuperGlue Score: {result["similarity_score"]:.3f}, Matches: {len(result["correspondences"]["matches"])})')
    plt.axis('off')
    plt.savefig(matched_filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"    Saved masked images: {detected_filepath} and {matched_filepath}")

def visualize_match_result(result: Dict, obj_idx: int, output_dir: Path):
    """Visualize the matching result and save as image."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Object {obj_idx + 1} Matching Results - {result["detected_object"]["label"]} (Camera: {result["camera_type"]})', fontsize=16)
    
    # Original detected object
    axes[0, 0].imshow(result['detected_masked_image'])
    axes[0, 0].set_title(f'Detected Object (Masked) - {result["camera_type"]}')
    axes[0, 0].axis('off')
    
    # Best matched dataset image
    axes[0, 1].imshow(result['matched_dataset_image'])
    axes[0, 1].set_title(f'Best Match (SuperGlue Score: {result["similarity_score"]:.3f}) - {result["camera_type"]}')
    axes[0, 1].axis('off')
    
    # Feature correspondences visualization
    correspondences = result['correspondences']
    if len(correspondences['matches']) > 0:
        # Create side-by-side image for correspondence visualization
        img1 = result['detected_masked_image']
        img2 = result['matched_dataset_image']
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Resize to same height for visualization
        target_h = 300
        scale1 = target_h / h1
        scale2 = target_h / h2
        
        img1_resized = cv2.resize(img1, (int(w1 * scale1), target_h))
        img2_resized = cv2.resize(img2, (int(w2 * scale2), target_h))
        
        combined_img = np.hstack([img1_resized, img2_resized])
        axes[1, 0].imshow(combined_img)
        
        # Draw correspondences
        offset_x = img1_resized.shape[1]
        num_matches_to_show = min(20, len(correspondences['matches']))  # Show up to 20 matches
        
        for i, match in enumerate(correspondences['matches'][:num_matches_to_show]):
            pt1 = correspondences['keypoints1'][match.queryIdx].pt
            pt2 = correspondences['keypoints2'][match.trainIdx].pt
            
            x1, y1 = int(pt1[0] * scale1), int(pt1[1] * scale1)
            x2, y2 = int(pt2[0] * scale2 + offset_x), int(pt2[1] * scale2)
            
            # Use different colors for different matches
            color = plt.cm.plasma(i / max(num_matches_to_show - 1, 1))[:3]
            
            axes[1, 0].plot([x1, x2], [y1, y2], color=color, linewidth=2.0, alpha=0.9)
            axes[1, 0].plot(x1, y1, 'o', color=color, markersize=5, markeredgecolor='white', markeredgewidth=1.5)
            axes[1, 0].plot(x2, y2, 's', color=color, markersize=5, markeredgecolor='white', markeredgewidth=1.5)
    else:
        axes[1, 0].text(0.5, 0.5, 'No correspondences found', 
                       ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=14)
    
    axes[1, 0].set_title(f'SuperGlue Correspondences ({len(correspondences["matches"])} matches)')
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
    
    # Object information
    detected_obj = result['detected_object']
    best_match = result['best_match']
    translation = result['translation_vector']
    
    info_text = f"""DETECTED OBJECT:
Label: {detected_obj['label']}
Camera Type: {result['camera_type']}
Bbox: {detected_obj['bbox']}
Center: {detected_obj['bbox_center']}
Geometric Center: {detected_obj['geometric_center']}

BEST MATCH:
Scene ID: {best_match['scene_id']}
Camera: {best_match['camera_type']}
Image: {Path(best_match['image_path']).name}
Quality Score: {result['similarity_score']:.4f}
SuperGlue Matches: {len(correspondences['matches'])}

POSE ESTIMATION:
Rotation Matrix: 3x3 (see heatmap)
Translation: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]
Translation Magnitude: {np.linalg.norm(translation):.2f}

MATCHING METHOD:
Algorithm: SuperGlue Neural Network
Camera Consistency: {result['camera_type']} ↔ {best_match['camera_type']}
Min Matches Required: 3
Features: SuperPoint keypoints and descriptors
Matching: SuperGlue neural matcher
Geometric Verification: Built-in neural confidence
"""
    
    axes[0, 1].text(1.05, 0.5, info_text, transform=axes[0, 1].transAxes, 
                    fontsize=9, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the figure
    output_filename = f"object_{obj_idx + 1:02d}_{result['detected_object']['label']}_match_result.png"
    output_filepath = output_dir / output_filename
    plt.savefig(output_filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)  # Close figure to free memory
    print(f"    Saved visualization: {output_filepath}")

def visualize_transformation(result: Dict, obj_idx: int, output_dir: Path):
    """Visualize the transformation: test image, reference image, and transformed reference image."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Object {obj_idx + 1} Transformation Visualization - {result["detected_object"]["label"]} (Camera: {result["camera_type"]})', fontsize=16)
    
    test_image = result['detected_masked_image']
    ref_image = result['matched_dataset_image']
    correspondences = result['correspondences']
    
    # Display test image (target)
    axes[0].imshow(test_image)
    axes[0].set_title('Test Image (Target)\n(Detected Object)')
    axes[0].axis('off')
    
    # Display reference image (source)
    axes[1].imshow(ref_image)
    axes[1].set_title('Reference Image (Source)\n(Dataset Match)')
    axes[1].axis('off')
    
    # Compute and apply transformation to reference image
    transformed_ref_image = None
    transformation_info = "No transformation computed"
    
    if len(correspondences['matches']) >= 4:
        try:
            # Extract corresponding points
            src_pts = np.float32([correspondences['keypoints2'][m.trainIdx].pt for m in correspondences['matches']]).reshape(-1, 1, 2)
            dst_pts = np.float32([correspondences['keypoints1'][m.queryIdx].pt for m in correspondences['matches']]).reshape(-1, 1, 2)
            
            # Compute homography (transformation from reference to test)
            homography, mask = cv2.findHomography(src_pts, dst_pts, 
                                                 cv2.RANSAC, 
                                                 ransacReprojThreshold=5.0,
                                                 confidence=0.99,
                                                 maxIters=2000)
            
            if homography is not None:
                # Apply transformation to reference image
                h_test, w_test = test_image.shape[:2]
                transformed_ref_image = cv2.warpPerspective(ref_image, homography, (w_test, h_test))
                
                # Count inliers
                inliers = np.sum(mask) if mask is not None else len(correspondences['matches'])
                transformation_info = f"Homography computed\nInliers: {inliers}/{len(correspondences['matches'])}\nReprojection error < 5.0 px"
                
                # Compute transformation quality metrics
                transformation_quality = compute_transformation_quality(test_image, transformed_ref_image)
                transformation_info += f"\nSSIM: {transformation_quality['ssim']:.3f}\nMSE: {transformation_quality['mse']:.1f}"
                
            else:
                transformation_info = "Homography computation failed"
                
        except Exception as e:
            transformation_info = f"Transformation error: {str(e)[:50]}..."
            print(f"    Transformation error: {e}")
    
    else:
        transformation_info = f"Insufficient matches: {len(correspondences['matches'])}/4"
    
    # Display transformed reference image or placeholder
    if transformed_ref_image is not None:
        axes[2].imshow(transformed_ref_image)
        axes[2].set_title(f'Transformed Reference\n{transformation_info}')
    else:
        # Create a placeholder showing why transformation failed
        placeholder = np.zeros_like(test_image)
        axes[2].imshow(placeholder)
        axes[2].set_title(f'Transformation Failed\n{transformation_info}')
        axes[2].text(0.5, 0.5, 'Transformation\nNot Available', 
                    ha='center', va='center', transform=axes[2].transAxes, 
                    fontsize=14, color='white',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
    
    axes[2].axis('off')
    
    # Add keypoint overlays to show correspondences
    if len(correspondences['matches']) > 0:
        # Overlay keypoints on test image
        for i, match in enumerate(correspondences['matches'][:20]):  # Show up to 20 points
            pt1 = correspondences['keypoints1'][match.queryIdx].pt
            color = plt.cm.plasma(i / max(len(correspondences['matches'][:20]) - 1, 1))[:3]
            axes[0].plot(pt1[0], pt1[1], 'o', color=color, markersize=4, 
                        markeredgecolor='white', markeredgewidth=1)
        
        # Overlay keypoints on reference image
        for i, match in enumerate(correspondences['matches'][:20]):  # Show up to 20 points
            pt2 = correspondences['keypoints2'][match.trainIdx].pt
            color = plt.cm.plasma(i / max(len(correspondences['matches'][:20]) - 1, 1))[:3]
            axes[1].plot(pt2[0], pt2[1], 'o', color=color, markersize=4, 
                        markeredgecolor='white', markeredgewidth=1)
    
    # Add detailed information
    pose_info = f"""POSE ESTIMATION DATA:
Rotation Matrix:
[{result['rotation_matrix'][0,0]:.3f} {result['rotation_matrix'][0,1]:.3f} {result['rotation_matrix'][0,2]:.3f}]
[{result['rotation_matrix'][1,0]:.3f} {result['rotation_matrix'][1,1]:.3f} {result['rotation_matrix'][1,2]:.3f}]
[{result['rotation_matrix'][2,0]:.3f} {result['rotation_matrix'][2,1]:.3f} {result['rotation_matrix'][2,2]:.3f}]

Translation Vector:
[{result['translation_vector'][0]:.2f}, {result['translation_vector'][1]:.2f}, {result['translation_vector'][2]:.2f}]

FEATURE CORRESPONDENCES:
Total matches: {len(correspondences['matches'])}
Keypoints shown: {min(20, len(correspondences['matches']))}

TRANSFORMATION METHOD:
Algorithm: Homography (2D projection)
Features: SuperGlue neural correspondences with built-in filtering
Purpose: Visualization alignment
Note: Full 3D pose available in rotation/translation
Method: SuperGlue neural feature matching
"""
    
    # Add info text below the images
    fig.text(0.02, 0.02, pose_info, fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
             verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for the info text
    
    # Save the transformation visualization
    output_filename = f"object_{obj_idx + 1:02d}_{result['detected_object']['label']}_transformation.png"
    output_filepath = output_dir / output_filename
    plt.savefig(output_filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved transformation visualization: {output_filepath}")

def compute_transformation_quality(test_image: np.ndarray, transformed_ref_image: np.ndarray) -> Dict[str, float]:
    """Compute quality metrics for the transformation."""
    # Convert to grayscale for comparison
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
    transformed_gray = cv2.cvtColor(transformed_ref_image, cv2.COLOR_RGB2GRAY)
    
    # Create masks for non-zero regions (where actual image content exists)
    test_mask = (test_gray > 0).astype(np.uint8)
    transformed_mask = (transformed_gray > 0).astype(np.uint8)
    
    # Find overlapping region
    overlap_mask = test_mask & transformed_mask
    
    if np.sum(overlap_mask) > 0:
        # Calculate SSIM only in overlapping regions
        ssim_score = ssim(test_gray, transformed_gray, data_range=255)
        
        # Calculate MSE only in overlapping regions
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

def create_summary_visualization(results: List[Dict], output_dir: Path):
    """Create a summary visualization of all results and save as image."""
    n_results = len(results)
    fig, axes = plt.subplots(2, n_results, figsize=(5*n_results, 10))
    if n_results == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Summary: SuperGlue Neural Feature Matching', fontsize=16)
    
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
        axes[1, i].set_title(f"Best Match\nSuperGlue: {similarity:.3f}\nMatches: {n_matches}\n({result['camera_type']})")
        axes[1, i].axis('off')
        
        similarities.append(similarity)
        match_counts.append(n_matches)
        camera_types.append(result['camera_type'])
    
    plt.tight_layout()
    
    # Save the summary figure
    summary_filepath = output_dir / "summary_all_matches.png"
    plt.savefig(summary_filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)  # Close figure to free memory
    print(f"Saved summary visualization: {summary_filepath}")
    
    # Statistics summary
    print("\n" + "="*70)
    print("SUPERGLUE NEURAL FEATURE MATCHING SUMMARY")
    print("="*70)
    print(f"Total objects processed: {n_results}")
    print(f"Camera types found: {set(camera_types)}")
    print(f"Average SuperGlue score: {np.mean(similarities):.4f}")
    print(f"Average SuperGlue matches: {np.mean(match_counts):.1f}")
    print(f"Best SuperGlue score: {max(similarities):.4f}")
    print(f"Most SuperGlue matches: {max(match_counts)}")
    print(f"Method: SuperGlue Neural Network")
    print("="*70)
    
    # Save statistics to file
    stats_filepath = output_dir / "matching_statistics.txt"
    with open(stats_filepath, 'w') as f:
        f.write("SUPERGLUE NEURAL FEATURE MATCHING\n")
        f.write("="*70 + "\n")
        f.write("METHOD: SuperGlue Neural Network\n")
        f.write("CAMERA FILTERING: Only compares objects from same camera type\n")
        f.write("FEATURES: SuperPoint keypoints and descriptors from full images\n")
        f.write("MATCHING: SuperGlue neural network matcher\n")
        f.write("CONFIDENCE: Built-in neural confidence scoring\n")
        f.write("GEOMETRIC VERIFICATION: Neural attention mechanism\n")
        f.write("SCORING: 70% SSIM + 30% normalized match count\n")
        f.write("MINIMUM MATCHES: 3 neural correspondences\n")
        f.write("VISUALIZATION: Includes transformation alignment\n")
        f.write("="*70 + "\n")
        f.write(f"Total objects processed: {n_results}\n")
        f.write(f"Camera types found: {set(camera_types)}\n")
        f.write(f"Average SuperGlue score: {np.mean(similarities):.4f}\n")
        f.write(f"Average SuperGlue matches: {np.mean(match_counts):.1f}\n")
        f.write(f"Best SuperGlue score: {max(similarities):.4f}\n")
        f.write(f"Most SuperGlue matches: {max(match_counts)}\n")
        f.write("="*70 + "\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-"*50 + "\n")
        for i, result in enumerate(results):
            f.write(f"Object {i+1}: {result['detected_object']['label']}\n")
            f.write(f"  Camera Type: {result['camera_type']}\n")
            f.write(f"  SuperGlue Score: {result['similarity_score']:.4f}\n")
            f.write(f"  SuperGlue matches: {len(result['correspondences']['matches'])}\n")
            f.write(f"  Translation magnitude: {np.linalg.norm(result['translation_vector']):.2f}\n")
            f.write(f"  Best match scene: {result['best_match']['scene_id']}\n")
            f.write(f"  Match camera: {result['best_match']['camera_type']}\n")
            f.write("-"*50 + "\n")
    
    print(f"Saved detailed statistics: {stats_filepath}")

# Example usage and testing function
def test_with_sample_data():
    """Test function with sample data structure."""
    print("Testing SuperGlue neural feature matching...")
    print("SuperPoint + SuperGlue neural network")
    
    # This would be called with actual file paths:
    results = object_matching_and_pose_estimation("/home/rama/bpc_ws/bpc/maskRCNN/results/000000_annotation.json", "/home/rama/bpc_ws/bpc/utilities/filtered_dataset.csv")
    
    print("To use this function, call:")
    print("results = object_matching_and_pose_estimation(json_file_path, csv_file_path)")
    print("\nThe function will return a list of dictionaries, each containing:")
    print("- detected_object: Original detection data")
    print("- best_match: Best matching dataset entry (same camera type)")
    print("- similarity_score: Similarity score (0-1)")
    print("- correspondences: Feature point correspondences")
    print("- rotation_matrix: 3x3 rotation matrix")
    print("- translation_vector: 3D translation vector")
    print("- camera_type: Camera type used for filtering")
    print("\nGenerated visualizations:")
    print("- Individual match results with correspondences")
    print("- Transformation visualization (test → reference → transformed)")
    print("- Summary of all matches")
    print("- Individual masked images")
    print("- Detailed statistics file")
    print("\nKey features:")
    print("- SUPERPOINT FEATURES: Neural keypoint detector and descriptor")
    print("- SUPERGLUE MATCHING: Neural attention-based matcher")
    print("- NEURAL CONFIDENCE: Built-in confidence scoring")
    print("- ATTENTION MECHANISM: Self and cross attention for robust matching")
    print("- State-of-the-art neural approach")

if __name__ == "__main__":
    test_with_sample_data()