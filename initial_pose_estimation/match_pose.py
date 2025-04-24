import os
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image
import pickle
import argparse
import matplotlib.pyplot as plt
import logging
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_feature_extractor():
    """
    Create a feature extractor using a pre-trained ResNet model
    """
    # Load pre-trained ResNet model
    model = models.resnet50(weights='IMAGENET1K_V2')
    
    # Remove the classification layer
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.to(device)
    
    return feature_extractor, device

def extract_features(model, image, device):
    """
    Extract features from an image using the model
    """
    with torch.no_grad():
        features = model(image.unsqueeze(0).to(device))
        features = features.squeeze().cpu().numpy()
    
    return features

def normalize_features(features):
    """
    L2 normalize features for cosine similarity
    """
    norm = np.linalg.norm(features, keepdims=True)
    return features / (norm + 1e-10)  # Add small epsilon to avoid division by zero

def load_image(image_path, transform=None):
    """
    Load an image and apply transformations
    """
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

def find_nearest_image(query_features, index_data):
    """
    Find the nearest image using the precomputed nearest neighbors index
    """
    # Apply PCA transformation to query features
    query_features_reduced = index_data['pca'].transform(query_features.reshape(1, -1))
    
    # Normalize features
    query_features_reduced = normalize_features(query_features_reduced)
    
    # Find nearest neighbor
    distances, indices = index_data['nn_index'].kneighbors(query_features_reduced)
    
    nearest_idx = indices[0][0]
    distance = distances[0][0]
    
    # Get corresponding image path and rotation matrix
    image_path = index_data['image_paths'][nearest_idx]
    image_name = index_data['image_names'][nearest_idx]
    rotation_matrix = index_data['rotation_matrices'][nearest_idx]
    
    return {
        'nearest_idx': nearest_idx,
        'distance': distance,
        'image_path': image_path,
        'image_name': image_name,
        'rotation_matrix': rotation_matrix
    }

def find_k_nearest_images(query_features, index_data, k=5):
    """
    Find k nearest images for keypoint verification
    """
    # Apply PCA transformation to query features
    query_features_reduced = index_data['pca'].transform(query_features.reshape(1, -1))
    
    # Normalize features
    query_features_reduced = normalize_features(query_features_reduced)
    
    # Modify the neighbors in kneighbors to be k
    nn_index = index_data['nn_index']
    original_k = nn_index.n_neighbors
    nn_index.n_neighbors = k
    
    # Find k nearest neighbors
    distances, indices = nn_index.kneighbors(query_features_reduced)
    
    # Reset the original k value
    nn_index.n_neighbors = original_k
    
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        distance = distances[0][i]
        
        results.append({
            'idx': idx,
            'distance': distance,
            'image_path': index_data['image_paths'][idx],
            'image_name': index_data['image_names'][idx],
            'rotation_matrix': index_data['rotation_matrices'][idx]
        })
    
    return results

def detect_keypoints(image):
    """
    Detect keypoints in the image using ORB
    """
    # Convert image from PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale for keypoint detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def match_keypoints(descriptors1, descriptors2):
    """
    Match keypoints between two images
    """
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    if descriptors1 is not None and descriptors2 is not None:
        matches = bf.match(descriptors1, descriptors2)
        # Sort them in order of distance
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    return []

def verify_with_keypoints(query_image, candidate_paths, top_k=5):
    """
    Verify the best match using keypoint matching
    """
    # Detect keypoints in query image
    query_keypoints, query_descriptors = detect_keypoints(query_image)
    
    results = []
    for path in candidate_paths[:top_k]:
        # Load candidate image
        candidate_image = cv2.imread(path['image_path'])
        
        # Detect keypoints in candidate image
        candidate_keypoints, candidate_descriptors = detect_keypoints(candidate_image)
        
        # Match keypoints
        matches = match_keypoints(query_descriptors, candidate_descriptors)
        
        # Store result
        results.append({
            'path': path,
            'num_matches': len(matches),
            'matches': matches,
            'keypoints1': query_keypoints,
            'keypoints2': candidate_keypoints
        })
    
    # Sort by number of matches (descending)
    results.sort(key=lambda x: x['num_matches'], reverse=True)
    
    return results[0] if results else None

def display_results(test_image_path, match_result, keypoint_result=None, show_keypoints=True):
    """
    Display the test image and its closest match
    """
    plt.figure(figsize=(12, 6))
    
    # Load and display test image
    test_image = Image.open(test_image_path).convert('RGB')
    plt.subplot(1, 2, 1)
    plt.imshow(test_image)
    plt.title('Test Image')
    plt.axis('off')
    
    # Load and display matched image
    matched_image = Image.open(match_result['image_path']).convert('RGB')
    plt.subplot(1, 2, 2)
    plt.imshow(matched_image)
    plt.title(f"Matched Image: {match_result['image_name']}\nDistance: {match_result['distance']:.4f}")
    plt.axis('off')
    
    # Display rotation matrix
    rotation_matrix = match_result['rotation_matrix']
    matrix_str = f"Rotation Matrix:\n{rotation_matrix[0]}\n{rotation_matrix[1]}\n{rotation_matrix[2]}"
    plt.figtext(0.5, 0.01, matrix_str, ha='center', fontsize=10, 
                bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # Show keypoint matches if available and requested
    if show_keypoints and keypoint_result:
        plt.figure(figsize=(12, 6))
        
        # Convert PIL images to OpenCV format for drawing
        query_img = cv2.cvtColor(np.array(test_image), cv2.COLOR_RGB2BGR)
        match_img = cv2.cvtColor(np.array(matched_image), cv2.COLOR_RGB2BGR)
        
        # Draw keypoint matches
        match_img_with_kp = cv2.drawKeypoints(match_img, keypoint_result['keypoints2'], None, color=(0, 255, 0), flags=0)
        query_img_with_kp = cv2.drawKeypoints(query_img, keypoint_result['keypoints1'], None, color=(0, 255, 0), flags=0)
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(query_img_with_kp, cv2.COLOR_BGR2RGB))
        plt.title(f'Query Image with {len(keypoint_result["keypoints1"])} Keypoints')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(match_img_with_kp, cv2.COLOR_BGR2RGB))
        plt.title(f'Matched Image with {len(keypoint_result["keypoints2"])} Keypoints')
        plt.axis('off')
        
        plt.figtext(0.5, 0.01, f"Matching Keypoints: {keypoint_result['num_matches']}", ha='center', fontsize=12,
                    bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    plt.show()

def main(args):
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load the test image
    test_image = load_image(args.test_image, transform)
    
    # Get feature extractor
    feature_extractor, device = get_feature_extractor()
    logger.info(f"Using device: {device}")
    
    # Extract features from test image
    logger.info("Extracting features from test image")
    test_features = extract_features(feature_extractor, test_image, device)
    
    # Load index data for the specified object
    index_file = os.path.join(args.index_dir, f"{args.object_id}_index.pkl")
    if not os.path.exists(index_file):
        logger.error(f"Index file not found: {index_file}")
        return
    
    logger.info(f"Loading index data for object {args.object_id}")
    with open(index_file, 'rb') as f:
        index_data = pickle.load(f)
    
    # First, use CNN features to find most similar images
    if args.use_keypoints:
        logger.info("Finding top candidates using CNN features")
        top_candidates = find_k_nearest_images(test_features, index_data, k=args.top_k)
        
        # Then verify with keypoint matching
        logger.info("Verifying with keypoint matching")
        test_image_pil = Image.open(args.test_image).convert('RGB')
        keypoint_result = verify_with_keypoints(test_image_pil, top_candidates, top_k=args.top_k)
        
        if keypoint_result:
            match_result = keypoint_result['path']
            logger.info(f"Best match: {match_result['image_name']} with {keypoint_result['num_matches']} keypoint matches")
        else:
            # Fall back to CNN result if keypoint matching fails
            match_result = top_candidates[0]
            keypoint_result = None
            logger.info(f"Keypoint matching failed, using CNN match: {match_result['image_name']}")
    else:
        # Just use CNN features
        logger.info("Finding nearest image using CNN features")
        match_result = find_nearest_image(test_features, index_data)
        keypoint_result = None
        logger.info(f"Best match: {match_result['image_name']} with distance {match_result['distance']:.4f}")
    
    # Print the result
    print(f"Closest image: {match_result['image_name']}")
    print(f"Distance: {match_result['distance']:.4f}")
    print("Rotation matrix:")
    print(match_result['rotation_matrix'])
    
    # Display results if requested
    if args.display:
        display_results(args.test_image, match_result, keypoint_result, args.show_keypoints)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find the closest matching image and rotation matrix')
    parser.add_argument('--test_image', type=str, required=True,
                        help='Path to the test image')
    parser.add_argument('--object_id', type=str, required=True,
                        help='Object ID for the test image')
    parser.add_argument('--index_dir', type=str, default='indices',
                        help='Directory containing index files')
    parser.add_argument('--use_keypoints', action='store_true',
                        help='Use keypoint matching for verification')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top candidates to consider for keypoint verification')
    parser.add_argument('--display', action='store_true',
                        help='Display the matching results')
    parser.add_argument('--show_keypoints', action='store_true',
                        help='Show keypoint matches in the visualization')
    
    args = parser.parse_args()
    main(args)