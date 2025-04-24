import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import argparse
import time
import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_features(features):
    """
    L2 normalize features for cosine similarity
    """
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / (norm + 1e-10)  # Add small epsilon to avoid division by zero

def reduce_dimensions(features, n_components=128):
    """
    Reduce dimensions of features using PCA
    """
    logger.info(f"Reducing dimensions from {features.shape[1]} to {n_components}")
    pca = PCA(n_components=n_components)
    start_time = time.time()
    reduced_features = pca.fit_transform(features)
    logger.info(f"PCA completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
    return reduced_features, pca

def build_nn_index(features, metric='cosine'):
    """
    Build a nearest neighbors index for similarity search
    
    Args:
        features: Feature vectors
        metric: Distance metric to use ('cosine' or 'euclidean')
    """
    nn_index = NearestNeighbors(n_neighbors=1, metric=metric, algorithm='auto', n_jobs=-1)
    nn_index.fit(features)
    return nn_index

def load_rotation_matrices(file_path):
    """
    Load rotation matrices from file
    """
    matrices = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            # Split the line by spaces and convert to floats
            values = [float(val) for val in line.strip().split()]
            
            # Check if we have exactly 9 values for a 3x3 matrix
            if len(values) == 9:
                # Reshape into a 3x3 matrix
                matrix = np.array(values).reshape(3, 3)
                matrices.append(matrix)
            else:
                logger.warning(f"Skipping line with {len(values)} values instead of 9")
    
    return matrices

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all feature files
    if args.object_ids:
        feature_files = [f"{obj_id}_features.pkl" for obj_id in args.object_ids]
    else:
        feature_files = [f for f in os.listdir(args.feature_dir) if f.endswith('_features.pkl')]
    
    logger.info(f"Found {len(feature_files)} feature files")
    
    # Process each object's features
    for feature_file in tqdm.tqdm(feature_files):
        object_id = feature_file.split('_')[0]
        logger.info(f"Processing features for object {object_id}")
        
        # Skip if index already exists and --force is not set
        index_file = os.path.join(args.output_dir, f"{object_id}_index.pkl")
        if os.path.exists(index_file) and not args.force:
            logger.info(f"Index for object {object_id} already exists. Skipping...")
            continue
        
        # Load features
        feature_path = os.path.join(args.feature_dir, feature_file)
        with open(feature_path, 'rb') as f:
            feature_data = pickle.load(f)
        
        features = feature_data['features']
        image_paths = feature_data['image_paths']
        image_names = feature_data['image_names']
        
        logger.info(f"Loaded features with shape {features.shape}")
        
        # Normalize features
        features = normalize_features(features)
        
        # Reduce dimensions with PCA
        reduced_features, pca = reduce_dimensions(features, args.n_components)
        
        # Normalize again after PCA
        reduced_features = normalize_features(reduced_features)
        
        # Build nearest neighbors index
        logger.info(f"Building nearest neighbors index")
        nn_index = build_nn_index(reduced_features, metric='cosine')
        
        # Load rotation matrices for this object
        rotation_file = os.path.join(args.rotation_dir, f"{object_id}.txt")
        if os.path.exists(rotation_file):
            logger.info(f"Loading rotation matrices from {rotation_file}")
            rotation_matrices = load_rotation_matrices(rotation_file)
            logger.info(f"Loaded {len(rotation_matrices)} rotation matrices")
            
            # Verify the number of matrices matches the number of images
            if len(rotation_matrices) != len(image_names):
                logger.warning(f"Number of rotation matrices ({len(rotation_matrices)}) does not match "
                               f"number of images ({len(image_names)})!")
        else:
            logger.warning(f"Rotation matrix file not found: {rotation_file}")
            rotation_matrices = []
        
        # Save index and metadata
        logger.info(f"Saving index and metadata")
        with open(index_file, 'wb') as f:
            pickle.dump({
                'pca': pca,
                'nn_index': nn_index,
                'image_paths': image_paths,
                'image_names': image_names,
                'rotation_matrices': rotation_matrices,
                'object_id': object_id
            }, f)
        
        logger.info(f"Successfully created index for object {object_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build nearest neighbor indices for similarity search')
    parser.add_argument('--feature_dir', type=str, default='features',
                        help='Directory containing extracted features')
    parser.add_argument('--rotation_dir', type=str, required=True,
                        help='Directory containing rotation matrices')
    parser.add_argument('--output_dir', type=str, default='indices',
                        help='Directory to save indices')
    parser.add_argument('--object_ids', nargs='+', type=str, default=None,
                        help='Specific object IDs to process (if not specified, all will be processed)')
    parser.add_argument('--n_components', type=int, default=128,
                        help='Number of PCA components to use')
    parser.add_argument('--force', action='store_true',
                        help='Force rebuilding of indices even if they already exist')
    
    args = parser.parse_args()
    main(args)