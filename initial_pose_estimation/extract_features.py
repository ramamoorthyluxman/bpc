import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import argparse
import tqdm
import pickle

class PoseDataset(Dataset):
    def __init__(self, root_dir, object_id, transform=None):
        """
        Dataset for loading images from a specific object folder
        
        Args:
            root_dir: Root directory of the dataset
            object_id: Object ID as string (e.g., '000000')
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.object_id = object_id
        self.transform = transform
        
        # Get all image file names for this object
        self.image_dir = os.path.join(root_dir, object_id)
        self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                  if f.endswith('.png')])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'image_path': img_name,
            'image_name': self.image_files[idx]
        }

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

def extract_features(model, dataloader, device):
    """
    Extract features from images using the model
    """
    features = []
    image_paths = []
    image_names = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            images = batch['image'].to(device)
            batch_features = model(images)
            batch_features = batch_features.squeeze().cpu().numpy()
            
            features.append(batch_features)
            image_paths.extend(batch['image_path'])
            image_names.extend(batch['image_name'])
    
    # Concatenate all features
    features = np.vstack(features)
    
    return features, image_paths, image_names

def main(args):
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get feature extractor
    feature_extractor, device = get_feature_extractor()
    print(f"Using device: {device}")
    
    # Get all object folders from the dataset directory
    if args.object_ids:
        # Use the specified object IDs
        object_ids = args.object_ids
    else:
        # Find all object folders in the dataset directory
        object_ids = [d for d in os.listdir(args.data_dir) 
                     if os.path.isdir(os.path.join(args.data_dir, d))]
        # Filter out the rotation_matrices folder if it exists
        object_ids = [d for d in object_ids if d != 'rotation_matrices']
        object_ids.sort()
    
    print(f"Found {len(object_ids)} object directories: {object_ids}")
    
    for object_id in object_ids:
        print(f"Processing object {object_id}...")
        
        # Skip if features already exist and --force is not set
        output_file = os.path.join(args.output_dir, f"{object_id}_features.pkl")
        if os.path.exists(output_file) and not args.force:
            print(f"Features for object {object_id} already exist. Skipping...")
            continue
        
        # Create dataset and dataloader
        dataset = PoseDataset(args.data_dir, object_id, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        # Extract features
        features, image_paths, image_names = extract_features(feature_extractor, dataloader, device)
        
        # Save features
        feature_data = {
            'features': features,
            'image_paths': image_paths,
            'image_names': image_names,
            'object_id': object_id
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(feature_data, f)
        
        print(f"Saved features for object {object_id} with shape {features.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract CNN features from images')
    parser.add_argument('--data_dir', type=str, default='defined_poses', 
                        help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default='features',
                        help='Directory to save extracted features')
    parser.add_argument('--object_ids', nargs='+', type=str, default=None,
                        help='Specific object IDs to process (if not specified, all folders will be processed)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--force', action='store_true',
                        help='Force re-extraction of features even if they already exist')
    
    args = parser.parse_args()
    main(args)