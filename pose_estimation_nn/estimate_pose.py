import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from typing import List, Union

class PoseEstimationNet(nn.Module):
    def __init__(self):
        super(PoseEstimationNet, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove final classification layer
        self.backbone.fc = nn.Identity()
        
        # Add pose regression head
        self.pose_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 4)  # Output quaternion [w, x, y, z]
        )
        
    def forward(self, x):
        features = self.backbone(x)
        quaternion = self.pose_head(features)
        
        # Normalize to unit quaternion
        quaternion = quaternion / torch.norm(quaternion, dim=1, keepdim=True)
        
        return quaternion

class PoseEstimator:
    def __init__(self, model_path: str):
        """
        Initialize pose estimator
        
        Args:
            model_path: Path to trained model checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = PoseEstimationNet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Image preprocessing (same as training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def estimate_pose(self, rgb_image: np.ndarray, polygon_mask: List) -> Union[np.ndarray, None]:
        """
        Estimate pose from BGR image and polygon mask coordinates
        
        Args:
            rgb_image: BGR image from cv2.imread() as numpy array (H, W, 3)
            polygon_mask: Polygon coordinates as list [[x1,y1], [x2,y2], ...]
            
        Returns:
            3x3 rotation matrix as numpy array, or None if error
        """
        try:
            # Convert BGR to RGB (since training used RGB)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
            # Convert polygon points to numpy array
            polygon_points = np.array(polygon_mask, dtype=np.int32)
            
            # Create masked image using polygon
            masked_image = self._create_masked_image(rgb_image, polygon_points)
            
            # Crop to polygon bounding box
            cropped_image = self._crop_to_polygon_bbox(masked_image, polygon_points)
            
            # Check if cropped image is valid
            if cropped_image.size == 0 or cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                return None
            
            # Preprocess image
            input_tensor = self.transform(cropped_image).unsqueeze(0).to(self.device)
            
            # Predict quaternion
            with torch.no_grad():
                pred_quaternion = self.model(input_tensor)
                quaternion = pred_quaternion.cpu().numpy().flatten()
            
            # Convert quaternion to rotation matrix
            rotation_matrix = self._quaternion_to_rotation_matrix(quaternion)
            
            return rotation_matrix
            
        except Exception as e:
            return None
    
    def _create_masked_image(self, image: np.ndarray, polygon_points: np.ndarray) -> np.ndarray:
        """
        Create masked image: keep RGB inside polygon, black outside
        """
        # Create binary mask from polygon
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_points], 255)
        
        # Apply mask: keep original RGB inside polygon, black outside
        masked_image = image.copy()
        masked_image[mask == 0] = [0, 0, 0]  # Set background to black
        
        return masked_image
    
    def _crop_to_polygon_bbox(self, image: np.ndarray, polygon_points: np.ndarray) -> np.ndarray:
        """
        Crop image to bounding box of polygon with small padding
        """
        # Get bounding box of polygon
        x_coords = polygon_points[:, 0]
        y_coords = polygon_points[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Add small padding
        padding = 20
        h, w = image.shape[:2]
        
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Crop image
        cropped = image[y_min:y_max, x_min:x_max]
        
        return cropped
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = q
        
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        
        return R