import os
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import time
import cv2

class PoseRegressor(nn.Module):
    def __init__(self, pretrained=False):
        """
        Neural network for pose regression
        
        Args:
            pretrained: Whether to use pretrained weights for the backbone
        """
        super(PoseRegressor, self).__init__()
        
        # Use ResNet50 as backbone
        resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Regression head for 9 values (flattened 3x3 rotation matrix)
        self.fc = nn.Linear(2048, 9)
    
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def orthogonalize_rotation_matrix(matrix):
    """
    Ensure the predicted matrix is a valid rotation matrix
    using SVD orthogonalization
    """
    # Reshape to 3x3 if flattened
    if matrix.size == 9:
        matrix = matrix.reshape(3, 3)
    
    # Perform SVD
    u, _, vh = np.linalg.svd(matrix, full_matrices=True)
    
    # Construct orthogonal matrix
    R = u @ vh
    
    # Ensure proper rotation (det=1)
    if np.linalg.det(R) < 0:
        u[:, -1] = -u[:, -1]
        R = u @ vh
    
    return R

def load_image(image_path, transform=None):
    """
    Load and preprocess an image
    """
    image = Image.open(image_path).convert('RGB')
    
    if transform:
        image = transform(image)
    
    return image

def predict_rotation(model, image, device):
    """
    Predict rotation matrix for an image
    """
    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get prediction
    with torch.no_grad():
        output = model(image)
    
    # Convert to numpy
    rotation_matrix = output.cpu().numpy()[0]
    
    # Ensure it's a valid rotation matrix
    rotation_matrix = orthogonalize_rotation_matrix(rotation_matrix)
    
    return rotation_matrix

def visualize_axes(image, rotation_matrix, axis_length=50, thickness=2):
    """
    Visualize coordinate axes based on rotation matrix
    """
    # Convert PIL image to numpy array for OpenCV
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Image center
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Define axis endpoints in camera coordinates
    # X-axis: Red, Y-axis: Green, Z-axis: Blue
    axes = np.array([
        [axis_length, 0, 0],  # X-axis
        [0, axis_length, 0],  # Y-axis
        [0, 0, axis_length]   # Z-axis
    ])
    
    # Project axes to image using rotation matrix
    for i, axis in enumerate(axes):
        # Apply rotation
        rotated_axis = rotation_matrix @ axis
        
        # Convert to image coordinates (perspective division)
        end_point = (
            int(center[0] + rotated_axis[0]),
            int(center[1] - rotated_axis[1])  # Y-axis is flipped in image coordinates
        )
        
        # Draw line
        color = [0, 0, 0]
        color[i] = 255  # RGB: R=0, G=1, B=2
        cv2.line(image, center, end_point, color, thickness)
    
    return image

def display_result(image_path, rotation_matrix):
    """
    Display the input image and visualize the rotation
    """
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    
    # Create visualization
    axis_image = visualize_axes(original_image.copy(), rotation_matrix)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(axis_image, cv2.COLOR_BGR2RGB))
    plt.title('Predicted Rotation')
    plt.axis('off')
    
    # Display rotation matrix
    matrix_str = f"Rotation Matrix:\n{rotation_matrix[0]}\n{rotation_matrix[1]}\n{rotation_matrix[2]}"
    plt.figtext(0.5, 0.01, matrix_str, ha='center', fontsize=10, 
                bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    
    # Save the figure if needed
    if args.save_visualization:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(args.output_dir, f"{base_name}_prediction.png")
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    if args.display:
        plt.show()

def main(args):
    # Create output directory if needed
    if args.save_visualization:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define image transform (same as validation transform during training)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load model
    model = PoseRegressor()
    model.to(device)
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path} (object ID: {checkpoint['object_id']})")
    
    # Process each test image
    for image_path in args.test_images:
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        print(f"Processing image: {image_path}")
        
        # Load and transform image
        image = load_image(image_path, transform)
        
        # Predict rotation
        start_time = time.time()
        rotation_matrix = predict_rotation(model, image, device)
        inference_time = time.time() - start_time
        
        # Print results
        print(f"Inference time: {inference_time*1000:.2f} ms")
        print("Predicted rotation matrix:")
        print(rotation_matrix)
        
        # Display results
        if args.display or args.save_visualization:
            display_result(image_path, rotation_matrix)
        
        # Save rotation matrix to file if requested
        if args.save_results:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(args.output_dir, f"{base_name}_rotation.txt")
            np.savetxt(save_path, rotation_matrix, fmt='%.6f')
            print(f"Rotation matrix saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict object pose from an image')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--test_images', nargs='+', required=True,
                        help='Paths to test images')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--display', action='store_true',
                        help='Display results')
    parser.add_argument('--save_visualization', action='store_true',
                        help='Save visualizations')
    parser.add_argument('--save_results', action='store_true',
                        help='Save rotation matrices to files')
    
    args = parser.parse_args()
    main(args)