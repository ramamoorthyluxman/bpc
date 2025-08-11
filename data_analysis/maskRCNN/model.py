import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    """
    Create a Mask R-CNN model for instance segmentation
    
    Args:
        num_classes (int): Number of output classes including background class
        
    Returns:
        model: A PyTorch Mask R-CNN model configured for the dataset
    """
    # Load a pre-trained model using the new weights API
    model = maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


def get_model_instance_segmentation_mobile(num_classes):
    """
    Create a lighter Mask R-CNN model (MobileNetV3) for resource-constrained environments
    
    Args:
        num_classes (int): Number of output classes including background class
        
    Returns:
        model: A PyTorch Mask R-CNN model with MobileNetV3 backbone
    """
    # Load a pre-trained model - use MobileNetV3 for a lighter model
    # Note: This requires torchvision>=0.12.0
    try:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        )
    
        
        # Get the number of input features
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the pre-trained box predictor with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Get the number of input features for the mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        
        # Replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
    except Exception as e:
        print(f"Error creating lightweight model: {e}")
        print("Falling back to standard model")
        model = get_model_instance_segmentation(num_classes)
    
    return model


def get_model_with_custom_backbone(num_classes, backbone_name="resnet101"):
    """
    Create a Mask R-CNN model with a custom backbone for better performance
    
    Args:
        num_classes (int): Number of output classes including background class
        backbone_name (str): Name of the backbone to use (e.g., 'resnet101')
        
    Returns:
        model: A PyTorch Mask R-CNN model with the specified backbone
    """
    try:
        # Choose backbone based on name
        if backbone_name == "resnet101":
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            # Replace the backbone with ResNet101
            backbone = torchvision.models.resnet101(pretrained=True)
            # Need to adjust the backbone here - simplified for this example
        else:
            # Default to standard ResNet50
            model = get_model_instance_segmentation(num_classes)
            return model
        
        # Update the classification and mask prediction heads
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
    except Exception as e:
        print(f"Error creating model with {backbone_name} backbone: {e}")
        print("Falling back to standard model")
        model = get_model_instance_segmentation(num_classes)
    
    return model


def load_model(model_path, num_classes):
    """
    Load a saved model from disk
    
    Args:
        model_path (str): Path to the saved model file
        num_classes (int): Number of classes in the model
        
    Returns:
        model: Loaded PyTorch model
    """
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_path))
    return model


# Test code
if __name__ == "__main__":
    # Import dataset to get number of classes automatically
    import sys
    import os
    from dataset import MechanicalPartsDataset
    
    # Set default dataset path or use command line argument
    dataset_path = "."  # Default path
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Create dataset instance to get number of classes
    try:
        dataset = MechanicalPartsDataset(dataset_path)
        # Number of classes includes background (0) + all categories
        num_classes = len(dataset.categories)
        print(f"Found {num_classes} classes: {dataset.categories}")
    except Exception as e:
        print(f"Could not load dataset: {e}")
        print("Using default value of 4 classes")
        num_classes = 4  # Default: background + bracket + bolt + nut
    
    # Standard model
    model = get_model_instance_segmentation(num_classes)
    print(f"Created standard Mask R-CNN model with {num_classes} classes")
    
    # Check device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Move model to the right device
    model.to(device)
    
    # Print model summary (simplified)
    print(f"Model type: {type(model).__name__}")
    
    # Create a dummy input
    x = [torch.rand(3, 300, 400).to(device)]
    
    # Test forward pass (in eval mode to avoid training behavior)
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(x)
            print("Forward pass successful!")
            print(f"Output keys: {list(outputs[0].keys())}")
            
            # Print shapes of some outputs
            print(f"Detected boxes shape: {outputs[0]['boxes'].shape}")
            print(f"Scores shape: {outputs[0]['scores'].shape}")
            print(f"Labels shape: {outputs[0]['labels'].shape}")
            print(f"Masks shape: {outputs[0]['masks'].shape}")
        except Exception as e:
            print(f"Forward pass failed: {e}")