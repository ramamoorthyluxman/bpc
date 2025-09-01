import cv2
import numpy as np
import torch
import matplotlib.cm as cm

# Note: You'll need to import these from your SuperGlue implementation
from models.matching import Matching
from models.utils import make_matching_plot
from models.utils import process_resize

class SuperGlueMatcher:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        config = {
            'superpoint': {
                'nms_radius': 8,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 100,
                'match_threshold': 0.4,
            }
        }
        
        self.matching = Matching(config).eval().to(device)
        self.device = device
        torch.set_grad_enabled(False)
    
    def crop_to_foreground(self, image):
        """
        Crop image to foreground (non-zero pixels) and return cropped image with offset info.
        
        Args:
            image: RGB mask image
            
        Returns:
            cropped_image: Cropped image
            crop_offset: (x_offset, y_offset) - offset of the crop in original image
            crop_info: dict with crop boundaries for debugging
        """
        # Convert to grayscale to find non-zero regions
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Find bounding box of non-zero pixels
        coords = np.argwhere(gray > 0)
        
        if len(coords) == 0:
            # If no foreground found, return original image
            return image, (0, 0), {'x_min': 0, 'y_min': 0, 'x_max': image.shape[1], 'y_max': image.shape[0]}
        
        # Get bounding box coordinates
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add small padding if possible
        padding = 10
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        y_max = min(image.shape[0], y_max + padding)
        x_max = min(image.shape[1], x_max + padding)
        
        # Crop the image
        cropped_image = image[y_min:y_max, x_min:x_max]
        crop_offset = (x_min, y_min)
        crop_info = {'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}
        
        return cropped_image, crop_offset, crop_info
    
    def map_keypoints_to_original(self, keypoints, crop_offset, scales):
        """
        Map keypoints from cropped image coordinates back to original image coordinates.
        
        Args:
            keypoints: Keypoints in cropped image coordinates
            crop_offset: (x_offset, y_offset) from cropping
            scales: Scaling factors from resizing
            
        Returns:
            keypoints_original: Keypoints in original image coordinates
        """
        # First, scale back from resized coordinates to cropped coordinates
        keypoints_cropped = keypoints * np.array([scales[0], scales[1]])
        
        # Then, add the crop offset to get original coordinates
        keypoints_original = keypoints_cropped + np.array([crop_offset[0], crop_offset[1]])
        
        return keypoints_original
    
    def validate_homography_matrix(self, h_mat):
        """Validate mathematical properties of homography matrix"""
        # Check determinant (should be positive, not too extreme)
        det = np.linalg.det(h_mat)
        if det <= 0 or det > 10 or det < 0.1:
            return False, f"Bad determinant: {det:.3f}"
        
        # Check condition number (well-conditioned matrix)
        cond_num = np.linalg.cond(h_mat)
        if cond_num > 1000:  # Threshold may need tuning
            return False, f"Poorly conditioned matrix: {cond_num:.2f}"
        
        # Check for reasonable scale factors using SVD
        U, s, Vt = np.linalg.svd(h_mat[:2, :2])  # Upper 2x2 for scale/rotation
        scale_ratio = s.max() / s.min()
        if scale_ratio > 5.0:  # Max 5x scaling difference
            return False, f"Extreme scaling: {scale_ratio:.2f}"
        
        return True, "Matrix properties OK"
    
    def validate_mask_alignment(self, img0, img1, h_mat, iou_threshold=0.4):
        """Validate alignment quality for mask images using IoU"""
        # Transform first mask to align with second
        warped_mask0 = cv2.warpPerspective(img0, h_mat, 
                                          (img1.shape[1], img1.shape[0]))
        
        # Convert to binary masks for IoU calculation
        mask0_bin = (warped_mask0 > 0).astype(np.uint8)
        mask1_bin = (img1 > 0).astype(np.uint8)
        
        # Calculate IoU (Intersection over Union)
        intersection = np.logical_and(mask0_bin, mask1_bin)
        union = np.logical_or(mask0_bin, mask1_bin)
        
        if np.sum(union) == 0:
            return False, "No union area - empty masks"
        
        iou = np.sum(intersection) / np.sum(union)
        
        if iou < iou_threshold:
            return False, f"Low IoU: {iou:.3f} < {iou_threshold}"
        
        return True, f"Good alignment: IoU={iou:.3f}"
    
    def validate_homography_combined(self, img0, img1, h_mat):
        """Combined validation using matrix condition and mask alignment"""
        validations = []
        
        # 1. Matrix properties validation
        # is_valid, msg = self.validate_homography_matrix(h_mat)
        # validations.append(("Matrix", is_valid, msg))
        
        # 2. Mask alignment validation
        is_valid, msg = self.validate_mask_alignment(img0, img1, h_mat)
        validations.append(("Alignment", is_valid, msg))
        
        # Both validations must pass
        overall_valid = all(valid for name, valid, msg in validations)
        
        # # print validation results
        for name, valid, msg in validations:
            status = "✓" if valid else "✗"
            # print(f"{status} {name}: {msg}")
        
        return overall_valid, validations

    def superglue(self, img0, img1):
        # Crop images to foreground
        img0_cropped, crop_offset0, crop_info0 = self.crop_to_foreground(img0)
        img1_cropped, crop_offset1, crop_info1 = self.crop_to_foreground(img1)
        
        # print(f"Image 0 crop offset: {crop_offset0}, crop info: {crop_info0}")
        # print(f"Image 1 crop offset: {crop_offset1}, crop info: {crop_info1}")
        
        # Prepare images - convert to grayscale tensors like the working script
        def image_to_tensor(image):
            # Convert to grayscale (SuperPoint expects 1 channel)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            w, h = gray.shape[1], gray.shape[0]
            w_new, h_new = process_resize(w, h, [640, 480])
            scales = (float(w) / float(w_new), float(h) / float(h_new))

            gray = cv2.resize(gray, (w_new, h_new)).astype('float32')

            # Normalize to [0,1] and convert to tensor
            inp = torch.from_numpy(gray/255.).float()[None, None].to(self.device)
            return inp, scales
        
        # Process cropped images
        frame0_tensor, scales0 = image_to_tensor(img0_cropped)
        frame1_tensor, scales1 = image_to_tensor(img1_cropped)

        # Perform the matching.
        pred = self.matching({'image0': frame0_tensor, 'image1': frame1_tensor})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Write the matches to disk.
        out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                        'matches': matches, 'match_confidence': conf}
        
        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        
        # Map keypoints back to original image coordinates
        mkpts0_org = self.map_keypoints_to_original(mkpts0, crop_offset0, scales0)
        mkpts1_org = self.map_keypoints_to_original(mkpts1, crop_offset1, scales1)
        
        # For visualization, create offset coordinates for second image
        mkpts1_org_w_offset = mkpts1_org.copy()
        mkpts1_org_w_offset[:, 0] += img0.shape[1]  # Offset for second image

        num_matches = None
        viz_image = None
        h_mat = None

        if mkpts0.shape[0] > 3 and mkpts1.shape[0] > 3:
            
            # Convert to float32 - using original coordinates for homography
            src_pts = np.float32(mkpts0_org).reshape(-1, 1, 2)
            dst_pts = np.float32(mkpts1_org).reshape(-1, 1, 2)
            
            # Compute homography with RANSAC (robust to outliers)
            h_mat, mask = cv2.findHomography(src_pts, dst_pts, 
                                        cv2.RANSAC, 
                                        ransacReprojThreshold=3.0)

            if h_mat is not None:
                
                # Use comprehensive validation instead of simple area check
                is_valid, validation_results = self.validate_homography_combined(img0, img1, h_mat)
                
                if not is_valid:
                    # print("Homography validation failed:")
                    for name, valid, msg in validation_results:
                        if not valid:
                            print(f"  - {name}: {msg}")
                    return None, None, None, None, None, None
                
                # print("Homography validation passed!")
                
                warped = cv2.warpPerspective(img0, h_mat, (img0.shape[1]*2, img0.shape[0]))
                
                num_matches = len(mkpts0)
                viz_image = np.hstack((img0, img1))

                # Draw matches using original image coordinates
                for pt0, pt1 in zip(mkpts0_org.astype(int), mkpts1_org_w_offset.astype(int)):
                    cv2.circle(viz_image, tuple(pt0), 2, (0,255,0), -1)
                    cv2.circle(viz_image, tuple(pt1), 2, (0,0,255), -1)
                    cv2.line(viz_image, tuple(pt0), tuple(pt1), (255,255,0), 2)
                
                viz_image = np.hstack((viz_image, warped))
            
                return num_matches, mconf, viz_image, h_mat, mkpts0_org.astype(int), mkpts1_org.astype(int)
            
        return None, None, None, None, None, None