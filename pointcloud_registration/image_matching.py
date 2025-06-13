import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

class ImageMatcher:
    def __init__(self, detector_type='SIFT', matcher_type='FLANN'):
        """
        Initialize the image matcher with specified detector and matcher types.
        
        Args:
            detector_type (str): 'SIFT', 'ORB', or 'AKAZE'
            matcher_type (str): 'FLANN' or 'BF' (Brute Force)
        """
        self.detector_type = detector_type
        self.matcher_type = matcher_type
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()
        
    def _create_detector(self):
        """Create feature detector based on specified type."""
        if self.detector_type == 'SIFT':
            return cv2.SIFT_create(nfeatures=100)
        elif self.detector_type == 'ORB':
            return cv2.ORB_create(nfeatures=5000)
        elif self.detector_type == 'AKAZE':
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")
    
    def _create_matcher(self):
        """Create feature matcher based on specified type."""
        if self.matcher_type == 'FLANN':
            if self.detector_type == 'SIFT':
                # FLANN parameters for SIFT
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                return cv2.FlannBasedMatcher(index_params, search_params)
            else:
                # FLANN parameters for ORB/AKAZE
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                  table_number=6,
                                  key_size=12,
                                  multi_probe_level=1)
                search_params = dict(checks=50)
                return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # Brute Force matcher
            if self.detector_type == 'SIFT':
                return cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            else:
                return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def load_images(self, test_path, ref_path):
        """Load and preprocess images."""
        self.test_img = cv2.imread(test_path)
        self.ref_img = cv2.imread(ref_path)
        
        if self.test_img is None:
            raise FileNotFoundError(f"Could not load test image: {test_path}")
        if self.ref_img is None:
            raise FileNotFoundError(f"Could not load reference image: {ref_path}")
        
        # Convert to grayscale for feature detection
        self.test_gray = cv2.cvtColor(self.test_img, cv2.COLOR_BGR2GRAY)
        self.ref_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
        
        print(f"Test image shape: {self.test_img.shape}")
        print(f"Reference image shape: {self.ref_img.shape}")
    
    def extract_features(self):
        """Extract keypoints and descriptors from both images."""
        print(f"Extracting features using {self.detector_type}...")
        
        # Detect keypoints and compute descriptors
        self.kp1, self.desc1 = self.detector.detectAndCompute(self.test_gray, None)
        self.kp2, self.desc2 = self.detector.detectAndCompute(self.ref_gray, None)
        
        print(f"Test image: {len(self.kp1)} keypoints")
        print(f"Reference image: {len(self.kp2)} keypoints")
        
        if len(self.kp1) < 4 or len(self.kp2) < 4:
            raise ValueError("Not enough keypoints detected for transformation estimation")
    
    def match_features(self):
        """Match features between the two images."""
        print(f"Matching features using {self.matcher_type}...")
        
        if self.matcher_type == 'FLANN':
            # FLANN matching
            matches = self.matcher.knnMatch(self.desc1, self.desc2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
        else:
            # Brute Force matching
            matches = self.matcher.match(self.desc1, self.desc2)
            good_matches = sorted(matches, key=lambda x: x.distance)
            # Keep only the best matches
            good_matches = good_matches[:int(len(matches) * 0.3)]
        
        self.good_matches = good_matches
        print(f"Found {len(good_matches)} good matches")
        
        if len(good_matches) < 4:
            raise ValueError("Not enough good matches for transformation estimation")
    
    def estimate_transformation(self):
        """Estimate homography transformation between images."""
        print("Estimating transformation...")
        
        # Extract matched point coordinates
        src_pts = np.float32([self.kp1[m.queryIdx].pt for m in self.good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in self.good_matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        self.homography, self.mask = cv2.findHomography(
            src_pts, dst_pts, 
            cv2.RANSAC, 
            ransacReprojThreshold=5.0
        )
        
        # Count inliers
        inliers = np.sum(self.mask)
        print(f"Homography found with {inliers} inliers out of {len(self.good_matches)} matches")
        
        if self.homography is None:
            raise ValueError("Could not estimate transformation")
        
        # Analyze transformation
        self._analyze_transformation()
    
    def _analyze_transformation(self):
        """Analyze the estimated transformation."""
        if self.homography is None:
            return
        
        # Decompose homography to understand transformation
        h = self.homography
        
        # Calculate scaling factors
        scale_x = np.sqrt(h[0,0]**2 + h[1,0]**2)
        scale_y = np.sqrt(h[0,1]**2 + h[1,1]**2)
        
        # Calculate rotation angle
        rotation = np.arctan2(h[1,0], h[0,0]) * 180 / np.pi
        
        # Calculate translation
        translation_x = h[0,2]
        translation_y = h[1,2]
        
        # Calculate skew
        skew = np.arctan2(h[0,1], h[1,1]) * 180 / np.pi - 90
        
        print("\n--- Transformation Analysis ---")
        print(f"Scale X: {scale_x:.3f}")
        print(f"Scale Y: {scale_y:.3f}")
        print(f"Rotation: {rotation:.2f} degrees")
        print(f"Translation X: {translation_x:.2f} pixels")
        print(f"Translation Y: {translation_y:.2f} pixels")
        print(f"Skew: {skew:.2f} degrees")
        
        # Store transformation parameters
        self.transform_params = {
            'scale_x': scale_x,
            'scale_y': scale_y,
            'rotation': rotation,
            'translation_x': translation_x,
            'translation_y': translation_y,
            'skew': skew,
            'homography': h
        }
    
    def visualize_results(self, save_path=None):
        """Visualize matching results and transformation."""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Matching and Transformation Analysis', fontsize=16)
        
        # 1. Show feature matches
        ax1 = axes[0, 0]
        matches_img = cv2.drawMatches(
            self.test_img, self.kp1,
            self.ref_img, self.kp2,
            self.good_matches[:50],  # Show top 50 matches
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        ax1.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Feature Matches ({len(self.good_matches)} total)')
        ax1.axis('off')
        
        # 2. Show inlier matches only
        ax2 = axes[0, 1]
        if hasattr(self, 'mask'):
            inlier_matches = [self.good_matches[i] for i in range(len(self.good_matches)) 
                            if self.mask[i]]
            inliers_img = cv2.drawMatches(
                self.test_img, self.kp1,
                self.ref_img, self.kp2,
                inlier_matches[:50],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            ax2.imshow(cv2.cvtColor(inliers_img, cv2.COLOR_BGR2RGB))
            ax2.set_title(f'Inlier Matches ({len(inlier_matches)} total)')
        ax2.axis('off')
        
        # 3. Show transformed test image
        ax3 = axes[1, 0]
        if hasattr(self, 'homography') and self.homography is not None:
            h, w, _ = self.ref_img.shape
            transformed = cv2.warpPerspective(self.test_img, self.homography, (w, h))
            ax3.imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
            ax3.set_title('Test Image Transformed to Reference')
        ax3.axis('off')
        
        # 4. Show transformation parameters
        ax4 = axes[1, 1]
        if hasattr(self, 'transform_params'):
            params = self.transform_params
            text = f"""Transformation Parameters:
            
Scale X: {params['scale_x']:.3f}
Scale Y: {params['scale_y']:.3f}
Rotation: {params['rotation']:.2f}°
Translation X: {params['translation_x']:.1f} px
Translation Y: {params['translation_y']:.1f} px
Skew: {params['skew']:.2f}°

Detector: {self.detector_type}
Matcher: {self.matcher_type}
Total Matches: {len(self.good_matches)}
Inliers: {np.sum(self.mask) if hasattr(self, 'mask') else 'N/A'}"""
            
            ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def save_transformation_matrix(self, output_path):
        """Save the transformation matrix to a file."""
        if hasattr(self, 'homography') and self.homography is not None:
            np.savetxt(output_path, self.homography, fmt='%.6f')
            print(f"Transformation matrix saved to: {output_path}")
    
    def process(self, test_path, ref_path, visualization_path=None, matrix_path=None):
        """Complete processing pipeline."""
        try:
            self.load_images(test_path, ref_path)
            self.extract_features()
            self.match_features()
            self.estimate_transformation()
            self.visualize_results(visualization_path)
            
            if matrix_path:
                self.save_transformation_matrix(matrix_path)
                
            return self.transform_params
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Feature matching and transformation estimation')
    parser.add_argument('test_image', help='Path to test image')
    parser.add_argument('ref_image', help='Path to reference image')
    parser.add_argument('--detector', choices=['SIFT', 'ORB', 'AKAZE'], 
                       default='SIFT', help='Feature detector type')
    parser.add_argument('--matcher', choices=['FLANN', 'BF'], 
                       default='FLANN', help='Matcher type')
    parser.add_argument('--save-viz', help='Path to save visualization')
    parser.add_argument('--save-matrix', help='Path to save transformation matrix')
    
    args = parser.parse_args()
    
    # Verify input files exist
    if not Path(args.test_image).exists():
        print(f"Error: Test image not found: {args.test_image}")
        sys.exit(1)
    if not Path(args.ref_image).exists():
        print(f"Error: Reference image not found: {args.ref_image}")
        sys.exit(1)
    
    # Create matcher and process images
    matcher = ImageMatcher(detector_type=args.detector, matcher_type=args.matcher)
    result = matcher.process(
        args.test_image, 
        args.ref_image,
        args.save_viz,
        args.save_matrix
    )
    
    if result:
        print("\nProcessing completed successfully!")
    else:
        print("Processing failed!")
        sys.exit(1)

# Example usage
if __name__ == "__main__":
    # If running directly with hardcoded paths (for testing)
    if len(sys.argv) == 1:
        print("Usage examples:")
        print("python script.py test.jpg ref.jpg")
        print("python script.py test.jpg ref.jpg --detector ORB --matcher BF")
        print("python script.py test.jpg ref.jpg --save-viz output.png --save-matrix transform.txt")
        
        # Uncomment and modify these lines for direct testing:
        test_path = "/home/rama/bpc_ws/bpc/pointcloud_registration/output/reference_images/ref_01_obj_000018_01_cam1_polygon.png"
        ref_path = "/home/rama/bpc_ws/bpc/pointcloud_registration/output/test_images/test_01_obj_000018_polygon.png"
        matcher = ImageMatcher()
        matcher.process(test_path, ref_path)
    else:
        main()