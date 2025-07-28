import cv2
import numpy as np
import torch
import matplotlib.cm as cm

# Note: You'll need to import these from your SuperGlue implementation
from models.matching import Matching
from models.utils import make_matching_plot
from models.utils import process_resize
from models.utils import compute_homography_ransac

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
                'sinkhorn_iterations': 30,
                'match_threshold': 0.4,
            }
        }
        
        self.matching = Matching(config).eval().to(device)
        self.device = device
        torch.set_grad_enabled(False)
    
    def superglue(self, img0, img1):

       

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
        
        frame0_tensor, scales = image_to_tensor(img0)
        frame1_tensor, scales = image_to_tensor(img1)


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

        
        
        mkpts0_org = mkpts0 * np.array([scales[0], scales[1]])
        mkpts1_org = mkpts1 * np.array([scales[0], scales[1]])

        mkpts1_org_w_offset = mkpts1 * np.array([scales[0], scales[1]])
        mkpts1_org_w_offset[:, 0] += img0.shape[1]  # Offset for second image

        

        num_matches = None
        viz_image = None


        if mkpts1_org.shape[0] > 3:
            
            # h_mat = compute_homography_ransac(mkpts0_org.astype(int), mkpts1_org.astype(int))
            # Convert to float32
            src_pts = np.float32(mkpts0_org).reshape(-1, 1, 2)
            dst_pts = np.float32(mkpts1_org_w_offset).reshape(-1, 1, 2)
            
            # Compute homography with RANSAC (robust to outliers)
            h_mat, mask = cv2.findHomography(src_pts, dst_pts, 
                                        cv2.RANSAC, 
                                        ransacReprojThreshold=3.0)

            
            if h_mat is not None:
                
                warped = cv2.warpPerspective(img0, h_mat, (img0.shape[1]*2, img0.shape[0]))
                # Check actual image content preservation
                area_orig = np.count_nonzero(img1)
                area_trans = np.count_nonzero(warped)

                if area_trans / area_orig > 4 or area_trans / area_orig < 0.6:
                    print(f"Extreme content change: {area_trans / area_orig}")
                    return None, None, None, None, None, None
                
                num_matches = len(mkpts0)
                viz_image = np.hstack((img0, img1))


                for pt0, pt1 in zip(mkpts0_org.astype(int), mkpts1_org_w_offset.astype(int)):
                    cv2.circle(viz_image, tuple(pt0), 2, (0,255,0), -1)
                    cv2.circle(viz_image, tuple(pt1), 2, (0,0,255), -1)
                    cv2.line(viz_image, tuple(pt0), tuple(pt1), (255,255,0), 2)
                
                viz_image = np.hstack((viz_image, warped))
            
                return num_matches, mconf, viz_image, h_mat, mkpts0_org.astype(int), mkpts1_org.astype(int)
            
        return None, None, None, None, None, None 