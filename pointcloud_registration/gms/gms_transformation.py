import numpy as np
from enum import Enum
import time
import cv2
from cv2.xfeatures2d import matchGMS
import math
import random

class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2
    COLOR_CODED_POINTS_X = 3
    COLOR_CODED_POINTS_Y = 4
    COLOR_CODED_POINTS_XpY = 5


def draw_matches(src1, src2, kp1, kp2, matches, drawing_type):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    if drawing_type == DrawingType.ONLY_LINES:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))

    elif drawing_type == DrawingType.LINES_AND_POINTS:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

    elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))

            if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                colormap_idx = int(left[0] * 256. / src1.shape[1])  # x-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                colormap_idx = int(left[1] * 256. / src1.shape[0])  # y-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                colormap_idx = int((left[0] - src1.shape[1]*.5 + left[1] - src1.shape[0]*.5) * 256. / (src1.shape[0]*.5 + src1.shape[1]*.5))  # manhattan gradient

            color = tuple(map(int, _colormap[colormap_idx, 0, :]))
            cv2.circle(output, tuple(map(int, left)), 1, color, 2)
            cv2.circle(output, tuple(map(int, right)), 1, color, 2)
    return output

def homography(pairs):
    rows = []
    for i in range(pairs.shape[0]):
        p1 = np.append(pairs[i][0:2], 1)
        p2 = np.append(pairs[i][2:4], 1)
        row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
        row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
        rows.append(row1)
        rows.append(row2)
    rows = np.array(rows)
    U, s, V = np.linalg.svd(rows)
    H = V[-1].reshape(3, 3)
    H = H/H[2, 2] # standardize to let w*H[2,2] = 1
    return H

def random_point(matches, k=4):
    idx = random.sample(range(len(matches)), k)
    point = [matches[i] for i in idx ]
    return np.array(point)

def get_error(points, H):
    num_points = len(points)
    all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
    all_p2 = points[:, 2:4]
    estimate_p2 = np.zeros((num_points, 2))
    for i in range(num_points):
        temp = np.dot(H, all_p1[i])
        estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
    # Compute error
    errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

    return errors

def ransac(matches, threshold, iters):
    num_best_inliers = 0
    
    for i in range(iters):
        points = random_point(matches)
        H = homography(points)
        
        #  avoid dividing by zero 
        if np.linalg.matrix_rank(H) < 3:
            continue
            
        errors = get_error(matches, H)
        idx = np.where(errors < threshold)[0]
        inliers = matches[idx]

        num_inliers = len(inliers)
        if num_inliers > num_best_inliers:
            best_inliers = inliers.copy()
            num_best_inliers = num_inliers
            best_H = H.copy()
            
    print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
    return best_inliers, best_H

def convert_matches_to_points(matches, kp1, kp2):
    """Convert cv2.DMatch objects to coordinate pairs"""
    points = []
    for match in matches:
        # Get coordinates from keypoints using the indices in DMatch
        pt2 = kp1[match.queryIdx].pt  # (x1, y1)
        pt1 = kp2[match.trainIdx].pt  # (x2, y2)
        points.append([pt1[0], pt1[1], pt2[0], pt2[1]])  # [x1, y1, x2, y2]
    return np.array(points)

def get_transformation(gms_matches, kp1, kp2):
    # Convert DMatch objects to coordinate pairs
    match_points = convert_matches_to_points(gms_matches, kp1, kp2)
    inliers, H = ransac(match_points, 0.6, 2000)
    return inliers, H

def gms_transformation(img1, img2):
    orb = cv2.ORB_create(10000)
    orb.setFastThreshold(0)

    

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_all = matcher.match(des1, des2)

    matches_gms = matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches_all, withScale=True, withRotation=True, thresholdFactor=16.0)

    print('Found', len(matches_gms), 'matches')

    matches_image = draw_matches(img1, img2, kp1, kp2, matches_gms, DrawingType.ONLY_LINES)

    inliers, H = get_transformation(matches_gms, kp1, kp2)

    transformed_image = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    
    return matches_image, matches_gms, H, transformed_image, inliers
    # return matches_image, None, None, transformed_image, None
    
if __name__ == '__main__':
    img1 = cv2.imread("/home/rama/bpc_ws/bpc/pointcloud_registration/output/test_images/test_05_obj_000018_polygon.png")
    img2 = cv2.imread("/home/rama/bpc_ws/bpc/pointcloud_registration/output/reference_images/ref_01_obj_000018_01_cam1_polygon.png")

    matches_image, matches_gms, H, transformed_src2, inliers = gms_transformation(img1, img2)

    cv2.imwrite("matches.png", matches_image)

    if len(inliers)>6:
        # Visualize side by side: src1, src2, transformed src2
        h = max(img1.shape[0], img2.shape[0], transformed_src2.shape[0])
        w_total = img1.shape[1] + img2.shape[1] + transformed_src2.shape[1]
        comparison = np.zeros((h, w_total, 3), dtype=np.uint8)
        
        comparison[0:img1.shape[0], 0:img1.shape[1]] = img1
        comparison[0:img2.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
        comparison[0:transformed_src2.shape[0], img1.shape[1]+img2.shape[1]:] = transformed_src2
        
        cv2.imwrite("transformation_result.png", comparison)
        cv2.waitKey(0)