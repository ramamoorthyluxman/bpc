import cv2
import numpy as np
import os
import re
from . import exceptions
import sys
from matplotlib import pyplot as plt

MINIMUM_MATCH_POINTS = 10
CONFIDENCE_THRESH = 0.05

def get_matches(img_a_gray, img_b_gray, num_keypoints=1000, threshold=0.7):
    '''Function to get matched keypoints from two images using ORB

    Args:
        img_a_gray (numpy array): of shape (H, W) representing grayscale image A
        img_b_gray (numpy array): of shape (H, W) representing grayscale image B
        num_keypoints (int): number of points to be matched (default=100)
        threshold (float): can be used to filter strong matches only. Lower the value, stronger the requirements and hence fewer matches.
    Returns:
        match_points_a (numpy array): of shape (n, 2) representing x,y pixel coordinates of image A keypoints
        match_points_b (numpy array): of shape (n, 2) representing x,y pixel coordianted of matched keypoints in image B
    '''

    num_keypoints = 3000
    threshold = 0.8
    
    orb = cv2.ORB_create(nfeatures=num_keypoints)
    kp_a, desc_a = orb.detectAndCompute(img_a_gray, None)
    kp_b, desc_b = orb.detectAndCompute(img_b_gray, None)

    print(f"Number of keypoints in image A: {len(kp_a)}")
    print(f"Number of keypoints in image B: {len(kp_b)}")

    if desc_a is None:
        print("Number of descriptors in image a is None. Aborting.")
        sys.exit()

    if desc_b is None:
        print("Number of descriptors in image b is None. Aborting.")
        sys.exit()

    
    dis_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_list = dis_matcher.knnMatch(desc_a, desc_b, k=2) # get the two nearest matches for each keypoint in image A

    # for each keypoint feature in image A, compare the distance of the two matched keypoints in image B
    # retain only if distance is less than a threshold 
    good_matches_list = []
    for match_1, match_2 in matches_list:
        if match_1.distance < threshold * match_2.distance:
            good_matches_list.append(match_1)
    
    #filter good matching keypoints 
    good_kp_a = []
    good_kp_b = []

    for match in good_matches_list:
        good_kp_a.append(kp_a[match.queryIdx].pt) # keypoint in image A
        good_kp_b.append(kp_b[match.trainIdx].pt) # matching keypoint in image B
    
    if len(good_kp_a) < MINIMUM_MATCH_POINTS:
        raise exceptions.NotEnoughMatchPointsError(len(good_kp_a), MINIMUM_MATCH_POINTS)
    
    # Draw matches
    img_matches = cv2.drawMatches(img_a_gray, kp_a, img_b_gray, kp_b, good_matches_list, None, flags=2)

    # Count and display the number of good matches on the image
    num_good_matches = len(good_matches_list)
    cv2.putText(img_matches, f"Good Matches: {num_good_matches}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    return np.array(good_kp_a), np.array(good_kp_b), img_matches


def calculate_homography(points_img_a, points_img_b):
    '''Function to calculate the homography matrix from point corresspondences using Direct Linear Transformation
        The resultant homography transforms points in image B into points in image A
        Homography H = [h1 h2 h3; 
                        h4 h5 h6;
                        h7 h8 h9]
        u, v ---> point in image A
        x, y ---> matched point in image B then,
        with n point correspondences the DLT equation is:
            A.h = 0
        where A = [-x1 -y1 -1 0 0 0 u1*x1 u1*y1 u1;
                   0 0 0 -x1 -y1 -1 v1*x1 v1*y1 v1;
                   ...............................;
                   ...............................;
                   -xn -yn -1 0 0 0 un*xn un*yn un;
                   0 0 0 -xn -yn -1 vn*xn vn*yn vn]
        This equation is then solved using SVD
        (At least 4 point correspondences are required to determine 8 unkwown parameters of homography matrix)
    Args:
        points_img_a (numpy array): of shape (n, 2) representing pixel coordinate points (u, v) in image A
        points_img_b (numpy array): of shape (n, 2) representing pixel coordinates (x, y) in image B
    
    Returns:
        h_mat: A (3, 3) numpy array of estimated homography
    '''
    # concatenate the two numpy points array to get 4 columns (u, v, x, y)
    points_a_and_b = np.concatenate((points_img_a, points_img_b), axis=1)
    A = []
    # fill the A matrix by looping through each row of points_a_and_b containing u, v, x, y
    # each row in the points_ab would fill two rows in the A matrix
    for u, v, x, y in points_a_and_b:
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    
    A = np.array(A)
    _, _, v_t = np.linalg.svd(A)

    # soltion is the last column of v which means the last row of its transpose v_t
    h_mat = v_t[-1, :].reshape(3,3)
    return h_mat

def transform_with_homography(h_mat, points_array):
    """Function to transform a set of points using the given homography matrix.
        Points are normalized after transformation with the last column which represents the scale
    
    Args:
        h_mat (numpy array): of shape (3, 3) representing the homography matrix
        points_array (numpy array): of shape (n, 2) represting n set of x, y pixel coordinates that are
            to be transformed
    """
    # add column of ones so that matrix multiplication with homography matrix is possible
    ones_col = np.ones((points_array.shape[0], 1))
    points_array = np.concatenate((points_array, ones_col), axis=1)
    transformed_points = np.matmul(h_mat, points_array.T)
    epsilon = 1e-7 # very small value to use it during normalization to avoid division by zero
    transformed_points = transformed_points / (transformed_points[2,:].reshape(1,-1) + epsilon)
    transformed_points = transformed_points[0:2,:].T
    return transformed_points


def compute_outliers(h_mat, points_img_a, points_img_b, threshold=3):
    '''Function to compute the error in the Homography matrix using the matching points in
        image A and image B
    
    Args:
        h_mat (numpy array): of shape (3, 3) representing the homography that transforms points in image B to points in image A
        points_img_a (numpy array): of shape (n, 2) representing pixel coordinate points (u, v) in image A
        points_img_b (numpy array): of shape (n, 2) representing pixel coordinates (x, y) in image B
        theshold (int): a number that represents the allowable euclidean distance (in pixels) between the transformed pixel coordinate from
            the image B to the matched pixel coordinate in image A, to be conisdered outliers
    
    Returns:
        error: a scalar float representing the error in the Homography matrix
    '''
    num_points = points_img_a.shape[0]
    outliers_count = 0

    # transform the match point in image B to image A using the homography
    points_img_b_hat = transform_with_homography(h_mat, points_img_b)
    
    # let x, y be coordinate representation of points in image A
    # let x_hat, y_hat be the coordinate representation of transformed points of image B with respect to image A
    x = points_img_a[:, 0]
    y = points_img_a[:, 1]
    x_hat = points_img_b_hat[:, 0]
    y_hat = points_img_b_hat[:, 1]
    euclid_dis = np.sqrt(np.power((x_hat - x), 2) + np.power((y_hat - y), 2)).reshape(-1)
    for dis in euclid_dis:
        if dis > threshold:
            outliers_count += 1
    return outliers_count


def compute_homography_ransac(matches_a, matches_b):
    """Function to estimate the best homography matrix using RANSAC on potentially matching
    points.
    
    Args:
        matches_a (numpy array): of shape (n, 2) representing the coordinates
            of possibly matching points in image A
        matches_b (numpy array): of shape (n, 2) representing the coordinates
            of possibly matching points in image B

    Returns:
        best_h_mat: A numpy array of shape (3, 3) representing the best homography
            matrix that transforms points in image B to points in image A
    """
    num_all_matches =  matches_a.shape[0]
    # RANSAC parameters
    SAMPLE_SIZE = 5 #number of point correspondances for estimation of Homgraphy
    SUCCESS_PROB = 0.995 #required probabilty of finding H with all samples being inliners 
    min_iterations = int(np.log(1.0 - SUCCESS_PROB)/np.log(1 - 0.5**SAMPLE_SIZE))
    
    # Let the initial error be large i.e consider all matched points as outliers
    lowest_outliers_count = num_all_matches
    best_h_mat = None
    best_i = 0 # just to know in which iteration the best h_mat was found

    for i in range(min_iterations):
        rand_ind = np.random.permutation(range(num_all_matches))[:SAMPLE_SIZE]
        h_mat = calculate_homography(matches_a[rand_ind], matches_b[rand_ind])
        outliers_count = compute_outliers(h_mat, matches_a, matches_b)
        if outliers_count < lowest_outliers_count:
            best_h_mat = h_mat
            lowest_outliers_count = outliers_count
            best_i = i
    best_confidence_obtained = int(100 - (100 * lowest_outliers_count / num_all_matches))
    if best_confidence_obtained < CONFIDENCE_THRESH:
        raise(exceptions.MatchesNotConfident(best_confidence_obtained))
    return best_h_mat


def get_corners_as_array(img_height, img_width):
    """Function to extract the corner points of an image from its width and height and arrange it in the form
        of a numpy array.
        
        The 4 corners are arranged as follows:
        corners = [top_left_x, top_left_y;
                   top_right_x, top_right_y;
                   bottom_right_x, bottom_right_y;
                   bottom_left_x, bottom_left_y]

    Args:
        img_height (str): height of the image
        img_width (str): width of the image
    
    Returns:
        corner_points_array (numpy array): of shape (4,2) representing for corners with x,y pixel coordinates
    """
    corners_array = np.array([[0, 0],
                            [img_width - 1, 0],
                            [img_width - 1, img_height - 1],
                            [0, img_height - 1]])
    return corners_array


def get_crop_points_horz(img_a_h, transfmd_corners_img_b):
    """Function to find the pixel corners in the horizontally stitched images to crop and remove the
        black space around.
    
    Args:
        img_a_h (int): the height of the pivot image that is image A
        transfmd_corners_img_b (numpy array): of shape (n, 2) representing the transformed corners of image B
            The corners need to be in the following sequence:
            corners = [top_left_x, top_left_y;
                   top_right_x, top_right_y;
                   bottom_right_x, bottom_right_y;
                   bottom_left_x, bottom_left_y]
    Returns:
        x_start (int): the x pixel-cordinate to start the crop on the stitched image
        y_start (int): the x pixel-cordinate to start the crop on the stitched image
        x_end (int): the x pixel-cordinate to end the crop on the stitched image
        y_end (int): the y pixel-cordinate to end the crop on the stitched image
    """
    # the four transformed corners of image B
    top_lft_x_hat, top_lft_y_hat = transfmd_corners_img_b[0, :]
    top_rht_x_hat, top_rht_y_hat = transfmd_corners_img_b[1, :]
    btm_rht_x_hat, btm_rht_y_hat = transfmd_corners_img_b[2, :]
    btm_lft_x_hat, btm_lft_y_hat = transfmd_corners_img_b[3, :]

    # initialize the crop points
    # since image A (on the left side) is used as pivot, x_start will always be zero
    x_start, y_start, x_end, y_end = (0, None, None, None)

    if (top_lft_y_hat > 0) and (top_lft_y_hat > top_rht_y_hat):
        y_start = top_lft_y_hat
    elif (top_rht_y_hat > 0) and (top_rht_y_hat > top_lft_y_hat):
        y_start = top_rht_y_hat
    else:
        y_start = 0
        
    if (btm_lft_y_hat < img_a_h - 1) and (btm_lft_y_hat < btm_rht_y_hat):
        y_end = btm_lft_y_hat
    elif (btm_rht_y_hat < img_a_h - 1) and (btm_rht_y_hat < btm_lft_y_hat):
        y_end = btm_rht_y_hat
    else:
        y_end = img_a_h - 1

    if (top_rht_x_hat < btm_rht_x_hat):
        x_end = top_rht_x_hat
    else:
        x_end = btm_rht_x_hat
    
    return int(x_start), int(y_start), int(x_end), int(y_end)


def get_crop_points_vert(img_a_w, transfmd_corners_img_b):
    """Function to find the pixel corners in the vertically stitched images to crop and remove the
        black space around.
    
    Args:
        img_a_h (int): the width of the pivot image that is image A
        transfmd_corners_img_b (numpy array): of shape (n, 2) representing the transformed corners of image B
            The corners need to be in the following sequence:
            corners = [top_left_x, top_left_y;
                   top_right_x, top_right_y;
                   bottom_right_x, bottom_right_y;
                   bottom_left_x, bottom_left_y]
    Returns:
        x_start (int): the x pixel-cordinate to start the crop on the stitched image
        y_start (int): the x pixel-cordinate to start the crop on the stitched image
        x_end (int): the x pixel-cordinate to end the crop on the stitched image
        y_end (int): the y pixel-cordinate to end the crop on the stitched image
    """
    # the four transformed corners of image B
    top_lft_x_hat, top_lft_y_hat = transfmd_corners_img_b[0, :]
    top_rht_x_hat, top_rht_y_hat = transfmd_corners_img_b[1, :]
    btm_rht_x_hat, btm_rht_y_hat = transfmd_corners_img_b[2, :]
    btm_lft_x_hat, btm_lft_y_hat = transfmd_corners_img_b[3, :]

    # initialize the crop points
    # since image A (on the top) is used as pivot, y_start will always be zero
    x_start, y_start, x_end, y_end = (None, 0, None, None)

    if (top_lft_x_hat > 0) and (top_lft_x_hat > btm_lft_x_hat):
        x_start = top_lft_x_hat
    elif (btm_lft_x_hat > 0) and (btm_lft_x_hat > top_lft_x_hat):
        x_start = btm_lft_x_hat
    else:
        x_start = 0
        
    if (top_rht_x_hat < img_a_w - 1) and (top_rht_x_hat < btm_rht_x_hat):
        x_end = top_rht_x_hat
    elif (btm_rht_x_hat < img_a_w - 1) and (btm_rht_x_hat < top_rht_x_hat):
        x_end = btm_rht_x_hat
    else:
        x_end = img_a_w - 1

    if (btm_lft_y_hat < btm_rht_y_hat):
        y_end = btm_lft_y_hat
    else:
        y_end = btm_rht_y_hat
    
    return int(x_start), int(y_start), int(x_end), int(y_end)


def get_crop_points(h_mat, img_a, img_b, stitch_direc):
    """Function to find the pixel corners to crop the stitched image such that the black space 
        in the stitched image is removed.
        The black space could be because either image B is not of the same dimensions as image A
        or image B is skewed after homographic transformation.
        Example: 
                  (Horizontal stitching)
                ____________                     _________________
                |           |                    |                |
                |           |__________          |                |
                |           |         /          |       A        |
                |     A     |   B    /           |________________|
                |           |       /                |          | 
                |           |______/                 |    B     |
                |___________|                        |          |
                                                     |__________|  <-imagine slant bottom edge
        
        This function returns the corner points to obtain the maximum area inside A and B combined and making
        sure the edges are straight (i.e horizontal and veritcal). 

    Args:
        h_mat (numpy array): of shape (3, 3) representing the homography from image B to image A
        img_a (numpy array): of shape (h, w, c) representing image A
        img_b (numpy array): of shape (h, w, c) representing image B
        stitch_direc (int): 0 when stitching vertically and 1 when stitching horizontally

    Returns:
        x_start (int): the x pixel-cordinate to start the crop on the stitched image
        y_start (int): the x pixel-cordinate to start the crop on the stitched image
        x_end (int): the x pixel-cordinate to end the crop on the stitched image
        y_end (int): the y pixel-cordinate to end the crop on the stitched image          
    """
    img_a_h, img_a_w, _ = img_a.shape
    img_b_h, img_b_w, _ = img_b.shape

    orig_corners_img_b = get_corners_as_array(img_b_h, img_b_w)
                
    transfmd_corners_img_b = transform_with_homography(h_mat, orig_corners_img_b)

    if stitch_direc == 1:
        x_start, y_start, x_end, y_end = get_crop_points_horz(img_a_w, transfmd_corners_img_b)
    # initialize the crop points
    x_start = None
    x_end = None
    y_start = None
    y_end = None

    if stitch_direc == 1: # 1 is horizontal
        x_start, y_start, x_end, y_end = get_crop_points_horz(img_a_h, transfmd_corners_img_b)
    else: # when stitching images in the vertical direction
        x_start, y_start, x_end, y_end = get_crop_points_vert(img_a_w, transfmd_corners_img_b)
    return x_start, y_start, x_end, y_end


def get_homography(img_a, img_b):
    """Function to stitch image B to image A in the mentioned direction

    Args:
        img_a (numpy array): of shape (H, W, C) with opencv representation of image A (i.e C: B,G,R)
        img_b (numpy array): of shape (H, W, C) with opencv representation of image B (i.e C: B,G,R)
        stitch_direc (int): 0 for vertical and 1 for horizontal stitching

    Returns:
        stitched_image (numpy array): stitched image with maximum content of image A and image B after cropping
            to remove the black space 
    """

    
    
    img_a_gray = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    img_b_gray = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    matches_a, matches_b, img_matches = get_matches(img_a_gray, img_b_gray, num_keypoints=1000, threshold=0.8)
    h_mat = compute_homography_ransac(matches_a, matches_b)
    
    return h_mat, img_matches

def get_overlap_mask(img_a, img_b, h_mat, stitch_direc):
    """
    Get a binary mask indicating the overlapping region between two images.

    Parameters:
    img_a (numpy.ndarray): The first image.
    img_b (numpy.ndarray): The second image.
    h_mat (numpy.ndarray): The homography matrix for transforming img_b to img_a's space.
    stitch_direc (int): The stitching direction (0 for vertical, 1 for horizontal).

    Returns:
    numpy.ndarray: A binary mask indicating the overlapping region.
    """

    # Warp img_b to img_a's space
    if stitch_direc == 0:
        canvas = cv2.warpPerspective(img_b, h_mat, (img_a.shape[1], img_a.shape[0] + img_b.shape[0]))
    else:
        canvas = cv2.warpPerspective(img_b, h_mat, (img_a.shape[1] + img_b.shape[1], img_a.shape[0]))

    # Create a binary mask where the pixels in the warped image are non-zero
    mask = np.where((canvas > 0).all(axis=2), 1, 0)

    return mask

def pad_to_match_height(img1, img2):
    """
    Pads img2 to match the height of img1 by adding black pixels.

    Parameters:
    img1 (numpy.ndarray): First image.
    img2 (numpy.ndarray): Second image.

    Returns:
    numpy.ndarray: Padded img2.
    """
    height1 = img1.shape[0]
    height2 = img2.shape[0]
    if height1 != height2:
        if height1 > height2:
            padding = ((0, height1 - height2), (0, 0), (0, 0))
            img2_padded = np.pad(img2, padding, mode='constant', constant_values=0)
            return img2_padded
        else:
            padding = ((0, height2 - height1), (0, 0), (0, 0))
            img1_padded = np.pad(img1, padding, mode='constant', constant_values=0)
            return img1_padded, img2
    return img1, img2

def get_corresponding_points_in_original_images_and_plot(h_mat, img_a, img_b, save_path=None):
    # Get the dimensions of img_a and img_b
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    # Define the corner points of img_b
    corners_b = np.array([
        [0, 0],
        [w_b, 0],
        [w_b, h_b],
        [0, h_b]
    ], dtype='float32')

    # Transform the corner points using the homography matrix
    corners_b_transformed = cv2.perspectiveTransform(np.array([corners_b]), h_mat)[0]

    # Find the bounding box of the transformed corners in img_a
    x_min = max(0, int(np.min(corners_b_transformed[:, 0])))
    x_max = min(w_a, int(np.max(corners_b_transformed[:, 0])))
    y_min = max(0, int(np.min(corners_b_transformed[:, 1])))
    y_max = min(h_a, int(np.max(corners_b_transformed[:, 1])))

    # Create a grid of (x, y) coordinates at every pixel in the bounding box
    x, y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    points_a = np.c_[x.ravel(), y.ravel()].astype('float32')

    # Transform the points in img_a to img_b's coordinate system
    h_mat_inv = np.linalg.inv(h_mat)
    points_b_transformed = cv2.perspectiveTransform(np.array([points_a]), h_mat_inv)[0]

    # Filter out points that fall outside img_b
    mask = (0 <= points_b_transformed[:, 0]) & (points_b_transformed[:, 0] < w_b) & \
           (0 <= points_b_transformed[:, 1]) & (points_b_transformed[:, 1] < h_b)
    overlap_indices_a = points_a[mask].astype(int)
    overlap_indices_b = points_b_transformed[mask].astype(int)

    # Convert images to RGBA
    img_a_copy = cv2.cvtColor(img_a, cv2.COLOR_BGR2BGRA)
    img_b_copy = cv2.cvtColor(img_b, cv2.COLOR_BGR2BGRA)

    # Create an overlay for img_a and img_b
    overlay_a = img_a_copy.copy()
    overlay_b = img_b_copy.copy()

    overlay_a[overlap_indices_a[:, 1], overlap_indices_a[:, 0]] = [0, 0, 255, 128]  # Blue with transparency
    overlay_b[overlap_indices_b[:, 1], overlap_indices_b[:, 0]] = [255, 0, 0, 128]  # Red with transparency

    # Blend the images with the overlays
    img_a_result = cv2.addWeighted(img_a_copy, 1, overlay_a, 0.5, 0)
    img_b_result = cv2.addWeighted(img_b_copy, 1, overlay_b, 0.5, 0)

    # Pad img_b_result to match the height of img_a_result
    img_a_padded, img_b_padded = pad_to_match_height(img_a_result, img_b_result)

    # Display the images side by side
    combined_img = np.hstack((img_a_padded, img_b_padded))

    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGRA2RGBA))
    plt.title('Overlapping Points: Red in img_a, Blue in img_b')
    plt.axis('off')
    plt.savefig(save_path)

    return overlap_indices_a, overlap_indices_b

def stitch_image_pair(img_a, img_b, h_mat, stitch_direc):
    """
    Stitch two images together using a homography matrix without blending.

    Parameters:
    img_a (numpy.ndarray): The first image.
    img_b (numpy.ndarray): The second image.
    h_mat (numpy.ndarray): The homography matrix for transforming img_b to img_a's space.
    stitch_direc (int): The stitching direction (0 for vertical, 1 for horizontal).

    Returns:
    numpy.ndarray: The stitched image without blending.
    """

    # Warp img_b to img_a's space
    if stitch_direc == 0:
        canvas = cv2.warpPerspective(img_b, h_mat, (img_a.shape[1], img_a.shape[0] + img_b.shape[0]))
        canvas[0:img_a.shape[0], :, :] = img_a[:, :, :]
    else:
        canvas = cv2.warpPerspective(img_b, h_mat, (img_a.shape[1] + img_b.shape[1], img_a.shape[0]))
        canvas[:, 0:img_a.shape[1], :] = img_a[:, :, :]

    # Crop the stitched image
    x_start, y_start, x_end, y_end = get_crop_points(h_mat, img_a, img_b, stitch_direc)
    stitched_img = canvas[y_start:y_end, x_start:x_end, :]

    return stitched_img



def alpha_blending(img_a, img_b, h_mat, stitch_direc):
    """
    Apply alpha blending to stitch two images together.

    Parameters:
    img_a (numpy.ndarray): The first image.
    img_b (numpy.ndarray): The second image.
    h_mat (numpy.ndarray): The homography matrix for transforming img_b to img_a's space.
    stitch_direc (int): The stitching direction (0 for vertical, 1 for horizontal).

    Returns:
    numpy.ndarray: The stitched image with alpha blending.
    """

    # Warp img_b to img_a's space
    if stitch_direc == 0:
        canvas = cv2.warpPerspective(img_b, h_mat, (img_a.shape[1], img_a.shape[0] + img_b.shape[0]))
        overlap = canvas[0:img_a.shape[0], :, :]
    else:
        canvas = cv2.warpPerspective(img_b, h_mat, (img_a.shape[1] + img_b.shape[1], img_a.shape[0]))
        overlap = canvas[:, 0:img_a.shape[1], :]

    # Get the mask indicating the overlapping region
    overlap_mask = get_overlap_mask(img_a, img_b, h_mat, stitch_direc)

    # Apply alpha blending on the overlapping region
    alpha = params.BLENDING_ALPHA
    blended_overlap = cv2.addWeighted(overlap, alpha, img_a, 1 - alpha, 0)

    # Replace the overlapping region in canvas with the blended overlap
    if stitch_direc == 0:
        canvas[0:img_a.shape[0], :, :] = blended_overlap
    else:
        canvas[:, 0:img_a.shape[1], :] = blended_overlap

    # Crop the stitched image
    x_start, y_start, x_end, y_end = get_crop_points(h_mat, img_a, img_b, stitch_direc)
    stitched_img = canvas[y_start:y_end, x_start:x_end, :]

    return stitched_img

def feather_blending(img_a, img_b, h_mat, stitch_direc):
    """
    Apply feather blending to stitch two images together.

    Parameters:
    img_a (numpy.ndarray): The first image.
    img_b (numpy.ndarray): The second image.
    h_mat (numpy.ndarray): The homography matrix for transforming img_b to img_a's space.
    stitch_direc (int): The stitching direction (0 for vertical, 1 for horizontal).

    Returns:
    numpy.ndarray: The stitched image with feather blending.
    """

    # Warp img_b to img_a's space
    if stitch_direc == 0:
        canvas = cv2.warpPerspective(img_b, h_mat, (img_a.shape[1], img_a.shape[0] + img_b.shape[0]))
        overlap = canvas[0:img_a.shape[0], :, :]
    else:
        canvas = cv2.warpPerspective(img_b, h_mat, (img_a.shape[1] + img_b.shape[1], img_a.shape[0]))
        overlap = canvas[:, 0:img_a.shape[1], :]

    # Get the mask indicating the overlapping region
    overlap_mask = get_overlap_mask(img_a, img_b, h_mat, stitch_direc)

    # Ensure overlap_mask is of type uint8 and has the same size as overlap
    overlap_mask = overlap_mask[0:overlap.shape[0], 0:overlap.shape[1]].astype(np.uint8)

    # Create an alpha mask with a linear gradient for the blending region
    alpha = cv2.GaussianBlur(overlap_mask, (31, 31), 0)

    # Expand the dimensions of alpha to match the number of channels in overlap
    alpha = np.dstack([alpha, alpha, alpha])

    # Manually compute the weighted sum of the input images
    blended_overlap = (overlap.astype(float) * alpha + img_a.astype(float) * (1 - alpha)).astype(np.uint8)

    # Replace the overlapping region in canvas with the blended overlap
    if stitch_direc == 0:
        canvas[0:img_a.shape[0], :, :] = blended_overlap
    else:
        canvas[:, 0:img_a.shape[1], :] = blended_overlap
        
    # Crop the stitched image
    x_start, y_start, x_end, y_end = get_crop_points(h_mat, img_a, img_b, stitch_direc)
    stitched_img = canvas[y_start:y_end, x_start:x_end, :]

    return stitched_img

def poisson_blending(img_a, img_b, h_mat, stitch_direc):
    """
    Apply Poisson blending to stitch two images together.

    Parameters:
    img_a (numpy.ndarray): The first image.
    img_b (numpy.ndarray): The second image.
    h_mat (numpy.ndarray): The homography matrix for transforming img_b to img_a's space.
    stitch_direc (int): The stitching direction (0 for vertical, 1 for horizontal).

    Returns:
    numpy.ndarray: The stitched image with Poisson blending.
    """

    # Warp img_b to img_a's space
    if stitch_direc == 0:
        canvas = cv2.warpPerspective(img_b, h_mat, (img_a.shape[1], img_a.shape[0] + img_b.shape[0]))
        overlap = canvas[0:img_a.shape[0], :, :]
    else:
        canvas = cv2.warpPerspective(img_b, h_mat, (img_a.shape[1] + img_b.shape[1], img_a.shape[0]))
        overlap = canvas[:, 0:img_a.shape[1], :]

    # Create a mask of the overlapping region
    mask = np.zeros_like(overlap, dtype=np.uint8)
    mask[overlap > 0] = 255

    # Compute the center of the mask
    center = (mask.shape[1] // 2, mask.shape[0] // 2)

    # Apply Poisson blending using cv2.seamlessClone()
    blended_overlap = cv2.seamlessClone(overlap, img_a, mask, center, cv2.NORMAL_CLONE)

    # Replace the overlapping region in canvas with the blended overlap
    if stitch_direc == 0:
        canvas[0:img_a.shape[0], :, :] = blended_overlap
    else:
        canvas[:, 0:img_a.shape[1], :] = blended_overlap
        
    # Crop the stitched image
    x_start, y_start, x_end, y_end = get_crop_points(h_mat, img_a, img_b, stitch_direc)
    stitched_img = canvas[y_start:y_end, x_start:x_end, :]

    return stitched_img

def multiband_blending(img_a, img_b, h_mat, stitch_direc, levels=6):
    # Convert images to float32 type for blending
    img_a = img_a.astype(np.float32)
    img_b = img_b.astype(np.float32)
    
    # Get dimensions of both images
    height_a, width_a = img_a.shape[:2]
    height_b, width_b = img_b.shape[:2]
    
    # Compute corners of img_b
    corners_b = np.array([
        [0, 0], [0, height_b - 1],
        [width_b - 1, height_b - 1], [width_b - 1, 0]
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    # Warp corners of img_b to see where they end up in img_a's coordinate system
    warped_corners_b = cv2.perspectiveTransform(corners_b, h_mat)
    
    # Include corners of img_a
    corners_a = np.array([
        [0, 0], [0, height_a - 1], [width_a - 1, height_a - 1], [width_a - 1, 0]
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    # Combine the arrays
    all_corners = np.vstack((warped_corners_b, corners_a))
    
    # Find the extents of both the transformed and original images
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Translation matrix to move the image in the positive axis direction
    translation_dist = [-xmin, -ymin]
    h_translation = np.array([
       [1, 0, translation_dist[0]],
       [0, 1, translation_dist[1]],
       [0, 0, 1]
    ], dtype=np.float32)
    
    # Output image size after stitching
    output_img_size = (xmax - xmin, ymax - ymin)
    
    # Warp both images to the new output image coordinate system
    warped_img_b = cv2.warpPerspective(img_b, h_translation.dot(h_mat), output_img_size)
    warped_img_a = cv2.warpPerspective(img_a, h_translation, output_img_size)

    # Convert images to 8-bit unsigned integer format before blending
    warped_img_a = cv2.convertScaleAbs(warped_img_a)
    warped_img_b = cv2.convertScaleAbs(warped_img_b)

    # Create masks for blending
    # Ensure masks are single-channel 8-bit unsigned integer
    mask_a = np.zeros(warped_img_a.shape[:2], dtype=np.uint8)  # Only use the spatial dimensions
    mask_b = np.zeros(warped_img_b.shape[:2], dtype=np.uint8)  # Only use the spatial dimensions
    mask_a[:, :] = 255  # Full coverage
    mask_b[:, :] = 255  # Full coverage

    # Initialize the multi-band blender and prepare the region
    blender = cv2.detail_MultiBandBlender(try_gpu=1)
    blender.prepare((xmin, ymin, xmax, ymax))

    # Feed the images and masks to the blender
    blender.feed(cv2.UMat(warped_img_a), cv2.UMat(mask_a), (0, 0))
    blender.feed(cv2.UMat(warped_img_b), cv2.UMat(mask_b), (0, 0))

    # Blend the images to create the final output
    result, result_mask = blender.blend(None, None)
    
    # If result is a cv2.UMat, convert it to a numpy array (only necessary if handling UMat explicitly)
    if isinstance(result, cv2.UMat):
        result = result.get()  # Convert UMat to NumPy array if needed

    # Crop the stitched image
    x_start, y_start, x_end, y_end = get_crop_points(h_mat, img_a, img_b, stitch_direc)
    stitched_img = result[y_start:y_end, x_start:x_end, :]

    return stitched_img

def stitch_image_pair_blended(img_a, img_b, h_mat, stitch_direc):
    if params.BLENDING_ALGO == 'alpha':
        return alpha_blending(img_a, img_b, h_mat, stitch_direc)
    elif params.BLENDING_ALGO == 'feather':
        return feather_blending(img_a, img_b, h_mat, stitch_direc)
    elif params.BLENDING_ALGO == 'poisson':
        return poisson_blending(img_a, img_b, h_mat, stitch_direc)
    elif params.BLENDING_ALGO == 'multiband':
        return multiband_blending(img_a, img_b, h_mat, stitch_direc)


def check_imgfile_validity(folder, filenames):
    """Function to check if the files in the given path are valid image files.
    
    Args:
        folder (str): path containing the image files
        filenames (list): a list of image filenames

    Returns:
        valid_files (bool): True if all the files are valid image files else False
        msg (str): Message that has to be displayed as error
    """
    for file in filenames:
        full_file_path = os.path.join(folder, file)
        regex = "([^\\s]+(\\.(?i:(jpe?g|png)))$)"
        p = re.compile(regex)

        if not os.path.isfile(full_file_path):
            return False, "File not found: " + full_file_path
        if not (re.search(p, file)):
            return False, "Invalid image file: " + file
    return True, None