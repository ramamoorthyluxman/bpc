import cv2
import numpy as np

def decompose_homography_to_pose(H, K1, K2):
    """
    Decompose homography to recover 6DOF pose transformation
    
    Args:
        H: 3x3 homography matrix
        K1: 3x3 intrinsic matrix of camera 1
        K2: 3x3 intrinsic matrix of camera 2
    
    Returns:
        List of possible (R, t, n) solutions
    """
    # Normalize homography
    H_norm = np.linalg.inv(K2) @ H @ K1
    
    # Decompose using OpenCV
    num_solutions, Rs, ts, normals = cv2.decomposeHomographyMat(H_norm, np.eye(3))
    
    solutions = []
    for i in range(num_solutions):
        R = Rs[i]
        t = ts[i].flatten()
        n = normals[i].flatten()
        solutions.append((R, t, n))
    
    return solutions

def select_valid_solution(solutions, test_points_3d, K1, K2):
    """
    Select physically valid solution using test points
    
    Args:
        solutions: List of (R, t, n) tuples
        test_points_3d: 3D points on the object plane (Nx3)
        K1, K2: Camera intrinsic matrices
    
    Returns:
        Best (R, t, n) solution
    """
    best_solution = None
    max_positive_depth = 0
    
    for R, t, n in solutions:
        # Check if points have positive depth in both cameras
        points_cam1 = test_points_3d  # Points in camera 1 frame
        points_cam2 = (R @ points_cam1.T + t.reshape(-1, 1)).T
        
        # Count points with positive Z (in front of camera)
        positive_cam1 = np.sum(points_cam1[:, 2] > 0)
        positive_cam2 = np.sum(points_cam2[:, 2] > 0)
        total_positive = positive_cam1 + positive_cam2
        
        if total_positive > max_positive_depth:
            max_positive_depth = total_positive
            best_solution = (R, t, n)
    
    return best_solution

def pose_from_homography_complete(H, K1, K2, object_points_2d, world_height=0):
    """
    Complete pipeline: homography -> 6DOF pose
    
    Args:
        H: Homography matrix
        K1, K2: Camera intrinsics
        object_points_2d: 2D points on object (for validation)
        world_height: Z-coordinate of object plane (default: 0)
    
    Returns:
        R, t: Rotation matrix and translation vector
    """
    # Decompose homography
    solutions = decompose_homography_to_pose(H, K1, K2)
    
    if len(solutions) == 0:
        raise ValueError("No valid solutions found")
    
    # Create 3D test points (assuming planar object at Z=world_height)
    test_points_3d = np.column_stack([
        object_points_2d,
        np.full(len(object_points_2d), world_height)
    ])
    
    # Select best solution
    R, t, n = select_valid_solution(solutions, test_points_3d, K1, K2)
    
    if R is None:
        # Fallback: return first solution
        R, t, n = solutions[0]
    
    return R, t, n

def compute_pose_error(R1, t1, R2, t2):
    """
    Compute angular and translation error between poses
    """
    # Rotation error (angle between rotation matrices)
    R_diff = R2 @ R1.T
    angle_error = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
    angle_error_deg = np.degrees(angle_error)
    
    # Translation error
    t_error = np.linalg.norm(t2 - t1)
    
    return angle_error_deg, t_error

# Example usage
if __name__ == "__main__":
    # Example camera intrinsics (you need real calibrated values)
    K1 = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=np.float64)
    
    K2 = np.array([
        [850, 0, 340],
        [0, 850, 250], 
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Simulate homography from known pose (for testing)
    # Real rotation and translation
    R_true = cv2.Rodrigues(np.array([0.1, 0.2, 0.3]))[0]
    t_true = np.array([0.1, 0.05, 0.2])
    n_true = np.array([0, 0, 1])  # Object plane normal
    d_true = 1.0  # Distance to plane
    
    # Construct homography
    H_true = K2 @ (R_true + np.outer(t_true, n_true) / d_true) @ np.linalg.inv(K1)
    
    print("Ground truth pose:")
    print(f"Rotation (Rodrigues): {cv2.Rodrigues(R_true)[0].flatten()}")
    print(f"Translation: {t_true}")
    
    # Object points on plane (Z=0)
    object_points_2d = np.array([
        [0.0, 0.0],
        [0.1, 0.0], 
        [0.1, 0.1],
        [0.0, 0.1]
    ])
    
    try:
        # Recover pose from homography
        R_recovered, t_recovered, n_recovered = pose_from_homography_complete(
            H_true, K1, K2, object_points_2d
        )
        
        print("\nRecovered pose:")
        print(f"Rotation (Rodrigues): {cv2.Rodrigues(R_recovered)[0].flatten()}")
        print(f"Translation: {t_recovered}")
        print(f"Plane normal: {n_recovered}")
        
        # Compute errors
        angle_err, trans_err = compute_pose_error(R_true, t_true, R_recovered, t_recovered)
        print(f"\nErrors:")
        print(f"Rotation error: {angle_err:.2f} degrees")
        print(f"Translation error: {trans_err:.4f} units")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*50)
    print("REQUIREMENTS FOR HOMOGRAPHY -> 6DOF:")
    print("1. Camera intrinsic matrices K1, K2")
    print("2. Homography matrix H")
    print("3. Knowledge of object plane structure")
    print("4. Method to resolve ambiguity (usually 2-4 solutions)")
    print("="*50)