import numpy as np
from scipy.linalg import svd

def compute_rigid_transform(points_A, points_B):
    """
    Find the rigid transformation (R, t) that best aligns points_A to points_B
    such that: points_B ≈ R @ points_A + t
    
    Args:
        points_A: (N, 3) array of source points
        points_B: (N, 3) array of target points
    
    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        rmse: root mean square error of the transformation
    """
    
    # Convert to numpy arrays
    A = np.array(points_A)
    B = np.array(points_B)
    
    assert A.shape == B.shape, "Point sets must have the same shape"
    assert A.shape[1] == 3, "Points must be 3D"
    assert A.shape[0] >= 4, "Need at least 4 point correspondences"
    
    # Step 1: Center both point sets
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    
    # Step 2: Compute cross-covariance matrix
    H = A_centered.T @ B_centered
    
    # Step 3: Perform SVD
    U, S, Vt = svd(H)
    
    # Step 4: Extract rotation matrix
    R = Vt.T @ U.T
    
    # Handle reflection case (ensure proper rotation)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Step 5: Compute translation
    t = centroid_B - R @ centroid_A
    
    # Compute RMSE
    transformed_A = (R @ A.T).T + t
    rmse = np.sqrt(np.mean(np.sum((transformed_A - B)**2, axis=1)))
    
    return R, t, rmse

def apply_transform(points, R, t):
    """Apply rigid transformation to points"""
    return (R @ points.T).T + t

# Example usage and test
if __name__ == "__main__":
    # Generate test data
    np.random.seed(42)
    
    # Original points
    original_points = np.random.rand(6, 3) * 10
    
    # Create a known transformation
    true_R = np.array([[0.8660, -0.5000, 0.0000],
                       [0.5000,  0.8660, 0.0000],
                       [0.0000,  0.0000, 1.0000]])  # 30° rotation around z-axis
    true_t = np.array([2.0, 3.0, 1.0])
    
    # Transform the points
    transformed_points = apply_transform(original_points, true_R, true_t)
    
    # Add small amount of noise to simulate real-world conditions
    noise = np.random.normal(0, 0.01, transformed_points.shape)
    transformed_points += noise
    
    # Solve for transformation
    R_est, t_est, rmse = compute_rigid_transform(original_points, transformed_points)
    
    # Print results
    print("True rotation matrix:")
    print(true_R)
    print("\nEstimated rotation matrix:")
    print(R_est)
    print("\nTrue translation:")
    print(true_t)
    print("\nEstimated translation:")
    print(t_est)
    print(f"\nRMSE: {rmse:.6f}")
    
    # Verify the transformation
    test_transformed = apply_transform(original_points, R_est, t_est)
    print(f"\nMax error after transformation: {np.max(np.abs(test_transformed - transformed_points)):.6f}")