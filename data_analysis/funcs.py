import cv2
import numpy as np
import os
from sympy import intersection
import trimesh
from scipy.spatial.transform import Rotation 
from scipy.optimize import minimize
from itertools import product
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def create_polygon_mask(image, polygon_coords):
    """Create binary mask from polygon coordinates."""
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array(polygon_coords, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask


# Alternative approach using a depth buffer for occlusion handling
def create_polygon_from_mesh(mesh, cam_K, R, t, img_shape, simplify_factor=0.001):
    # Get mesh faces and vertices
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    # Transform vertices to camera coordinates
    vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    camera_coords = np.dot(vertices_homogeneous, np.vstack([
        np.hstack([R, t.reshape(3, 1)]),
        [0, 0, 0, 1]
    ]).T)
    
    # Project to image plane
    projected_points = np.dot(camera_coords[:, :3], cam_K.T)
    projected_points = projected_points[:, :2] / projected_points[:, 2:3]
    
    # Create an empty mask
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    
    # Filter visible faces (those with all vertices in front of camera)
    visible_faces = []
    for face in faces:
        if all(camera_coords[v, 2] > 0 for v in face):
            visible_faces.append(face)
    
    # Draw the visible faces on the mask
    for face in visible_faces:
        pts = projected_points[face].astype(np.int32)
        # Check if points are within image boundaries
        if np.all((pts[:, 0] >= 0) & (pts[:, 0] < img_shape[1]) & 
                  (pts[:, 1] >= 0) & (pts[:, 1] < img_shape[0])):
            cv2.fillPoly(mask, [pts], 255)
    
    # Find contours on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour
    max_contour = max(contours, key=cv2.contourArea)
    
    # Simplify the contour
    epsilon = simplify_factor * cv2.arcLength(max_contour, True)
    approx_polygon = cv2.approxPolyDP(max_contour, epsilon, True)
    
    # Convert to the format needed for the annotation
    polygon = approx_polygon.reshape(-1, 2).tolist()
    
    # Make sure the polygon has at least 3 points
    if len(polygon) < 3:
        return None
        
    return polygon, mask


def overlay_and_save(mask1, mask2, save_path):
    base = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR) if len(mask1.shape) == 2 else mask1
    base[mask2 > 0] = [0, 0, 255]  # Red overlay
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    # Auto-increment filename
    counter = 1
    path = save_path
    
    while os.path.exists(path):
        name, ext = os.path.splitext(save_path)
        path = f"{name}_{counter}{ext}"
        counter += 1
    
    cv2.imwrite(path, base)
    print(f"Saved overlay image to {path}")
    return 


def estimate_pose(image, polygon_coords, mesh_path, cam_K, init_tvec, save_path=None):
    
    image_size = image.shape[:2]
    test_mask = create_polygon_mask(image, polygon_coords)

    mesh = trimesh.load(mesh_path)

    _, rendered_mask = create_polygon_from_mesh(mesh, cam_K, np.eye(3), init_tvec, image.shape)    
    
    overlay_and_save(test_mask, rendered_mask, os.path.join(save_path, "initial_overlay.png")) if save_path else None

    

    ############## Optimization by size ##################

    def cost_function(Rmat):
        _, rendered_mask = create_polygon_from_mesh(mesh, cam_K, Rmat, init_tvec, image_size)  
        
        if rendered_mask is None:
            return float('inf')
        
        mask1 = test_mask.astype(np.uint8) 
        mask2 = rendered_mask.astype(np.uint8)
        
        cost = abs(np.sum(mask1>0) - np.sum(mask2>0))/np.sum(mask1>0)
      
        return cost
    
    init_v = np.zeros(3)           # start at no rotation
    
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    best_cost, best_R = float("inf"), None

    samples = []

    for ax, ay, az in product(angles, angles, angles):
        Rmat = Rotation.from_euler("xyz", [ax, ay, az]).as_matrix()
        c = cost_function(Rmat)
        if c < best_cost:
            best_ax, best_ay, best_az, best_cost = ax, ay, az, c
        samples.append((ax, ay, az, c))

    global_best_R = Rotation.from_euler("xyz", [best_ax, best_ay, best_az]).as_matrix()

    _, global_opt_rendered_mask = create_polygon_from_mesh(mesh, cam_K, global_best_R, init_tvec, image_size)
    overlay_and_save(test_mask, global_opt_rendered_mask, os.path.join(save_path, "global_optimized_overlay.png")) if save_path else None
    
    ############## Optimization by z rotation with random kicks ##################

    def z_rotation_costfunction(x, y , z, angle_rad):
        R_z = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                        [np.sin(angle_rad),  np.cos(angle_rad), 0],
                        [0, 0, 1]])
        new_R =  R_z @ global_best_R

        _, rendered_mask = create_polygon_from_mesh(mesh, cam_K, new_R, np.array([x, y, z]), image_size)  
        
        if rendered_mask is None:
            return float('inf')
        
        mask1 = test_mask.astype(np.uint8) 
        mask2 = rendered_mask.astype(np.uint8)

        difference = cv2.bitwise_xor(mask1, mask2)

        # Auto-increment filename
        counter = 1
        file_path = save_path + "/debug_diff.png" if save_path else None
        path = file_path
        
        if save_path:
            cv2.imwrite(path, difference)
        cost = np.sum(difference > 0)/(np.sum(mask1 > 0) + np.sum(mask2 > 0))
        print(f"z_rotation_costfunction({x:.2f}, {y:.2f}, {z:.2f}, {angle_rad:.2f}) = {cost}")
        return cost
    
    # vector wrapper
    f = lambda v: z_rotation_costfunction(v[0], v[1], v[2], v[3])
    
    # ---- 2) numerical gradient ---------------------------------------
    def grad(v, rel_step=1e-3, abs_step=1e-3):
        g = np.zeros_like(v, dtype=float)
        for i in range(len(v)):
            # relative step scaled to variable magnitude (important for large z)
            h = max(abs_step, rel_step * max(1.0, abs(v[i])))
            e = np.zeros_like(v); e[i] = h
            f_plus = f(v + e)
            f_minus = f(v - e)
            # if cost returns inf/nan, try a bigger step (fallback)
            if not (np.isfinite(f_plus) and np.isfinite(f_minus)):
                h2 = max(1.0, h)
                e[i] = h2
                f_plus = f(v + e)
                f_minus = f(v)
                g[i] = (f_plus - f_minus) / h2
            else:
                g[i] = (f_plus - f_minus) / (2.0 * h)
        return g

    # ---- 3) gradient descent with random kicks when stuck ----------------
    def gradient_descent_with_kicks(x0, bounds, lr=0.1, iters=200, tol=1e-4, 
                                   max_kicks=5, target_cost=0.2, stagnation_threshold=20):
        x = x0.astype(float)
        best_x = x.copy()
        best_cost = f(x)
        
        print(f"Initial cost: {best_cost:.4f}")
        
        kick_count = 0
        
        while kick_count <= max_kicks and best_cost > target_cost:
            print(f"\n--- Attempt {kick_count + 1} (Kick #{kick_count}) ---")
            
            # Track cost history for stagnation detection
            cost_history = []
            stagnation_counter = 0
            
            for it in range(iters):
                current_cost = f(x)
                cost_history.append(current_cost)
                
                # Update best solution if improved
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
                    stagnation_counter = 0
                    print(f"  Iter {it}: New best cost = {best_cost:.4f}")
                else:
                    stagnation_counter += 1
                
                # Check if we've reached target
                if current_cost <= target_cost:
                    print(f"  Target cost {target_cost} reached! Final cost: {current_cost:.4f}")
                    return best_x
                
                # Check for stagnation
                if stagnation_counter >= stagnation_threshold:
                    print(f"  Stagnation detected after {stagnation_threshold} iterations without improvement")
                    break
                
                # Compute gradient and update
                g = grad(x)
                ng = np.linalg.norm(g)
                if ng < tol:
                    print(f"  Gradient norm {ng:.6f} below tolerance {tol}")
                    break
                
                x -= lr * g

                # clip translations, wrap angle
                for i, (lo, hi) in enumerate(bounds):
                    if i < 3:
                        x[i] = np.clip(x[i], lo, hi)
                    else:
                        # wrap angle into [0, 2pi)
                        x[3] = x[3] % (2 * np.pi)
            
            # If we haven't reached target and have kicks left, apply random kick
            if best_cost > target_cost and kick_count < max_kicks:
                kick_count += 1
                print(f"  Applying random kick #{kick_count}")
                
                # Generate random kick within bounds
                kick_strength = 0.1  # Adjust this to control kick magnitude
                for i in range(len(x)):
                    if i < 3:  # translation parameters
                        bound_range = bounds[i][1] - bounds[i][0]
                        kick = np.random.uniform(-kick_strength * bound_range, 
                                               kick_strength * bound_range)
                        x[i] = best_x[i] + kick
                        x[i] = np.clip(x[i], bounds[i][0], bounds[i][1])
                    else:  # angle parameter
                        kick = np.random.uniform(-kick_strength * 2 * np.pi, 
                                               kick_strength * 2 * np.pi)
                        x[i] = (best_x[i] + kick) % (2 * np.pi)
                
                print(f"  Kicked from {best_x} to {x}")
            else:
                break
        
        print(f"\nOptimization finished:")
        print(f"  Final cost: {best_cost:.4f}")
        print(f"  Kicks used: {kick_count}")
        print(f"  Target reached: {'Yes' if best_cost <= target_cost else 'No'}")
        
        return best_x

    x0 = np.array([init_tvec[0], init_tvec[1], init_tvec[2], 0.0])  # initial guess
    bnds = []
    for val in init_tvec:
        low  = min(0.8*val, 1.2*val)
        high = max(0.8*val, 1.2*val)
        bnds.append((low, high))

    # add bounds for 'a'
    bnds.append((0.0, 2*np.pi))
    
    # Use the new optimization function with kicks
    opt = gradient_descent_with_kicks(x0, bnds, lr=0.05, iters=150, 
                                     max_kicks=7, target_cost=0.2, 
                                     stagnation_threshold=20)
    
    print("Final solution:", opt, "cost:", f(opt))

    final_R = Rotation.from_euler("z", opt[3]).as_matrix() @ global_best_R
    final_tvec = opt[:3]

    _, final_rendered_mask = create_polygon_from_mesh(mesh, cam_K, final_R, np.array(final_tvec), image_size)
    overlay_and_save(test_mask, final_rendered_mask, os.path.join(save_path, "final_optimized_overlay.png")) if save_path else None