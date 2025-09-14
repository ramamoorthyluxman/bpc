import cv2
import numpy as np
import os
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
        
        cost1 = abs(np.sum(mask1>0) - np.sum(mask2>0))/np.sum(mask1>0)
      
        # # Compute the cost as the number of differing pixels
        # diff = cv2.bitwise_xor(mask1, mask2)
        # cost2= np.sum(diff > 0)/(image_size[0] * image_size[1])  # Normalize by image size

        # cost = cost1 * cost2
        # print(f"Current cost: {cost} for rotation vector {Rmat}")
        return cost1
    
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
    samples = np.array(samples)
    X = samples[:, :3]
    y = samples[:, 3]

    poly = PolynomialFeatures(degree=3, include_bias=True)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    def poly_cost(v):
        # v is [ax, ay, az]
        # return model.predict(poly.transform([v]))[0]
        ax, ay, az = v[:3]
        Rmat = Rotation.from_euler("xyz", [ax, ay, az]).as_matrix()
        
        _, rendered_mask = create_polygon_from_mesh(mesh, cam_K, Rmat, init_tvec, image_size)  
        
        if rendered_mask is None:
            return float('inf')
        
        mask1 = test_mask.astype(np.uint8) 
        mask2 = rendered_mask.astype(np.uint8)
        
        cost = abs(np.sum(mask1>0) - np.sum(mask2>0))/np.sum(mask1>0)
        
        return cost

    res = minimize(poly_cost, [best_ax, best_ay, best_az], method="Powell", options={"maxiter": 1000, "disp": True})
    best_R = Rotation.from_euler("xyz", [res.x[0], res.x[1], res.x[2]]).as_matrix()

    _, size_opt_rendered_mask = create_polygon_from_mesh(mesh, cam_K, best_R, init_tvec, image_size)
    overlay_and_save(test_mask, size_opt_rendered_mask, os.path.join(save_path, "size_optimized_overlay.png")) if save_path else None

    
    # run a local optimizer on the polynomial
    # # res = minimize(poly_cost, [best_ax, best_ay, best_az], method="Powell", options={"maxiter": 1000, "disp": True})
    # bounds = [
    #     tuple(sorted((0.7*init_tvec[0], 1.3*init_tvec[0]))),
    #     tuple(sorted((0.7*init_tvec[1], 1.3*init_tvec[1]))),
    #     tuple(sorted((0.7*init_tvec[2], 1.3*init_tvec[2]))),
    #     (0, 2*np.pi),
    #     (0, 2*np.pi),
    #     (0, 2*np.pi)
    # ]
    # res = minimize(poly_cost, [init_tvec[0], init_tvec[1], init_tvec[2], best_ax, best_ay, best_az], method="L-BFGS-B", bounds=bounds)

    # best_x, best_y, best_z, best_ax, best_ay, best_az = res.x
    # best_R = Rotation.from_euler("xyz", [best_ax, best_ay, best_az]).as_matrix()

    # _, rot_opt_rendered_mask = create_polygon_from_mesh(mesh, cam_K, best_R, np.array([best_x, best_y, best_z]), image_size)
    # overlay_and_save(test_mask, rot_opt_rendered_mask, os.path.join(save_path, "rot_optimized_overlay.png")) if save_path else None
    


    
