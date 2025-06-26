import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.patches as patches

class HomographyOptimizer:
    def __init__(self, device='cpu'):
        self.device = device
        
    def create_homography_matrix(self, params):
        """
        Create 3x3 homography matrix from 8 parameters
        Format: [h11, h12, h13, h21, h22, h23, h31, h32]
        h33 is fixed to 1.0
        """
        H = torch.zeros(3, 3, device=self.device, dtype=torch.float32)
        H[0, 0] = params[0]  # h11
        H[0, 1] = params[1]  # h12
        H[0, 2] = params[2]  # h13
        H[1, 0] = params[3]  # h21
        H[1, 1] = params[4]  # h22
        H[1, 2] = params[5]  # h23
        H[2, 0] = params[6]  # h31
        H[2, 1] = params[7]  # h32
        H[2, 2] = 1.0        # h33 (fixed)
        return H
    
    def transform_polygon(self, polygon_points, H):
        """
        Transform polygon points using homography matrix
        polygon_points: Nx2 tensor of (x, y) coordinates
        H: 3x3 homography matrix
        """
        # Convert to homogeneous coordinates
        ones = torch.ones(polygon_points.shape[0], 1, device=self.device)
        homogeneous_points = torch.cat([polygon_points, ones], dim=1)
        
        # Apply transformation
        transformed_homogeneous = torch.matmul(H, homogeneous_points.T).T
        
        # Convert back to cartesian coordinates
        w = transformed_homogeneous[:, 2].unsqueeze(1)
        transformed_points = transformed_homogeneous[:, :2] / (w + 1e-8)  # Add small epsilon to avoid division by zero
        
        return transformed_points
    
    def polygon_area_shoelace(self, points):
        """
        Calculate polygon area using shoelace formula (differentiable)
        """
        x = points[:, 0]
        y = points[:, 1]
        
        # Shoelace formula
        n = len(x)
        area = 0.5 * torch.abs(
            torch.sum(x[:-1] * y[1:]) - torch.sum(x[1:] * y[:-1]) +
            x[-1] * y[0] - x[0] * y[-1]
        )
        return area
    
    def soft_polygon_intersection_area(self, poly1, poly2, grid_resolution=100):
        """
        Approximate polygon intersection using a soft sampling approach
        This creates a differentiable approximation of intersection area
        """
        # Find bounding box of both polygons
        all_points = torch.cat([poly1, poly2], dim=0)
        min_x, max_x = torch.min(all_points[:, 0]), torch.max(all_points[:, 0])
        min_y, max_y = torch.min(all_points[:, 1]), torch.max(all_points[:, 1])
        
        # Add some padding
        padding = 0.1 * max(max_x - min_x, max_y - min_y)
        min_x, max_x = min_x - padding, max_x + padding
        min_y, max_y = min_y - padding, max_y + padding
        
        # Create grid
        x_grid = torch.linspace(min_x, max_x, grid_resolution, device=self.device)
        y_grid = torch.linspace(min_y, max_y, grid_resolution, device=self.device)
        xx, yy = torch.meshgrid(x_grid, y_grid, indexing='ij')
        grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        # Check if points are inside both polygons
        inside_poly1 = self.point_in_polygon_soft(grid_points, poly1)
        inside_poly2 = self.point_in_polygon_soft(grid_points, poly2)
        
        # Intersection is where both are true
        intersection = inside_poly1 * inside_poly2
        
        # Calculate area (grid cell area * number of cells inside intersection)
        cell_area = ((max_x - min_x) / grid_resolution) * ((max_y - min_y) / grid_resolution)
        intersection_area = torch.sum(intersection) * cell_area
        
        return intersection_area
    
    def point_in_polygon_soft(self, points, polygon, temperature=50.0):
        """
        Soft (differentiable) version of point-in-polygon test using ray casting
        """
        n_points = points.shape[0]
        n_vertices = polygon.shape[0]
        
        # Use ray casting approach
        inside_score = torch.zeros(n_points, device=self.device)
        
        for i in range(n_vertices):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n_vertices]
            
            # Soft ray casting
            y_diff = p2[1] - p1[1]
            x_diff = p2[0] - p1[0]
            
            # Check if ray crosses edge
            t = torch.clamp((points[:, 1] - p1[1]) / (y_diff + 1e-8), 0, 1)
            x_intersect = p1[0] + t * x_diff
            
            # Soft counting of crossings
            crosses = torch.sigmoid(temperature * (x_intersect - points[:, 0]))
            valid_crossing = torch.sigmoid(temperature * torch.min(
                torch.stack([points[:, 1] - torch.min(p1[1], p2[1]),
                           torch.max(p1[1], p2[1]) - points[:, 1]]), dim=0
            )[0])
            
            inside_score += crosses * valid_crossing
        
        # Even number of crossings = outside, odd = inside
        return torch.sigmoid(temperature * (torch.sin(np.pi * inside_score)))
    
    def exact_polygon_intersection_area(self, poly1_np, poly2_np):
        """
        Calculate exact intersection area using Shapely (for comparison/ground truth)
        """
        try:
            shapely_poly1 = Polygon(poly1_np)
            shapely_poly2 = Polygon(poly2_np)
            
            if not shapely_poly1.is_valid:
                shapely_poly1 = shapely_poly1.buffer(0)
            if not shapely_poly2.is_valid:
                shapely_poly2 = shapely_poly2.buffer(0)
                
            intersection = shapely_poly1.intersection(shapely_poly2)
            return intersection.area
        except:
            return 0.0
    
    def optimize_homography(self, polygon1, polygon2, max_iterations=1000, lr=0.01, verbose=True):
        """
        Main optimization function to find homography that maximizes polygon overlap
        
        polygon1: Nx2 numpy array - source polygon
        polygon2: Nx2 numpy array - target polygon
        """
        # Convert to tensors
        poly1_tensor = torch.tensor(polygon1, dtype=torch.float32, device=self.device)
        poly2_tensor = torch.tensor(polygon2, dtype=torch.float32, device=self.device)
        
        # Initialize homography parameters with better starting values
        params = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
                            device=self.device, requires_grad=True)
        
        # Create parameter groups with different learning rates
        optimizer = optim.Adam([
            {'params': [params], 'lr': lr}
        ])
        
        # Training loop
        losses = []
        intersection_areas = []
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Create homography matrix
            H = self.create_homography_matrix(params)
            
            # Transform polygon1 to match polygon2's coordinate system
            transformed_poly1 = self.transform_polygon(poly1_tensor, H)
            
            # Calculate intersection area (our objective to maximize)
            intersection_area = self.soft_polygon_intersection_area(
                transformed_poly1, poly2_tensor, grid_resolution=100  # Increased resolution
            )
            
            # We want to maximize intersection, so minimize negative intersection
            loss = -intersection_area
            
            # Progressive regularization (stronger early, weaker later)
            reg_weight = 0.01 * (1 - iteration / max_iterations)
            
            # Different regularization for different parameter types
            # Affine parameters (h11, h12, h21, h22)
            affine_reg = reg_weight * (
                (params[0] - 1.0)**2 + params[1]**2 + 
                params[3]**2 + (params[4] - 1.0)**2
            )
            
            # Translation parameters (h13, h23) - lighter regularization
            translation_reg = reg_weight * 0.1 * (params[2]**2 + params[5]**2)
            
            # Perspective parameters (h31, h32) - stronger regularization
            perspective_reg = reg_weight * 10.0 * (params[6]**2 + params[7]**2)
            
            total_loss = loss + affine_reg + translation_reg + perspective_reg
            
            total_loss.backward()
            
            # Apply different step sizes for different parameter types
            with torch.no_grad():
                if iteration < max_iterations // 3:
                    # Early stage: focus on translation and rotation
                    params.grad[0:2] *= 0.1  # Scale/rotation gradients
                    params.grad[2:6] *= 1.0  # Translation gradients  
                    params.grad[6:8] *= 0.01 # Perspective gradients
                elif iteration < 2 * max_iterations // 3:
                    # Middle stage: balance all parameters
                    params.grad[0:2] *= 0.5
                    params.grad[2:6] *= 1.0
                    params.grad[6:8] *= 0.1
                else:
                    # Late stage: fine-tune all parameters
                    params.grad *= 1.0
                
                # Constrain perspective parameters to reasonable range
                params[6:8].clamp_(-0.01, 0.01)
            
            optimizer.step()
            
            losses.append(total_loss.item())
            intersection_areas.append(intersection_area.item())
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {total_loss.item():.6f}, "
                      f"Intersection Area = {intersection_area.item():.6f}")
        
        # Return final homography matrix and optimization history
        final_H = self.create_homography_matrix(params)
        
        return {
            'homography_matrix': final_H.detach().cpu().numpy(),
            'homography_params': params.detach().cpu().numpy(),
            'losses': losses,
            'intersection_areas': intersection_areas,
            'transformed_polygon': self.transform_polygon(poly1_tensor, final_H).detach().cpu().numpy()
        }