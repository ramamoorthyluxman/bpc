"""
Clean, elegant 6-DOF pose refinement using silhouette matching.

Key improvements:
- Efficient vectorized rasterization using OpenCV
- IoU-based optimization with gradient-free methods
- Clean class-based architecture
- Robust error handling and validation
- Multiple optimization strategies
"""

import numpy as np
import cv2
import trimesh
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Optional, Dict, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SilhouetteRenderer:
    """Efficient silhouette renderer using OpenCV triangle rasterization."""
    
    def __init__(self, mesh_path: str):
        self.mesh = self._load_mesh(mesh_path)
        self.vertices = np.array(self.mesh.vertices, dtype=np.float32)
        self.faces = np.array(self.mesh.faces, dtype=np.int32)
        
    def _load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        """Load and validate mesh."""
        if not Path(mesh_path).exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        
        mesh = trimesh.load(mesh_path, force='mesh')
        if mesh.is_empty:
            raise ValueError(f"Empty mesh loaded from: {mesh_path}")
        
        logger.info(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh
    
    def render(self, K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, 
               image_size: Tuple[int, int]) -> np.ndarray:
        """Render binary silhouette mask."""
        width, height = image_size
        
        # Project vertices to image plane
        projected_points, _ = cv2.projectPoints(
            self.vertices, rvec, tvec, K, None
        )
        projected_points = projected_points.reshape(-1, 2)
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get camera position for backface culling
        R, _ = cv2.Rodrigues(rvec)
        camera_pos = -R.T @ tvec
        
        # Render each triangle
        for face in self.faces:
            # Backface culling
            if self._is_backface(face, camera_pos):
                continue
                
            # Get triangle vertices in image coordinates
            triangle = projected_points[face].astype(np.int32)
            
            # Skip if triangle is completely outside image
            if not self._triangle_in_bounds(triangle, width, height):
                continue
                
            # Fill triangle
            cv2.fillPoly(mask, [triangle], 255)
        
        return mask
    
    def _is_backface(self, face: np.ndarray, camera_pos: np.ndarray) -> bool:
        """Check if triangle face is pointing away from camera."""
        v0, v1, v2 = self.vertices[face]
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        
        # Vector from triangle center to camera
        center = (v0 + v1 + v2) / 3
        to_camera = camera_pos - center
        to_camera = to_camera / (np.linalg.norm(to_camera) + 1e-8)
        
        return np.dot(normal, to_camera) < 0
    
    def _triangle_in_bounds(self, triangle: np.ndarray, width: int, height: int) -> bool:
        """Check if triangle intersects with image bounds."""
        min_x, min_y = triangle.min(axis=0)
        max_x, max_y = triangle.max(axis=0)
        return not (max_x < 0 or min_x >= width or max_y < 0 or min_y >= height)


class PoseOptimizer:
    """Pose optimization using multiple strategies."""
    
    def __init__(self, renderer: SilhouetteRenderer, K: np.ndarray, image_size: Tuple[int, int]):
        self.renderer = renderer
        self.K = K
        self.image_size = image_size
        
    def _pose_to_params(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Convert pose to optimization parameters."""
        return np.concatenate([rvec, tvec])
    
    def _params_to_pose(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert optimization parameters to pose."""
        return params[:3], params[3:6]
    
    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Intersection over Union."""
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        return intersection / (union + 1e-8)
    
    def _objective_function(self, params: np.ndarray, target_mask: np.ndarray, 
                           metric: str = 'iou') -> float:
        """Objective function for optimization."""
        try:
            rvec, tvec = self._params_to_pose(params)
            rendered_mask = self.renderer.render(self.K, rvec, tvec, self.image_size)
            
            if metric == 'iou':
                return 1.0 - self._compute_iou(rendered_mask, target_mask)
            elif metric == 'l2':
                diff = (rendered_mask.astype(np.float32) - target_mask.astype(np.float32)) / 255.0
                return np.mean(diff**2)
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
        except Exception as e:
            logger.warning(f"Rendering failed during optimization: {e}")
            return 1e6
    
    def optimize(self, target_mask: np.ndarray, init_rvec: np.ndarray, init_tvec: np.ndarray,
                method: str = 'nelder-mead', metric: str = 'iou', 
                max_iterations: int = 500) -> Dict[str, Any]:
        """
        Optimize pose to match target silhouette.
        
        Args:
            target_mask: Binary target silhouette
            init_rvec: Initial rotation vector
            init_tvec: Initial translation vector
            method: Optimization method ('nelder-mead', 'powell', 'differential_evolution')
            metric: Loss metric ('iou', 'l2')
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting pose optimization with {method} method, {metric} metric")
        
        init_params = self._pose_to_params(init_rvec, init_tvec)
        init_loss = self._objective_function(init_params, target_mask, metric)
        
        logger.info(f"Initial loss: {init_loss:.4f}")
        
        if method == 'differential_evolution':
            # Global optimization with bounds
            bounds = [
                (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),  # rotation bounds
                (init_tvec[0] - 500, init_tvec[0] + 500),  # translation bounds
                (init_tvec[1] - 500, init_tvec[1] + 500),
                (init_tvec[2] - 500, init_tvec[2] + 500)
            ]
            
            result = differential_evolution(
                self._objective_function, bounds,
                args=(target_mask, metric),
                maxiter=max_iterations // 10,
                seed=42
            )
        else:
            # Local optimization
            result = minimize(
                self._objective_function, init_params,
                args=(target_mask, metric),
                method=method,
                options={'maxiter': max_iterations}
            )
        
        final_rvec, final_tvec = self._params_to_pose(result.x)
        final_loss = result.fun
        
        logger.info(f"Optimization completed. Final loss: {final_loss:.4f}")
        logger.info(f"Improvement: {((init_loss - final_loss) / init_loss * 100):.1f}%")
        
        return {
            'success': result.success,
            'rvec': final_rvec,
            'tvec': final_tvec,
            'loss': final_loss,
            'init_loss': init_loss,
            'iterations': result.nit if hasattr(result, 'nit') else None,
            'message': result.message if hasattr(result, 'message') else str(result)
        }


class PoseRefiner:
    """Main pose refinement interface."""
    
    def __init__(self, mesh_path: str, K: np.ndarray, image_size: Tuple[int, int]):
        self.renderer = SilhouetteRenderer(mesh_path)
        self.optimizer = PoseOptimizer(self.renderer, K, image_size)
        self.K = K
        self.image_size = image_size
        
    def create_polygon_mask(self, polygon: list, image_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Create binary mask from polygon points."""
        if image_size is None:
            image_size = self.image_size
        
        width, height = image_size
        mask = np.zeros((height, width), dtype=np.uint8)
        
        poly_array = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [poly_array], 255)
        
        return mask
    
    def refine_pose(self, target_mask: np.ndarray, init_rvec: np.ndarray, init_tvec: np.ndarray,
                   strategy: str = 'multi_stage') -> Dict[str, Any]:
        """
        Refine pose using the specified strategy.
        
        Strategies:
        - 'single': Single optimization run
        - 'multi_stage': Coarse-to-fine optimization
        - 'ensemble': Multiple methods with best result selection
        """
        
        if strategy == 'single':
            return self.optimizer.optimize(target_mask, init_rvec, init_tvec)
        
        elif strategy == 'multi_stage':
            # Stage 1: Global search with differential evolution
            logger.info("Stage 1: Global optimization")
            result1 = self.optimizer.optimize(
                target_mask, init_rvec, init_tvec,
                method='differential_evolution', metric='iou', max_iterations=200
            )
            
            # Stage 2: Local refinement
            logger.info("Stage 2: Local refinement")
            result2 = self.optimizer.optimize(
                target_mask, result1['rvec'], result1['tvec'],
                method='nelder-mead', metric='iou', max_iterations=300
            )
            
            return result2
        
        elif strategy == 'ensemble':
            methods = ['nelder-mead', 'powell', 'differential_evolution']
            results = []
            
            for method in methods:
                logger.info(f"Running optimization with {method}")
                result = self.optimizer.optimize(
                    target_mask, init_rvec, init_tvec,
                    method=method, metric='iou'
                )
                results.append(result)
            
            # Return best result
            best_result = min(results, key=lambda x: x['loss'])
            logger.info(f"Best method: {methods[results.index(best_result)]}")
            
            return best_result
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def visualize_results(self, image_path: str, target_mask: np.ndarray, 
                         rvec: np.ndarray, tvec: np.ndarray, output_dir: str = '.'):
        """Generate visualization of optimization results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load original image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Render optimized silhouette
        rendered_mask = self.renderer.render(self.K, rvec, tvec, self.image_size)
        
        # Create overlay
        overlay = image.copy()
        overlay[target_mask > 0] = [0, 255, 0]  # Green for target
        overlay[rendered_mask > 0] = [0, 0, 255]  # Red for rendered
        
        # Blend with original
        result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        
        # Save outputs
        cv2.imwrite(str(output_path / 'target_mask.png'), target_mask)
        cv2.imwrite(str(output_path / 'rendered_mask.png'), rendered_mask)
        cv2.imwrite(str(output_path / 'overlay.png'), result)
        
        # Compute metrics
        iou = self.optimizer._compute_iou(rendered_mask, target_mask)
        logger.info(f"Final IoU: {iou:.4f}")
        
        return {
            'iou': iou,
            'target_area': (target_mask > 0).sum(),
            'rendered_area': (rendered_mask > 0).sum()
        }


def main():
    """Example usage."""
    
    # Configuration
    config = {
        'image_path': "/home/ram/ipd_data/val/000000/rgb_cam1/000000.png",
        'mesh_path': "/home/ram/ipd_data/models/obj_000018.ply",
        'K': np.array([[3981.9859, 0, 1954.1872],
                      [0, 3981.9859, 1103.6978],
                      [0, 0, 1]], dtype=np.float64),
        'init_rvec': np.array([0.0, 0.0, 0.0], dtype=np.float64),
        'init_tvec': np.array([2.452, 256.10981, 1758.9929], dtype=np.float64),
        'polygon': [[1784, 1774], [1778, 2015], [1829, 2018], [1829, 2003], 
                   [1830, 2002], [1830, 1971], [1831, 1970], [1831, 1939], 
                   [1832, 1938], [1832, 1924], [1833, 1923], [1842, 1923], 
                   [1843, 1924], [1862, 1924], [1863, 1925], [1883, 1925], 
                   [1884, 1926], [1903, 1926], [1904, 1927], [1924, 1927], 
                   [1925, 1928], [1944, 1928], [1945, 1929], [1965, 1929], 
                   [1966, 1930], [1985, 1930], [1986, 1931], [2006, 1931], 
                   [2007, 1932], [2026, 1932], [2027, 1933], [2037, 1933], 
                   [2039, 1884], [2030, 1884], [2029, 1883], [2011, 1883], 
                   [2010, 1882], [1993, 1882], [1992, 1881], [1974, 1881], 
                   [1973, 1880], [1955, 1880], [1954, 1879], [1937, 1879], 
                   [1936, 1878], [1918, 1878], [1917, 1877], [1899, 1877], 
                   [1898, 1876], [1880, 1876], [1879, 1875], [1862, 1875], 
                   [1861, 1874], [1843, 1874], [1842, 1873], [1834, 1873], 
                   [1833, 1872], [1833, 1857], [1834, 1856], [1834, 1825], 
                   [1835, 1824], [1835, 1793], [1836, 1792], [1836, 1777]]
    }
    
    # Get image dimensions
    image = cv2.imread(config['image_path'])
    height, width = image.shape[:2]
    image_size = (width, height)
    
    # Initialize pose refiner
    refiner = PoseRefiner(config['mesh_path'], config['K'], image_size)
    
    # Create target mask from polygon
    target_mask = refiner.create_polygon_mask(config['polygon'])
    
    # Refine pose
    result = refiner.refine_pose(
        target_mask, 
        config['init_rvec'], 
        config['init_tvec'],
        strategy='multi_stage'
    )
    
    if result['success']:
        print("✓ Pose optimization successful!")
        print(f"Final rotation: {result['rvec']}")
        print(f"Final translation: {result['tvec']}")
        print(f"Loss improvement: {((result['init_loss'] - result['loss']) / result['init_loss'] * 100):.1f}%")
        
        # Generate visualizations
        metrics = refiner.visualize_results(
            config['image_path'], target_mask, 
            result['rvec'], result['tvec']
        )
        print(f"Final IoU: {metrics['iou']:.4f}")
        
    else:
        print("✗ Pose optimization failed")
        print(f"Message: {result['message']}")


if __name__ == '__main__':
    main()