"""
Unified Pose Refiner Class - Clean silhouette-based 6-DOF pose refinement.

Single class interface with all functionality:
- Efficient mesh rendering
- Multi-strategy optimization  
- Visualization and metrics
- Easy-to-use API

Usage:
    refiner = PoseRefiner(mesh_path, camera_matrix, image_size)
    result = refiner.optimize_pose(target_mask, init_pose)
"""

import numpy as np
import cv2
import trimesh
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Optional, Dict, Any, Union, List
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoseRefiner:
    """
    Unified class for 6-DOF pose refinement using silhouette matching.
    
    Combines efficient rendering, robust optimization, and clean API.
    """
    
    def __init__(self, mesh_path: str, camera_matrix: np.ndarray, image_size: Tuple[int, int]):
        """
        Initialize pose refiner.
        
        Args:
            mesh_path: Path to 3D mesh file (PLY, OBJ, etc.)
            camera_matrix: 3x3 camera intrinsic matrix
            image_size: (width, height) of target images
        """
        self.camera_matrix = np.array(camera_matrix, dtype=np.float64)
        self.image_size = image_size
        self.width, self.height = image_size
        
        # Load and validate mesh
        self._load_mesh(mesh_path)
        
        logger.info(f"PoseRefiner initialized: {len(self.vertices)} vertices, {len(self.faces)} faces")
    
    def _load_mesh(self, mesh_path: str) -> None:
        """Load and validate mesh."""
        if not Path(mesh_path).exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        
        mesh = trimesh.load(mesh_path, force='mesh')
        if mesh.is_empty:
            raise ValueError(f"Empty mesh loaded from: {mesh_path}")
        
        self.vertices = np.array(mesh.vertices, dtype=np.float32)
        self.faces = np.array(mesh.faces, dtype=np.int32)
        self.mesh_path = mesh_path
    
    def render_silhouette(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """
        Render binary silhouette mask for given pose.
        
        Args:
            rvec: 3D rotation vector (Rodrigues format)
            tvec: 3D translation vector
            
        Returns:
            Binary mask (H, W) with 255 for object pixels
        """
        # Project vertices to image plane
        projected_points, _ = cv2.projectPoints(
            self.vertices, rvec.astype(np.float64), tvec.astype(np.float64), 
            self.camera_matrix, None
        )
        projected_points = projected_points.reshape(-1, 2)
        
        # Create mask
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Get camera position for backface culling
        R, _ = cv2.Rodrigues(rvec.astype(np.float64))
        camera_pos = -R.T @ tvec.astype(np.float64)
        
        # Render each triangle with backface culling
        for face in self.faces:
            if self._is_backface(face, camera_pos):
                continue
                
            triangle = projected_points[face].astype(np.int32)
            if self._triangle_in_bounds(triangle):
                cv2.fillPoly(mask, [triangle], 255)
        
        return mask
    
    def _is_backface(self, face: np.ndarray, camera_pos: np.ndarray) -> bool:
        """Check if triangle face is pointing away from camera."""
        v0, v1, v2 = self.vertices[face]
        normal = np.cross(v1 - v0, v2 - v0)
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        
        center = (v0 + v1 + v2) / 3
        to_camera = camera_pos - center
        to_camera = to_camera / (np.linalg.norm(to_camera) + 1e-8)
        
        return np.dot(normal, to_camera) < 0
    
    def _triangle_in_bounds(self, triangle: np.ndarray) -> bool:
        """Check if triangle intersects with image bounds."""
        min_x, min_y = triangle.min(axis=0)
        max_x, max_y = triangle.max(axis=0)
        return not (max_x < 0 or min_x >= self.width or max_y < 0 or min_y >= self.height)
    
    def create_mask_from_polygon(self, polygon: List[List[int]]) -> np.ndarray:
        """
        Create binary mask from polygon points.
        
        Args:
            polygon: List of [x, y] coordinates
            
        Returns:
            Binary mask (H, W) with 255 for polygon interior
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        poly_array = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [poly_array], 255)
        return mask
    
    def compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Intersection over Union between two binary masks."""
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        return intersection / (union + 1e-8)
    
    def compute_loss(self, rvec: np.ndarray, tvec: np.ndarray, target_mask: np.ndarray, 
                    metric: str = 'iou') -> float:
        """
        Compute loss between rendered silhouette and target mask.
        
        Args:
            rvec: Rotation vector
            tvec: Translation vector  
            target_mask: Target binary mask
            metric: Loss metric ('iou', 'l2', 'dice')
            
        Returns:
            Loss value (lower is better)
        """
        try:
            rendered_mask = self.render_silhouette(rvec, tvec)
            
            if metric == 'iou':
                return 1.0 - self.compute_iou(rendered_mask, target_mask)
            elif metric == 'l2':
                diff = (rendered_mask.astype(np.float32) - target_mask.astype(np.float32)) / 255.0
                return np.mean(diff**2)
            elif metric == 'dice':
                intersection = np.logical_and(rendered_mask > 0, target_mask > 0).sum()
                total = (rendered_mask > 0).sum() + (target_mask > 0).sum()
                dice = 2.0 * intersection / (total + 1e-8)
                return 1.0 - dice
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
        except Exception as e:
            logger.warning(f"Rendering failed during loss computation: {e}")
            return 1e6
    
    def _objective_function(self, params: np.ndarray, target_mask: np.ndarray, metric: str) -> float:
        """Objective function for scipy optimizers."""
        rvec, tvec = params[:3], params[3:6]
        return self.compute_loss(rvec, tvec, target_mask, metric)
    
    def optimize_pose(self, target_mask: np.ndarray, init_rvec: np.ndarray, init_tvec: np.ndarray,
                     method: str = 'multi_stage', metric: str = 'iou', 
                     max_iterations: int = 500, bounds_scale: float = 500.0) -> Dict[str, Any]:
        """
        Optimize pose to match target silhouette.
        
        Args:
            target_mask: Binary target silhouette mask
            init_rvec: Initial rotation vector (3,)
            init_tvec: Initial translation vector (3,)
            method: Optimization strategy:
                   - 'nelder-mead': Local simplex optimization
                   - 'powell': Local Powell optimization  
                   - 'differential_evolution': Global optimization
                   - 'multi_stage': Global then local (recommended)
                   - 'ensemble': Try multiple methods, return best
            metric: Loss metric ('iou', 'l2', 'dice')
            max_iterations: Maximum optimization iterations
            bounds_scale: Translation bounds scale around initial guess
            
        Returns:
            Dictionary with optimization results:
            {
                'success': bool,
                'rvec': np.ndarray,
                'tvec': np.ndarray, 
                'loss': float,
                'init_loss': float,
                'iterations': int,
                'method': str,
                'message': str
            }
        """
        logger.info(f"Starting pose optimization: {method} method, {metric} metric")
        
        init_params = np.concatenate([init_rvec, init_tvec])
        init_loss = self._objective_function(init_params, target_mask, metric)
        
        logger.info(f"Initial {metric} loss: {init_loss:.4f}")
        
        if method == 'multi_stage':
            return self._multi_stage_optimization(target_mask, init_rvec, init_tvec, 
                                                metric, max_iterations, bounds_scale)
        elif method == 'ensemble':
            return self._ensemble_optimization(target_mask, init_rvec, init_tvec,
                                             metric, max_iterations, bounds_scale)
        else:
            return self._single_optimization(target_mask, init_rvec, init_tvec, 
                                           method, metric, max_iterations, bounds_scale)
    
    def _single_optimization(self, target_mask: np.ndarray, init_rvec: np.ndarray, 
                           init_tvec: np.ndarray, method: str, metric: str,
                           max_iterations: int, bounds_scale: float) -> Dict[str, Any]:
        """Single optimization run."""
        init_params = np.concatenate([init_rvec, init_tvec])
        init_loss = self._objective_function(init_params, target_mask, metric)
        
        if method == 'differential_evolution':
            bounds = [
                (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi),  # rotation bounds
                (init_tvec[0] - bounds_scale, init_tvec[0] + bounds_scale),  # translation bounds
                (init_tvec[1] - bounds_scale, init_tvec[1] + bounds_scale),
                (init_tvec[2] - bounds_scale, init_tvec[2] + bounds_scale)
            ]
            
            result = differential_evolution(
                self._objective_function, bounds,
                args=(target_mask, metric),
                maxiter=max_iterations // 10,
                seed=42
            )
        else:
            result = minimize(
                self._objective_function, init_params,
                args=(target_mask, metric),
                method=method,
                options={'maxiter': max_iterations}
            )
        
        final_rvec, final_tvec = result.x[:3], result.x[3:6]
        
        return {
            'success': result.success,
            'rvec': final_rvec,
            'tvec': final_tvec,
            'loss': result.fun,
            'init_loss': init_loss,
            'iterations': getattr(result, 'nit', None),
            'method': method,
            'message': getattr(result, 'message', str(result))
        }
    
    def _multi_stage_optimization(self, target_mask: np.ndarray, init_rvec: np.ndarray,
                                init_tvec: np.ndarray, metric: str, max_iterations: int,
                                bounds_scale: float) -> Dict[str, Any]:
        """Multi-stage optimization: global search followed by local refinement."""
        # Stage 1: Global search
        logger.info("Stage 1: Global optimization with differential evolution")
        result1 = self._single_optimization(
            target_mask, init_rvec, init_tvec, 'differential_evolution',
            metric, max_iterations, bounds_scale
        )
        
        # Stage 2: Local refinement
        logger.info("Stage 2: Local refinement with Nelder-Mead")
        result2 = self._single_optimization(
            target_mask, result1['rvec'], result1['tvec'], 'nelder-mead',
            metric, max_iterations, bounds_scale
        )
        
        result2['method'] = 'multi_stage'
        logger.info(f"Multi-stage optimization completed. Final loss: {result2['loss']:.4f}")
        
        return result2
    
    def _ensemble_optimization(self, target_mask: np.ndarray, init_rvec: np.ndarray,
                             init_tvec: np.ndarray, metric: str, max_iterations: int,
                             bounds_scale: float) -> Dict[str, Any]:
        """Ensemble optimization: try multiple methods and return best result."""
        methods = ['nelder-mead', 'powell', 'differential_evolution']
        results = []
        
        for method in methods:
            logger.info(f"Ensemble: Running {method}")
            result = self._single_optimization(
                target_mask, init_rvec, init_tvec, method,
                metric, max_iterations, bounds_scale
            )
            results.append(result)
        
        # Return best result
        best_result = min(results, key=lambda x: x['loss'])
        best_method = methods[results.index(best_result)]
        
        best_result['method'] = f'ensemble_{best_method}'
        logger.info(f"Ensemble optimization completed. Best method: {best_method}")
        
        return best_result
    
    def evaluate_pose(self, rvec: np.ndarray, tvec: np.ndarray, target_mask: np.ndarray) -> Dict[str, float]:
        """
        Evaluate pose quality with multiple metrics.
        
        Returns:
            Dictionary with evaluation metrics
        """
        rendered_mask = self.render_silhouette(rvec, tvec)
        
        iou = self.compute_iou(rendered_mask, target_mask)
        
        intersection = np.logical_and(rendered_mask > 0, target_mask > 0).sum()
        total = (rendered_mask > 0).sum() + (target_mask > 0).sum()
        dice = 2.0 * intersection / (total + 1e-8)
        
        diff = (rendered_mask.astype(np.float32) - target_mask.astype(np.float32)) / 255.0
        l2_loss = np.mean(diff**2)
        
        precision = intersection / ((rendered_mask > 0).sum() + 1e-8)
        recall = intersection / ((target_mask > 0).sum() + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'iou': iou,
            'dice': dice,
            'l2_loss': l2_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'target_area': (target_mask > 0).sum(),
            'rendered_area': (rendered_mask > 0).sum()
        }
    
    def visualize_result(self, image_path: str, target_mask: np.ndarray, 
                        rvec: np.ndarray, tvec: np.ndarray, 
                        output_dir: str = '.', save_individual: bool = True) -> Dict[str, Any]:
        """
        Generate visualization of optimization results.
        
        Args:
            image_path: Path to original image
            target_mask: Target binary mask
            rvec: Optimized rotation vector
            tvec: Optimized translation vector
            output_dir: Directory to save outputs
            save_individual: Whether to save individual mask images
            
        Returns:
            Dictionary with evaluation metrics and file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load original image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Render optimized silhouette
        rendered_mask = self.render_silhouette(rvec, tvec)
        
        # Create overlay visualization
        overlay = image.copy()
        overlay[target_mask > 0] = [0, 255, 0]  # Green for target
        overlay[rendered_mask > 0] = [0, 0, 255]  # Red for rendered
        
        # Intersection in yellow
        intersection = np.logical_and(target_mask > 0, rendered_mask > 0)
        overlay[intersection] = [0, 255, 255]
        
        # Blend with original image
        result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        
        # Save main visualization
        overlay_path = output_path / 'pose_optimization_result.png'
        cv2.imwrite(str(overlay_path), result)
        
        file_paths = {'overlay': str(overlay_path)}
        
        # Save individual masks if requested
        if save_individual:
            target_path = output_path / 'target_mask.png'
            rendered_path = output_path / 'rendered_mask.png'
            
            cv2.imwrite(str(target_path), target_mask)
            cv2.imwrite(str(rendered_path), rendered_mask)
            
            file_paths.update({
                'target_mask': str(target_path),
                'rendered_mask': str(rendered_path)
            })
        
        # Compute evaluation metrics
        metrics = self.evaluate_pose(rvec, tvec, target_mask)
        
        logger.info(f"Pose evaluation - IoU: {metrics['iou']:.4f}, "
                   f"Dice: {metrics['dice']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return {**metrics, 'file_paths': file_paths}


def demo():
    """Demonstration of PoseRefiner usage."""
    
    # Configuration
    config = {
        'image_path': "/home/ram/ipd_data/val/000000/rgb_cam1/000000.png",
        'mesh_path': "/home/ram/ipd_data/models/obj_000018.ply",
        'camera_matrix': np.array([[3981.9859, 0, 1954.1872],
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
    refiner = PoseRefiner(
        mesh_path=config['mesh_path'],
        camera_matrix=config['camera_matrix'],
        image_size=image_size
    )
    
    # Create target mask from polygon
    target_mask = refiner.create_mask_from_polygon(config['polygon'])
    
    # Optimize pose
    result = refiner.optimize_pose(
        target_mask=target_mask,
        init_rvec=config['init_rvec'],
        init_tvec=config['init_tvec'],
        method='multi_stage',  # or 'ensemble', 'nelder-mead', etc.
        metric='iou'
    )
    
    # Display results
    if result['success']:
        print("✓ Pose optimization successful!")
        print(f"Method: {result['method']}")
        print(f"Final rotation: {result['rvec']}")
        print(f"Final translation: {result['tvec']}")
        print(f"Loss improvement: {((result['init_loss'] - result['loss']) / result['init_loss'] * 100):.1f}%")
        
        # Generate comprehensive visualization and metrics
        eval_result = refiner.visualize_result(
            image_path=config['image_path'],
            target_mask=target_mask,
            rvec=result['rvec'],
            tvec=result['tvec'],
            output_dir='./pose_refinement_results'
        )
        
        print(f"\nEvaluation Metrics:")
        print(f"  IoU: {eval_result['iou']:.4f}")
        print(f"  Dice: {eval_result['dice']:.4f}")
        print(f"  F1-Score: {eval_result['f1_score']:.4f}")
        print(f"  Precision: {eval_result['precision']:.4f}")
        print(f"  Recall: {eval_result['recall']:.4f}")
        
        print(f"\nOutput files:")
        for name, path in eval_result['file_paths'].items():
            print(f"  {name}: {path}")
        
    else:
        print("✗ Pose optimization failed")
        print(f"Message: {result['message']}")


if __name__ == '__main__':
    demo()