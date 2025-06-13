import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from sklearn.decomposition import PCA
import os

def point_in_polygon(point, polygon_points):
    """
    Check if a point is inside a polygon using OpenCV
    
    Args:
        point: (x, y) coordinates
        polygon_points: List of [x, y] coordinates forming the polygon
    
    Returns:
        bool: True if point is inside polygon, False otherwise
    """
    polygon_array = np.array(polygon_points, dtype=np.float32)
    result = cv2.pointPolygonTest(polygon_array, point, False)
    return result >= 0  # >= 0 means inside or on boundary

def compute_polygon_centroid(points):
    """
    Compute the geometric centroid of a polygon
    
    Args:
        points: List of [x, y] coordinates
    
    Returns:
        tuple: (cx, cy) centroid coordinates
    """
    points_array = np.array(points)
    centroid = np.mean(points_array, axis=0)
    return tuple(centroid)

def compute_pca_orientation(points):
    """
    Compute the principal orientation of the polygon using PCA
    
    Args:
        points: List of [x, y] coordinates
    
    Returns:
        float: Angle in degrees of the principal axis
    """
    points_array = np.array(points)
    
    # Center the points
    centered_points = points_array - np.mean(points_array, axis=0)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(centered_points)
    
    # Get the principal component (first eigenvector)
    principal_axis = pca.components_[0]
    
    # Calculate angle in degrees
    angle = np.degrees(np.arctan2(principal_axis[1], principal_axis[0]))
    
    return angle

def compute_oriented_bbox_size(points, center, angle_degrees):
    """
    Compute the size of a rotated bounding box given center and orientation
    
    Args:
        points: List of [x, y] coordinates
        center: (cx, cy) center of the box
        angle_degrees: Rotation angle in degrees
    
    Returns:
        tuple: (width, height) of the bounding box
    """
    points_array = np.array(points)
    cx, cy = center
    
    # Convert angle to radians
    angle_rad = np.radians(angle_degrees)
    
    # Rotation matrix to align with axes
    cos_a = np.cos(-angle_rad)  # Negative to rotate points instead of axes
    sin_a = np.sin(-angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # Translate points to center and rotate
    centered_points = points_array - np.array([cx, cy])
    rotated_points = centered_points @ rotation_matrix.T
    
    # Find bounding box in rotated coordinate system
    min_x, max_x = np.min(rotated_points[:, 0]), np.max(rotated_points[:, 0])
    min_y, max_y = np.min(rotated_points[:, 1]), np.max(rotated_points[:, 1])
    
    width = max_x - min_x
    height = max_y - min_y
    
    return width, height

def find_interior_center(points):
    """
    Find a point that is well inside the polygon
    Uses multiple strategies to ensure the center is interior
    
    Args:
        points: List of [x, y] coordinates
    
    Returns:
        tuple: (cx, cy) interior center coordinates
    """
    # Strategy 1: Try geometric centroid
    centroid = compute_polygon_centroid(points)
    if point_in_polygon(centroid, points):
        return centroid
    
    # Strategy 2: Try center of bounding box
    points_array = np.array(points)
    min_x, max_x = np.min(points_array[:, 0]), np.max(points_array[:, 0])
    min_y, max_y = np.min(points_array[:, 1]), np.max(points_array[:, 1])
    bbox_center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
    
    if point_in_polygon(bbox_center, points):
        return bbox_center
    
    # Strategy 3: Find interior point using erosion approach
    # Sample multiple points inside the polygon and choose the one most central
    polygon_array = np.array(points, dtype=np.int32)
    
    # Create a mask from the polygon
    min_x, min_y = np.min(polygon_array, axis=0)
    max_x, max_y = np.max(polygon_array, axis=0)
    
    mask_width = max_x - min_x + 1
    mask_height = max_y - min_y + 1
    
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    
    # Adjust polygon coordinates to mask space
    adjusted_polygon = polygon_array - np.array([min_x, min_y])
    
    cv2.fillPoly(mask, [adjusted_polygon], 255)
    
    # Find the point with maximum distance to boundary (most interior)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
    
    # Convert back to original coordinate space
    interior_center = (max_loc[0] + min_x, max_loc[1] + min_y)
    
    return interior_center

def compute_rotated_bbox_constrained(points):
    """
    Compute rotated bounding box with center constrained to be inside polygon
    
    Args:
        points: List of [x, y] coordinates
    
    Returns:
        tuple: (center, (width, height), angle_degrees)
    """
    # First try standard OpenCV approach
    points_array = np.array(points, dtype=np.float32)
    rect = cv2.minAreaRect(points_array)
    center, size, angle = rect
    
    # Check if center is inside polygon
    if point_in_polygon(center, points):
        print(f"  Standard minAreaRect center is inside polygon")
        return center, size, angle
    
    print(f"  Standard minAreaRect center is outside polygon, using constrained approach")
    
    # Find a center that's guaranteed to be inside
    interior_center = find_interior_center(points)
    
    # Method 1: Use PCA to find orientation
    pca_angle = compute_pca_orientation(points)
    pca_size = compute_oriented_bbox_size(points, interior_center, pca_angle)
    
    # Method 2: Try the original minAreaRect angle with new center
    original_size = compute_oriented_bbox_size(points, interior_center, angle)
    
    # Method 3: Try angle + 90 degrees
    perpendicular_angle = angle + 90
    perpendicular_size = compute_oriented_bbox_size(points, interior_center, perpendicular_angle)
    
    # Choose the orientation that gives the smallest area
    pca_area = pca_size[0] * pca_size[1]
    original_area = original_size[0] * original_size[1]
    perpendicular_area = perpendicular_size[0] * perpendicular_size[1]
    
    if pca_area <= original_area and pca_area <= perpendicular_area:
        return interior_center, pca_size, pca_angle
    elif original_area <= perpendicular_area:
        return interior_center, original_size, angle
    else:
        return interior_center, perpendicular_size, perpendicular_angle

def normalize_rotation_angle(angle, width, height):
    """
    Normalize rotation angle according to convention:
    - Neutral pose: longer side along Y axis, shorter side along X axis
    - Return angle in degrees relative to this convention
    
    Args:
        angle: Angle from cv2.minAreaRect (in degrees)
        width: Width of bounding box
        height: Height of bounding box
    
    Returns:
        float: Normalized angle in degrees
    """
    # Ensure we have the longer side as height (Y-axis)
    if width > height:
        # Swap dimensions and adjust angle
        normalized_angle = angle + 90
    else:
        normalized_angle = angle
    
    # Normalize angle to [0, 180) range
    while normalized_angle < 0:
        normalized_angle += 180
    while normalized_angle >= 180:
        normalized_angle -= 180
    
    return normalized_angle

def get_rotated_bbox_corners(center, size, angle):
    """
    Get the four corners of the rotated bounding box
    
    Args:
        center: (cx, cy) center point
        size: (width, height) of the box
        angle: rotation angle in degrees
    
    Returns:
        numpy array: 4x2 array of corner coordinates
    """
    cx, cy = center
    w, h = size
    
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Define corners in local coordinate system (relative to center)
    corners_local = np.array([
        [-w/2, -h/2],  # bottom-left
        [w/2, -h/2],   # bottom-right
        [w/2, h/2],    # top-right
        [-w/2, h/2]    # top-left
    ])
    
    # Rotation matrix
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # Rotate corners and translate to global coordinates
    corners_rotated = corners_local @ rotation_matrix.T
    corners_global = corners_rotated + np.array([cx, cy])
    
    return corners_global

def process_detection_results(json_data):
    """
    Process the detection results and compute rotated bounding boxes
    
    Args:
        json_data: Dictionary containing detection results
    
    Returns:
        list: List of dictionaries containing bbox information
    """
    results = []
    
    for i, mask in enumerate(json_data['masks']):
        # Extract polygon points
        points = mask['points']
        label = mask['label']
        
        print(f"Processing Object {i} ({label}):")
        
        # Compute rotated bounding box with constraint
        center, size, angle = compute_rotated_bbox_constrained(points)
        
        # Verify center is inside polygon
        center_inside = point_in_polygon(center, points)
        print(f"  Center inside polygon: {center_inside}")
        
        if not center_inside:
            print(f"  WARNING: Center is still outside polygon!")
        
        # Normalize angle according to convention
        normalized_angle = normalize_rotation_angle(angle, size[0], size[1])
        
        # Ensure we report the correct dimensions (longer side as height)
        if size[0] > size[1]:
            final_size = (size[1], size[0])  # Swap to (width, height) with height > width
        else:
            final_size = size
        
        # Get corner coordinates
        corners = get_rotated_bbox_corners(center, size, angle)
        
        # Store results
        bbox_info = {
            'object_id': i,
            'label': label,
            'polygon_points': points,
            'center': center,
            'size': final_size,
            'original_angle': angle,
            'normalized_angle': normalized_angle,
            'corners': corners.tolist(),
            'center_inside_polygon': center_inside
        }
        
        results.append(bbox_info)
        
        # Print information
        print(f"  Center: ({center[0]:.1f}, {center[1]:.1f})")
        print(f"  Size: {final_size[0]:.1f} x {final_size[1]:.1f} (W x H)")
        print(f"  Original angle: {angle:.1f}°")
        print(f"  Normalized angle: {normalized_angle:.1f}°")
        print()
    
    return results

def visualize_results(json_data, bbox_results, output_path='rotated_bboxes_visualization.png'):
    """
    Create visualization of the original polygons and rotated bounding boxes
    
    Args:
        json_data: Original detection data
        bbox_results: Processed bounding box results
        output_path: Path to save the visualization
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # Set up the plot with image dimensions
    height = json_data['height']
    width = json_data['width']
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert Y axis to match image coordinates
    ax.set_aspect('equal')
    
    # Colors for different objects
    colors = plt.cm.tab10(np.linspace(0, 1, len(bbox_results)))
    
    for i, (bbox_info, color) in enumerate(zip(bbox_results, colors)):
        # Draw original polygon
        polygon_points = np.array(bbox_info['polygon_points'])
        polygon = Polygon(polygon_points, fill=False, edgecolor=color, 
                         linewidth=1, alpha=0.7, linestyle='--', label=f'Polygon {i}')
        ax.add_patch(polygon)
        
        # Draw rotated bounding box
        corners = np.array(bbox_info['corners'])
        # Close the rectangle by adding the first point at the end
        rect_points = np.vstack([corners, corners[0]])
        
        ax.plot(rect_points[:, 0], rect_points[:, 1], 
               color=color, linewidth=2, alpha=0.8, label=f'Rotated BBox {i}')
        
        # Add center point - use different marker if center is outside polygon
        center = bbox_info['center']
        marker_style = 'o' if bbox_info['center_inside_polygon'] else 'x'
        marker_size = 8 if bbox_info['center_inside_polygon'] else 10
        ax.plot(center[0], center[1], marker_style, color=color, markersize=marker_size)
        
        # Add angle annotation
        ax.annotate(f'{bbox_info["normalized_angle"]:.1f}°', 
                   xy=center, xytext=(center[0]+20, center[1]-20),
                   fontsize=8, color=color, weight='bold',
                   arrowprops=dict(arrowstyle='->', color=color, alpha=0.7))
        
        # Add warning if center is outside
        if not bbox_info['center_inside_polygon']:
            ax.annotate('CENTER OUTSIDE!', 
                       xy=center, xytext=(center[0]+20, center[1]+20),
                       fontsize=8, color='red', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax.set_title('Rotated Bounding Boxes with Interior Center Constraint\n(Angles shown relative to Y-axis convention)', 
                fontsize=14, weight='bold')
    ax.set_xlabel('X coordinate (pixels)')
    ax.set_ylabel('Y coordinate (pixels)')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to: {output_path}")

def main():
    # Example JSON data (replace with your actual JSON file path)
    json_data = {
        "image_path": "/home/rama/bpc_ws/bpc/datasets/ipd_val/000000/images/000000_cam1.jpg",
        "height": 2160,
        "width": 3840,
        "masks": [
            {
                "label": "obj_000018",
                "points": [[2287, 1603], [2285, 1605], [2284, 1605], [2283, 1606], [2282, 1606], [2281, 1607], [2280, 1607], [2278, 1609], [2278, 1610], [2283, 1615], [2283, 1616], [2284, 1617], [2284, 1621], [2285, 1622], [2285, 1625], [2286, 1626], [2286, 1627], [2287, 1628], [2287, 1629], [2289, 1631], [2298, 1631], [2299, 1632], [2301, 1632], [2304, 1635], [2304, 1636], [2305, 1637], [2305, 1641], [2306, 1642], [2308, 1642], [2309, 1643], [2312, 1643], [2314, 1645], [2314, 1647], [2315, 1648], [2315, 1650], [2316, 1651], [2316, 1652], [2317, 1653], [2317, 1655], [2315, 1657], [2314, 1657], [2313, 1658], [2310, 1658], [2309, 1659], [2301, 1659], [2300, 1658], [2299, 1658], [2298, 1659], [2276, 1659], [2275, 1660], [2266, 1660], [2265, 1659], [2255, 1659], [2254, 1658], [2248, 1658], [2247, 1657], [2246, 1657], [2245, 1656], [2244, 1656], [2242, 1654], [2240, 1654], [2239, 1653], [2237, 1653], [2236, 1652], [2227, 1652], [2226, 1651], [2213, 1651], [2212, 1650], [2205, 1650], [2204, 1649], [2201, 1649], [2200, 1648], [2198, 1648], [2197, 1647], [2195, 1647], [2194, 1646], [2193, 1646], [2192, 1645], [2190, 1645], [2189, 1644], [2174, 1644], [2173, 1643], [2167, 1643], [2166, 1642], [2164, 1642], [2163, 1643], [2140, 1643], [2139, 1644], [2135, 1644], [2134, 1645], [2129, 1645], [2128, 1644], [2125, 1644], [2124, 1645], [2123, 1645], [2122, 1646], [2122, 1648], [2121, 1649], [2121, 1652], [2120, 1653], [2120, 1656], [2121, 1657], [2121, 1660], [2122, 1661], [2122, 1662], [2123, 1663], [2123, 1664], [2125, 1666], [2125, 1667], [2128, 1670], [2128, 1671], [2129, 1672], [2129, 1673], [2132, 1676], [2133, 1676], [2134, 1677], [2136, 1677], [2137, 1678], [2138, 1678], [2139, 1679], [2140, 1679], [2141, 1680], [2142, 1680], [2144, 1682], [2145, 1682], [2147, 1684], [2148, 1684], [2150, 1686], [2152, 1686], [2153, 1687], [2154, 1687], [2155, 1688], [2157, 1688], [2158, 1689], [2160, 1689], [2162, 1691], [2163, 1691], [2164, 1692], [2165, 1692], [2166, 1693], [2172, 1693], [2173, 1694], [2197, 1694], [2198, 1695], [2210, 1695], [2211, 1696], [2218, 1696], [2219, 1697], [2222, 1697], [2223, 1698], [2225, 1698], [2226, 1699], [2227, 1699], [2228, 1700], [2230, 1700], [2231, 1701], [2232, 1701], [2233, 1702], [2235, 1702], [2236, 1703], [2240, 1703], [2241, 1704], [2246, 1704], [2247, 1705], [2254, 1705], [2255, 1706], [2259, 1706], [2260, 1707], [2263, 1707], [2264, 1708], [2266, 1708], [2267, 1709], [2268, 1709], [2269, 1710], [2270, 1710], [2271, 1711], [2272, 1711], [2273, 1712], [2275, 1712], [2276, 1713], [2280, 1713], [2281, 1714], [2288, 1714], [2289, 1715], [2290, 1715], [2291, 1716], [2292, 1716], [2296, 1720], [2297, 1720], [2299, 1722], [2300, 1722], [2309, 1731], [2310, 1731], [2313, 1734], [2313, 1735], [2315, 1737], [2315, 1739], [2316, 1740], [2316, 1741], [2317, 1742], [2317, 1743], [2319, 1745], [2319, 1746], [2322, 1749], [2322, 1750], [2324, 1752], [2324, 1753], [2325, 1754], [2325, 1756], [2326, 1757], [2326, 1758], [2327, 1759], [2327, 1761], [2328, 1762], [2328, 1764], [2329, 1765], [2329, 1766], [2331, 1768], [2331, 1769], [2332, 1770], [2332, 1771], [2333, 1772], [2333, 1774], [2334, 1775], [2334, 1776], [2335, 1777], [2335, 1778], [2336, 1779], [2336, 1780], [2337, 1781], [2337, 1783], [2338, 1784], [2338, 1786], [2339, 1787], [2339, 1788], [2341, 1790], [2341, 1791], [2343, 1793], [2344, 1793], [2345, 1794], [2346, 1794], [2348, 1796], [2349, 1796], [2350, 1797], [2356, 1797], [2357, 1798], [2360, 1798], [2361, 1799], [2362, 1798], [2374, 1798], [2375, 1797], [2377, 1797], [2378, 1796], [2379, 1796], [2381, 1794], [2385, 1794], [2386, 1793], [2390, 1793], [2391, 1792], [2391, 1790], [2389, 1788], [2389, 1787], [2388, 1786], [2388, 1782], [2387, 1781], [2387, 1774], [2386, 1773], [2386, 1772], [2385, 1771], [2385, 1769], [2384, 1768], [2384, 1766], [2383, 1765], [2383, 1764], [2382, 1763], [2382, 1761], [2381, 1760], [2381, 1759], [2380, 1758], [2380, 1757], [2379, 1756], [2379, 1755], [2378, 1754], [2378, 1753], [2377, 1752], [2377, 1751], [2376, 1750], [2376, 1748], [2375, 1747], [2375, 1746], [2374, 1745], [2374, 1743], [2373, 1742], [2373, 1741], [2371, 1739], [2371, 1738], [2369, 1736], [2369, 1735], [2368, 1734], [2368, 1733], [2367, 1732], [2367, 1730], [2366, 1729], [2366, 1728], [2365, 1727], [2365, 1722], [2364, 1721], [2364, 1718], [2362, 1716], [2362, 1715], [2357, 1710], [2357, 1706], [2356, 1705], [2356, 1701], [2354, 1699], [2354, 1698], [2350, 1694], [2350, 1693], [2348, 1691], [2348, 1690], [2347, 1689], [2347, 1687], [2346, 1686], [2346, 1684], [2345, 1683], [2345, 1682], [2342, 1679], [2342, 1678], [2340, 1676], [2339, 1676], [2335, 1672], [2335, 1671], [2332, 1668], [2331, 1668], [2330, 1667], [2329, 1667], [2326, 1664], [2326, 1663], [2325, 1662], [2325, 1661], [2323, 1659], [2323, 1658], [2320, 1655], [2321, 1654], [2321, 1645], [2319, 1643], [2318, 1643], [2317, 1642], [2316, 1642], [2314, 1640], [2314, 1634], [2313, 1633], [2313, 1631], [2309, 1627], [2309, 1619], [2306, 1619], [2305, 1620], [2303, 1620], [2302, 1621], [2299, 1621], [2297, 1619], [2297, 1618], [2299, 1616], [2301, 1616], [2302, 1615], [2302, 1609], [2298, 1605], [2295, 1605], [2294, 1604], [2292, 1604], [2291, 1603]],
                "mask_path": "masks/000000_cam1_0.png",
                "bbox": [2110, 1560, 2396, 1814],
                "bbox_center": [2253, 1687],
                "geometric_center": [2282, 1695]
            }
            # Add more masks here from your full JSON...
        ]
    }
    
    # To use with a JSON file, uncomment the following lines:
    with open('/home/rama/bpc_ws/bpc/maskRCNN/results/000000_cam1_annotation.json', 'r') as f:
        json_data = json.load(f)
    
    print("Processing detection results with interior center constraint...")
    bbox_results = process_detection_results(json_data)
    
    print("Creating visualization...")
    visualize_results(json_data, bbox_results)
    
    return bbox_results

if __name__ == "__main__":
    # Install required package if not already installed
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("Please install scikit-learn: pip install scikit-learn")
        exit(1)
    
    results = main()