import open3d as o3d

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

def close(vis):
    vis.close()

o3d.visualization.draw_geometries_with_key_callbacks([mesh], {ord("Q"): close})
