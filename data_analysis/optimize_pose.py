import funcs

def estimate_pose(image, polygon_coords, mesh, cam_K, init_tvec):
    mesh_path = self.mesh_dir + '/obj_' + str(f"{object_id:06d}") + '.ply'
    image_size = (int(self.scene_data[row_num]['image_width']), int(self.scene_data[row_num]['image_height']))  
    polygon_mask_coordinates = self.scene_data[result['scene_cam_row_num']]['detection_json']['results'][0]['masks'][result['mask_idx']]['points']
    image = self.scene_data[result['scene_cam_row_num']]["rgb_img"]

    mask_image = funcs.create_polygon_mask(image, polygon_mask_coordinates)