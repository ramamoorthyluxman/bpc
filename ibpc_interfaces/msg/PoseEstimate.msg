# ID of the scene.
uint64 scene_id

# ID of the image.
uint64 im_id

# ID of the object.
uint64 obj_id

# A confidence of the estimate (the range of confidence values is not restricted)
float64 score

# A 3x3 rotation matrix whose elements are saved row-wise.
# loat64[9]  R # 3x3 row-major matrix

# A 3x1 translation vector (in mm).
# float64 t[3]

# 6D pose of object.
geometry_msgs/PoseStamped pose

# The time the method took to estimate poses for all objects in image im_id
# from scene scene_id. All estimates with the same scene_id and im_id must have
# the same value of time. Report the wall time from the point right after the
# raw data (the image, 3D object models etc.) is loaded to the point when the
# final pose estimates are available.
builtin_interfaces/Time time
