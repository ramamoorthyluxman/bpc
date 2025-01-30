# ID of the object.
uint64 obj_id

# A confidence of the estimate (the range of confidence values is not restricted)
float64 score

# A 3x3 rotation matrix whose elements are saved row-wise.
# float64[9]  R # 3x3 row-major matrix

# A 3x1 translation vector (in mm).
# float64 t[3]

# 6D pose of object.
geometry_msgs/PoseStamped pose
