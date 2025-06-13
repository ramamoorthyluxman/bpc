from utilities import utils
import cv2
img1 = cv2.imread("/home/rama/bpc_ws/bpc/pointcloud_registration/output/test_images/test_06_obj_000018_polygon.png")
img2 = cv2.imread("/home/rama/bpc_ws/bpc/pointcloud_registration/output/reference_images/ref_01_obj_000018_01_cam1_polygon.png")

homography_matrix, image_matches = utils.get_homography(img1, img2)

cv2.imwrite("matches.png", image_matches)

if homography_matrix is None:
    print("No valid homography found")

else:
    transformed_image = cv2.warpPerspective(img2, homography_matrix, (img1.shape[1], img1.shape[0]))

    cv2.imwrite("transformed image.png", transformed_image)