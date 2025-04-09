import os
import cv2
import numpy as np

def get_test_2_images():
    current_directory = os.getcwd()
    path_to_test_img = os.path.join(current_directory, "courtyard_dslr_undistorted",
                                    "courtyard", "images", "dslr_images_undistorted")

    left_img_name = "DSC_0323.JPG"
    right_img_name = "DSC_0322.JPG"
    left_img = os.path.join(path_to_test_img, left_img_name)
    right_img = os.path.join(path_to_test_img, right_img_name)
    img1 = cv2.imread(left_img)
    img2 = cv2.imread(right_img)
    return img1, img2

def get_test_images_calibration():
    K0 = np.array([
        [3411.42, 0, 3116.72],
        [0, 3410.02, 2062.52],
        [0, 0, 1]
    ], dtype=np.float32)

    K1 = np.array([
        [3409.58, 0, 3115.16],
        [0, 3409.44, 2064.73],
        [0, 0, 1]
    ], dtype=np.float32)
    return K0, K1


def get_predicted_test_extrinsic():
    R = np.array([
        [-0.93181539, -0.1529001, -0.32915291],
        [0.32449636, -0.75718143, -0.56690246],
        [-0.16254903, -0.63505736, 0.75516883]
    ], dtype=np.float32)

    t = np.array([-0.1960961, -0.08981855, 0.97646247], dtype=np.float32)
    return R, t