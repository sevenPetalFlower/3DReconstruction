import os.path

import cv2
import numpy as np

def calculateDisparityMap(img_left, img_right):
    # Create StereoBM object and compute disparity
    num_disparities = 16 * 6  # must be divisible by 16
    block_size = 15  # must be odd, typically in 5..51

    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

    # Normalize for visualization
    disp_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_normalized = np.uint8(disp_normalized)
    return disp_normalized


if __name__=="__main__":
    current_directory = os.getcwd()
    image_folder = os.path.join(current_directory, "courtyard_dslr_undistorted",
                                    "courtyard", "images", "dslr_images_undistorted")
    img_left_path = os.path.join(image_folder, 'DSC_0286.JPG')
    img_right_path = os.path.join(image_folder, 'DSC_0287.JPG')

    # Load rectified grayscale stereo pair
    img_left = cv2.imread(img_left_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(img_right_path, cv2.IMREAD_GRAYSCALE)
    # Check if images loaded properly
    if img_left is None or img_right is None:
        raise ValueError("Could not load input images!")

    disp_normalized = calculateDisparityMap(img_left, img_right)
