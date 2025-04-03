import cv2
import os
from algorithms.structureFromMotion import getMatches, safeMatches

# Get the current working directory
current_directory = os.getcwd()

path_to_test_img = os.path.join(current_directory, "courtyard_dslr_undistorted",
                                "courtyard", "images", "dslr_images_undistorted")

left_img_name = "DSC_0286.JPG"
right_img_name = "DSC_0287.JPG"
left_img = os.path.join(path_to_test_img, left_img_name)
right_img = os.path.join(path_to_test_img, right_img_name)

# Load two images (ensure they are from the same scene with overlapping views)
img1 = cv2.imread(left_img, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE)

matches, kp1, kp2 = getMatches(img1, img2)
safeMatches(img1, img2, matches, kp1, kp2, 50, "keyPointsMatch.png")

