import cv2
import numpy as np
from utils.datasetOperations import get_test_2_images, get_test_images_calibration
from algorithms.structureFromMotion import get_matches, get_n_best_matches, safe_matches
from algorithms.epipolarGeometry import (from_image_to_camera_coord, recoverExtrinsic,
                                         predict_essential_matrix_from_point_correspondence,
                                         filter_point_crrespondence_essential_matrix)
from ploting.cameras_setup import plot_stereo_camera_setup

# -------------------------------------------
# 0. Parameters
# -------------------------------------------
numKeyPoints = 100 # Num key points used to reconstruct the esetial matrix
keyPointCorrespondanceName = "keyPointsMatch.png"
isSafe = False
isPlotting = True

# -------------------------------------------
# 1. Load images and specify calibration matrices
# -------------------------------------------
img1, img2 = get_test_2_images()
img1 = cv2.cvtColor(img1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.cvtColor(img2, cv2.IMREAD_GRAYSCALE)
K1, K2 = get_test_images_calibration()
print("Laded data")

# -------------------------------------------
# 2. Detect keypoints and compute descriptors using SIFT
# -------------------------------------------
matches, kp1, kp2 = get_matches(img1, img2)
matches, kp1, kp2  = get_n_best_matches(matches, kp1, kp2, numKeyPoints)
print("Key points detected")
if isSafe:
    safe_matches(img1, img2, matches, kp1, kp2, 50, "keyPointsMatch.png")

# -------------------------------------------
# 4. Compute the Essential matrix
# -------------------------------------------
pts1_cam = from_image_to_camera_coord(kp1, K1)
pts2_cam = from_image_to_camera_coord(kp2, K2)
# Since the points are camera coordinates, we use an identity camera matrix.
E, mask = predict_essential_matrix_from_point_correspondence(pts1_cam, pts2_cam, np.eye(3))
pts1_cam_essentail, pts2_cam_essentail = filter_point_crrespondence_essential_matrix(pts1_cam,
                                                                                     pts2_cam, mask)
print("Number of keypoints in image 1 after filtering by Essential matrix:", len(pts1_cam))
print("Number of keypoints in image 2 after filtering by Essential matrix:", len(pts2_cam))

# -------------------------------------------
# 6. Decompose the Essential matrix to obtain R and t
# -------------------------------------------
R, t = recoverExtrinsic(E, pts1_cam_essentail, pts2_cam_essentail, np.eye(3))
print("Rotation")
print(R)
print("Translation")
print(t)

# Plot camera setup
if isPlotting:
    plot_stereo_camera_setup(R, t)
