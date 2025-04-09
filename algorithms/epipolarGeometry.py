import numpy as np
import cv2

# Converting points from mage frame to camera coordinate system.
# By applying inverse calibration matrix
def from_image_to_camera_coord(pts, K):
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])  # homogeneous coordinates
    pts_norm = (np.linalg.inv(K) @ pts_h.T).T  # normalize: X_norm = inv(K) * x
    return pts_norm[:, :2]

def predict_essential_matrix_from_point_correspondence(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(pts1, pts2,
                                   cameraMatrix=K,
                                   method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    return E, mask


def filter_point_crrespondence_essential_matrix(pts1, pts2, mask):
    inlier_mask = mask.ravel() == 1
    pts1_essential = pts1[inlier_mask]
    pts2_essential = pts2[inlier_mask]
    return pts1_essential, pts2_essential


def recoverExtrinsic(E, pts1_cam_essentail, pts2_cam_essentail, K):
    _, R, t, mask_pose = cv2.recoverPose(E, pts1_cam_essentail, pts2_cam_essentail, cameraMatrix=K)
    return R, t
