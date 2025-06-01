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


def estimate_projection_matrices(pts1, pts2):
    """
    Estimate projection matrices P1 and P2 from 2D correspondences (projective case).
    No intrinsics are required.
    """

    # Step 1: Estimate Fundamental matrix using 8-point algorithm
    F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_8POINT)

    if F is None:
        raise ValueError("Fundamental matrix could not be estimated.")

    # Step 2: Construct P1 and P2 from F
    # P1 is identity (canonical)
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))

    # Step 3: Compute epipole in second image (right null space of F)
    U, S, Vt = np.linalg.svd(F.T)
    e = Vt[-1]
    e = e / e[2]  # Normalize to homogeneous

    # Step 4: Skew-symmetric matrix of epipole
    ex = np.array([
        [0, -e[2], e[1]],
        [e[2], 0, -e[0]],
        [-e[1], e[0], 0]
    ])

    # Step 5: Build P2 = [e]_x * F | e
    P2 = np.hstack((ex @ F, e.reshape(3, 1)))

    return P1, P2, F, mask