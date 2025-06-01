import cv2
import numpy as np
import open3d as o3d
import cv2
from algorithms.structureFromMotion import get_matches, safe_matches
from algorithms.epipolarGeometry import estimate_projection_matrices
from utils.datasetOperations import get_test_2_images, get_predicted_test_extrinsic, get_test_images_calibration

img1, img2 = get_test_2_images()
imgGray1 = cv2.cvtColor(img1, cv2.IMREAD_GRAYSCALE)
imgGray2 = cv2.cvtColor(img2, cv2.IMREAD_GRAYSCALE)

matches, kp1, kp2 = get_matches(imgGray1, imgGray2)
safe_matches(img1, img2, matches, kp1, kp2, 50, "keyPointsMatchBig.png")



##########################################


# Matched 2D keypoints in two images
pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
print(np.size(pts1))

# === Triangulation ===
P1, P2, F, mask = estimate_projection_matrices(pts1, pts2)

pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
pts3d = (pts4d[:3] / pts4d[3]).T

# === Create Point Cloud ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts3d)
pcd.estimate_normals()

# === Refining: Denoising and Downsampling ===
#pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=2, std_ratio=1.0)
#pcd = pcd.voxel_down_sample(voxel_size=0.01)

# === Meshing: Poisson Reconstruction ===
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
mesh.compute_vertex_normals()

# === Optional: Basic Coloring ===
# Replace with your actual image used for projection
h, w, _ = img1.shape
colors = []

for pt in mesh.vertices:
    pt = np.array(pt)
    proj = K1 @ pt
    proj = proj[:2] / proj[2]
    u, v = int(proj[0]), int(proj[1])
    if 0 <= u < w and 0 <= v < h:
        color = img1[v, u] / 255.0
    else:
        color = [0.5, 0.5, 0.5]
    colors.append(color)

mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

# === Save Outputs ===
o3d.io.write_point_cloud("triangulated_refined.ply", pcd)
o3d.io.write_triangle_mesh("meshed_colored.obj", mesh)
print("Saved")
