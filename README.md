# 3DReconstruction

This repository implements a basic multi-view 3D reconstruction pipeline based on epipolar geometry, structure from motion, and disparity/depth map computation. 
It is designed to work with ETH3D dataset.

---

## 🛠️ How to Setup

1. **Download Dataset**  
   Download the `courtyard_dslr_undistorted` dataset from the [ETH3D High-Res Multi-View Stereo Benchmark](https://www.eth3d.net/datasets#high-res-multi-view).

2. **Directory Setup**  
   Place the extracted folder in the root of the repository with the following structure:

   ```
   3DReconstruction/
   ├── courtyard_dslr_undistorted/
   │   └── courtyard/
   │       └── images/
   │           └── dslr_images_undistorted/
   └── ...
   ```
---

## 📁 Project Structure

| File / Module | Description |
|---------------|-------------|
| `structureFromMotionPipeline.py` | **Main script** that run the Structure-from-Motion pipeline. |
| `resize.py` | Resizes input images to 1/4 of the original size. |
| `calculateDepthMaps.py` | Computes depth maps from stereo image pairs using MiDaS. |
| `calculateDisparityMap.py` | Computes the disparity map using OpenCV StereoBM. |
| `get2dPointCorrespondence.py` | Extracts and matches 2D keypoints between image pairs using OpenCV SIFT algorithm. |
| `cameraExtrinsicEpipolarGeoetry.py` | Computes camera extrinsic parametes using Epipoar geometry. |

### Submodules

#### `algorithms/`

| File | Description |
|------|-------------|
| `structureFromMotion.py` | Contains structure from motion algorithms. |
| `epipolarGeometry.py` | Contains epipolar geometry logic. |

#### `utils/`

| File | Description |
|------|-------------|
| `datasetOperations.py` | Utility functions for loading image data, camera parameters, and parsing datasets. |

#### `ploting/`

| File | Description |
|------|-------------|
| `cameras_setup.py` | Visualization utilities to plot camera poses and 3D points using Open3D. |
