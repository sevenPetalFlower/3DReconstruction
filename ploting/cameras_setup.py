import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo

def plot_stereo_camera_setup(R, t):
    # For visualization, assume camera1 is at the origin (0,0,0) with an identity rotation.
    cam1_center = np.array([0, 0, 0])
    # Compute the second camera center: when a point is observed as X_cam2 = R * X_cam1 + t,
    # the camera center (in world coordinates) is given by -R^T * t.
    cam2_center = -R.T @ t

    print("Camera 1 center:", cam1_center.ravel())
    print("Camera 2 center:", cam2_center.ravel())

    # -------------------------------------------
    # 7. 3D Visualization using Plotly: Plot camera centers and axes
    # -------------------------------------------
    def create_camera_marker(center, rotation, scale=0.1, name='Camera'):
        """
        Creates Plotly traces for a camera marker (center point and coordinate axes).
        - 'center': The 3D position of the camera center.
        - 'rotation': The 3x3 rotation matrix defining the camera orientation.
        - 'scale': Length of the axes vectors.
        """
        traces = []

        # Scatter the camera center
        traces.append(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode='markers',
            marker=dict(size=5),
            name=name
        ))

        # Define axes directions: columns of the rotation matrix (X, Y, Z)
        axes_labels = ['X', 'Y', 'Z']
        colors = ['red', 'green', 'blue']
        for idx, color in enumerate(colors):
            axis_vector = rotation[:, idx] * scale
            axis_end = center + axis_vector.ravel()
            traces.append(go.Scatter3d(
                x=[center[0], axis_end[0]],
                y=[center[1], axis_end[1]],
                z=[center[2], axis_end[2]],
                mode='lines',
                line=dict(color=color, width=4),
                name=f'{name} {axes_labels[idx]}'
            ))
        return traces

    # Camera 1: at origin with identity rotation.
    cam1_traces = create_camera_marker(cam1_center, np.eye(3), scale=0.2, name='Camera1')

    # Camera 2: position and rotation from the pose decomposition.
    cam2_traces = create_camera_marker(cam2_center.ravel(), R, scale=0.2, name='Camera2')

    # Combine all traces and set up the plot.
    data = cam1_traces + cam2_traces
    layout = go.Layout(
        title='Camera Positions and Orientations (Different Calibrations)',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data'
        ),
        width=800,
        height=600
    )
    fig = go.Figure(data=data, layout=layout)
    pyo.plot(fig, filename='camera_positions_diff_calibration.html')