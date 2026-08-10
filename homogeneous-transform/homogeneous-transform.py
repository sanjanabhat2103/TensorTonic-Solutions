import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    points = np.asarray(points)
    single_point = points.ndim == 1
    if single_point:
        points = points[None, :]
    homogeneous = np.hstack([
        points,
        np.ones((points.shape[0], 1))
    ])
    transformed = (T @ homogeneous.T).T
    transformed = transformed[:, :3] / transformed[:, 3, None]
    return transformed[0] if single_point else transformed