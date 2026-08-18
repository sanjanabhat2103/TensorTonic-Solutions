import numpy as np

def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    """
    points = np.asarray(points)
    is_single_point = (points.ndim == 1)
    if is_single_point:
        points = points[np.newaxis, :]
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    x_new = x * np.cos(theta) - y * np.sin(theta)
    y_new = x * np.sin(theta) + y * np.cos(theta)
    z_new = z
    result = np.stack([x_new, y_new, z_new], axis=-1)
    return result[0] if is_single_point else result