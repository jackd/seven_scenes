from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

principle_points = (320, 240)  # x, y
focal_length = (585, 585)
image_shape = (480, 640)  # H, W


def get_camera_matrix(dtype=np.float32):
    fx, fy = focal_length
    x0, y0 = principle_points
    s = 0
    K = np.array([
        [fx, s, x0],
        [0, fy, y0],
        [0, 0, 1]
    ], dtype=dtype)
    # if homogeneous:
    #     data = np.zeros((3, 4), dtype=dtype)
    #     data[:, :3] = get_camera_matrix(dtype)
    #     K = data
    return K
