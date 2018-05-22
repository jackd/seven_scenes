#!/usr/bin/python
"""Script for visualizing the difference camera poses from blue to green."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import division
import numpy as np
from seven_scenes.camera import get_camera_matrix, image_shape
from seven_scenes.scene import Scene
from mayavi import mlab


def vis_axes():
    mlab.quiver3d([0], [0], [0], [1], [0], [0], color=(1, 0, 0))
    mlab.quiver3d([0], [0], [0], [0], [1], [0], color=(0, 1, 0))
    mlab.quiver3d([0], [0], [0], [0], [0], [1], color=(0, 0, 1))


def vis_origin():
    vis_points([0, 0, 0], color=(0, 0, 0), scale_factor=0.2)


def vis_points(p, **kwargs):
    if p.shape == (3,):
        p = np.expand_dims(p, axis=0)
    mlab.points3d(*p.T, **kwargs)


def vis_normals(points, normals, **kwargs):
    x, y, z = points.T
    u, v, w = normals.T
    mlab.quiver3d(x, y, z, u, v, w, **kwargs)


def vis_contours(sdf_vals, coords, contours=[0]):
    x, y, z = (coords[..., i] for i in range(3))
    mlab.contour3d(x, y, z, sdf_vals, contours=contours, transparent=True)


def vis_rays(offset, directions, **kwargs):
    """
    Based on mayavi doc.

    http://docs.enthought.com/mayavi/mayavi/auto/example_plotting_many_lines.html
    """
    start = np.tile(np.expand_dims(
        offset, axis=0), (directions.shape[0], 1))
    end = start + directions
    start_end = np.stack((start, end), axis=1)
    connections = []
    x = []
    y = []
    z = []
    index = 0
    for se in start_end:
        x.append(se[:, 0])
        y.append(se[:, 1])
        z.append(se[:, 2])
        N = len(se)
        connections.append(np.vstack(
                   [np.arange(index,   index + N - 1.5),
                    np.arange(index + 1, index + N - .5)]).T)
        index += N

    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)
    connections = np.vstack(connections)
    # Create the points
    src = mlab.pipeline.scalar_scatter(x, y, z)

    # Connect them
    src.mlab_source.dataset.lines = connections
    src.update()

    # The stripper filter cleans up connected lines
    lines = mlab.pipeline.stripper(src)

    # Finally, display the set of lines
    mlab.pipeline.surface(
        lines, line_width=0.2, opacity=.4, **kwargs)


def get_rays():
    K = get_camera_matrix()
    h, w = image_shape
    tl = np.linalg.solve(K, np.array([0, 0, 1], dtype=np.float32))
    br = np.linalg.solve(K, np.array([w-1, h-1, 1], dtype=np.float32))

    xmin = tl[0]
    xmax = br[0]
    ymin = tl[1]
    ymax = br[1]

    x = np.linspace(xmin, xmax, w)
    y = np.linspace(ymin, ymax, h)
    x, y = np.meshgrid(x, y, indexing='ij')
    z = -np.ones_like(x)
    rays = np.stack((x, y, z), axis=-1)
    return rays


def get_data(n, scene_id='chess', sequence_index=2):
    scene = Scene(scene_id)
    with scene.get_sequence(sequence_index) as sequence:
        n_frames = sequence.n_frames
        for frame_index in range(0, n_frames, n_frames // n):
            frame = sequence.get_frame(frame_index)
            yield frame.rgb(), frame.depth(), frame.pose()


rays = get_rays()
# rays = rays[[0, 0, -1, -1], [0, -1, 0, -1]]
rays = rays[::16, ::16]
rays = np.reshape(rays, (-1, 3))
n = 8
for i, (rgb, depth, pose) in enumerate(get_data(n)):
    R = pose[:3, :3]
    t = pose[:3, 3]
    rays = np.matmul(rays, R)
    alpha = i / (n-1)
    color = (0, alpha, 1-alpha)
    vis_rays(t, rays, color=color)

vis_axes()
mlab.show()
