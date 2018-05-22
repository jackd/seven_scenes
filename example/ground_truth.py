#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from mayavi import mlab
import numpy as np
from transformations import euler_from_matrix
from seven_scenes.camera import get_camera_matrix, image_shape
from seven_scenes.tsdf import load_tsdf
from seven_scenes.scene import Scene, invalid_depth
import sdf_renderer.vis as vis

skip = 8


def get_scene_data(scene_id='chess', sequence_index=0, frame_index=200):
    tsdf = load_tsdf(scene_id)
    scene = Scene(scene_id)
    with scene.get_sequence(sequence_index) as sequence:
        frame = sequence.get_frame(frame_index)
        rgb = frame.rgb()
        depth = frame.depth()
        pose = frame.pose()
    return tsdf, rgb, depth, pose


def get_rays():
    K = get_camera_matrix()
    h, w = image_shape
    # tl = np.matmul(K, np.array([0, 0, -1]))
    # br = np.matmul(K, np.array([w-1, h-1, -1]))
    tl = np.linalg.solve(K, np.array([0, 0, 1], dtype=np.float32))
    br = np.linalg.solve(K, np.array([w-1, h-1, 1], dtype=np.float32))
    br /= br[-1]
    tl /= tl[-1]
    xmin, ymin = tl[:2]
    xmax, ymax = br[:2]

    x = np.linspace(xmin, xmax, w)
    y = np.linspace(ymin, ymax, h)
    x = x[::skip]
    y = y[::skip]
    x, y = np.meshgrid(x, y, indexing='ij')
    z = np.ones_like(x)
    rays = np.stack((x, y, z), axis=-1)
    # rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    rays = np.reshape(rays, (-1, 3))
    return rays


def sphere_trace(sdf_fn, offset, rays, max_dist=5000, tol=1):
    dist = sdf_fn(np.expand_dims(offset, axis=0))[0]
    assert(dist.shape == ())
    print(dist)
    exit()
    if dist == invalid_depth:
        return np.ones(rays.shape[:-1], dtype=np.int32)*invalid_depth
    elif dist == 0:
        return np.zeros(rays.shape[:-1], dtype=np.int32)
    x = offset + rays*dist
    converged = False
    dist = np.expand_dims(dist, 0)
    dist = np.tile(dist, (len(rays),))
    i = 0
    while not converged:
        dist += sdf_fn(x)
        x[:] = offset + rays*np.expand_dims(dist, axis=1)
        converged = np.all(np.logical_or(dist <= tol, dist >= max_dist))
        i += 1
    return dist


tsdf, rgb, depth, pose = get_scene_data()
# rgb = np.array(rgb)[::skip, ::skip]
depth = np.array(depth)
ray_depth = np.reshape(depth[-1::-skip, ::skip].T, (-1,)) / 1000
upper_depth = 4
ray_depth[ray_depth > upper_depth] = upper_depth

print('mid depth: %d' % depth[depth.shape[0] // 2, depth.shape[1] // 2])
# print(euler_from_matrix(pose))
# print(t)
angles = list(euler_from_matrix(pose))
print('angles: %s' % str(angles))
print('t: %s' % str(pose[:3, 3]))
# angles[1] *= -1
# pose[:3, :3] = euler_matrix(*angles)[:3, :3]
rays0 = get_rays()
s = 8
i, j, k = np.where(tsdf.data[::s, ::s, ::s] < 0)
i *= s
j *= s
k *= s
ijk = np.stack((i, j, k), axis=-1)
xyz = tsdf.ijk_to_xyz(ijk)

R = pose[:3, :3]
t = pose[:3, 3]
# rays = np.matmul(rays, R)

# t2 = -np.matmul(R.T, t)
# rays2 = np.matmul(rays0, R)
rays = np.matmul(rays0, R)

# dist = sphere_trace(tsdf.get_signed_distance, x, r)

# plt.figure()
depth[depth == invalid_depth] = 5000
# plt.imshow(depth)
# plt.show()
plt.figure()
plt.imshow(rgb)
plt.show(block=False)

vis.vis_axes()
# x, y, z = xyz.T
# t2 = -t2
print('t', t)
# t[1] *= 0
# t2[1] = 0.38
# t = np.matmul(R, t)

# x, y, z = xyz.T
z, y, x = xyz.T
# x, z, y = xyz.T
# z, x, y = xyz.T
# x *= -1
# y *= -1
# z *= -1
rays[..., 1] *= -1
t[1] *= -1
mlab.points3d(x, y, z, color=(0, 0, 1), scale_factor=0.05)
vis.vis_rays(np.zeros((3,)), rays0, color=(1, 0, 0))
vis.vis_rays(t, rays)

scaled_rays = np.expand_dims(ray_depth, -1) * rays
end_points = t + scaled_rays
valid_rays = ray_depth < upper_depth
vis.vis_points(end_points[valid_rays], color=(0, 1, 0), scale_factor=0.02)
vis.vis_points(
    end_points[np.logical_not(valid_rays)], color=(1, 0, 0), scale_factor=0.02)

mlab.show()
plt.close()
