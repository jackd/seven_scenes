#!/usr/bin/python
"""Demonstrates loading/visualizing the tsdf data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from seven_scenes.tsdf import load_tsdf
from mayavi import mlab


def vis_voxels(voxels, **kwargs):
    data = np.where(voxels)
    if len(data[0]) == 0:
        # raise ValueError('No voxels to display')
        Warning('No voxels to display')
    else:
        if 'mode' not in kwargs:
            kwargs['mode'] = 'cube'
        mlab.points3d(*data, **kwargs)


tsdf = load_tsdf('chess')
data = tsdf.data
print('data: dtype: %s, min: %d, max: %d'
      % (data.dtype, np.min(data), np.max(data)))

# speeds things up a bit
data = data[::4, ::4, ::4]

data[data == 0] = 16384
voxels = data < 5
vis_voxels(voxels, color=(0, 0, 1))

plt.figure()
plt.imshow(data[..., data.shape[-1]//2])
plt.title('midlevel z-slice')

data = np.reshape(data, (-1,))
data = data[np.logical_and(np.abs(data) < 10000, data != 0)]
plt.figure()
plt.hist(np.reshape(data, (-1,)))
plt.title('Cropped value distribution')
plt.show(block=False)

mlab.show()
plt.close()
