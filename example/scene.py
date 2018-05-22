#!/usr/bin/python
"""Demonstrates usage of `scene.Scene`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from seven_scenes.scene import Scene, invalid_depth

scene = Scene('chess')

print('Total number of sequences: %d' % scene.n_sequences)
for mode in ('train', 'test'):
    print('%s: %s' % (mode, tuple(scene.get_split(mode))))


with scene.get_sequence(0) as sequence:
    n = sequence.n_frames
    depth = sequence.get_depth(0)
    depth = np.array(depth)
    plt.figure()
    plt.imshow(depth, cmap='gray')
    plt.title('original depth')
    depth[depth == invalid_depth] = 4000
    plt.figure()
    plt.imshow(depth, cmap='gray')
    plt.title('valid depth')
    plt.show()
