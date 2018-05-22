#!/usr/bin/python
"""Example usage of `preprocessed.bin.index_filenames` fn."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import seven_scenes.preprocessed.bin as b

filename_index = b.index_filenames()

for scene_id, examples in filename_index.items():
    print(scene_id, len(examples))
    for example_id, fns in examples.items():
        if any((f is None for f in fns)):
            print(scene_id, example_id, fns)

raw, gt = filename_index['chess'][1]


def vis(raw, gt):
    import matplotlib.pyplot as plt
    raw_data = b.load_bin_data(raw)
    gt_data = b.load_bin_data(gt)
    print(raw_data.shape)
    print(gt_data.shape)

    plt.figure()
    plt.imshow(raw_data, cmap='gray')
    plt.title(raw)
    plt.figure()
    plt.imshow(gt_data, cmap='gray')
    plt.title(gt)
    plt.show()


vis(raw, gt)
