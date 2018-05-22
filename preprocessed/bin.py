from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from .path import bin_dir

image_shape = (160, 214)
max_gt_value = 5157  # see example/max_preprocessed


def get_filename_split(mode):
    if mode == 'train':
        fn = 'train.lst'
    elif mode in ('eval', 'test', 'predict'):
        fn = 'val.lst'
    path = os.path.join(bin_dir, fn)
    with open(path, 'r') as fp:
        for line in fp.readlines():
            line = line.rstrip()
            if len(line) > 0:
                yield line.split()[1:]


def parse_bin_filename(filename):
    if filename[-4:] != '.bin':
        raise ValueError('Not a binary file')
    if filename[:5] == 'noisy':
        ground_truth = False
        if filename[5] == '_':
            start = 6
        else:
            start = 5
    else:
        assert(filename[:12] == 'groundtruth_')
        ground_truth = True
        start = 12
    rest = filename[start:-4]
    scene_id = rest[:-6]
    if scene_id[-1] == '_':
        scene_id = scene_id[:-1]
    example_id = int(rest[-6:])
    return scene_id, example_id, ground_truth


def index_filenames():
    """
    Get a nested dict of filenames indexed by scene_id/example_id.

    Example usage:
    filename_index = index_filenames()
    raw_fn, ground_truth_fn = filename_index['chess'][1]
    """
    filename_index = {}
    filenames = get_bin_filenames()

    for fn in filenames:
        if fn[-4:] == '.bin':
            scene_id, example_id, gt = parse_bin_filename(fn)
            filename_index.setdefault(
                scene_id, {}).setdefault(example_id, [None, None])[gt] = fn
    return filename_index


def get_bin_filenames():
    return os.listdir(bin_dir)


def get_bin_path(filename):
    return os.path.join(bin_dir, filename)


def load_bin_data(filename):
    buff = np.fromfile(get_bin_path(filename), dtype=np.int16)
    image = np.reshape(buff[2:], buff[:2])
    return image
