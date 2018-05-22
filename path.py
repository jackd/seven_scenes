from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def get_data_dir():
    key = 'SEVEN_SCENES_PATH'
    if key in os.environ:
        dataset_dir = os.environ[key]
        if not os.path.isdir(dataset_dir):
            raise Exception('%s directory does not exist' % key)
        return dataset_dir
    else:
        raise Exception('%s environment variable not set.' % key)


root_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)))


def get_zip_path(scene_id):
    return os.path.join(get_data_dir(), '%s.zip' % scene_id)


def get_tsdf_dir():
    return os.path.join(get_data_dir(), 'tsdf')


def get_scene_ids():
    ids = os.listdir(get_data_dir())
    ids = [i for i in ids if i[-4:] == '.zip']
    ids.sort()
    ids = [i[:-4] for i in ids]
    return ids


def get_scene_dir(scene_id):
    return os.path.join(get_data_dir(), scene_id)


def get_split_subpath(mode):
    mode = mode.lower()
    if mode == 'train':
        return 'TrainSplit.txt'
    elif mode in ('eval', 'test', 'predict'):
        return 'TestSplit.txt'
    else:
        raise ValueError('mode "%s" not recognized' % mode)


def get_split_path(scene_dir, mode):
    return os.path.join(scene_dir, get_split_subpath(mode))


def get_sequence_id(index):
    return 'seq-%02d' % (index + 1)


def get_sequence_subpath(index):
    return '%s.zip' % get_sequence_id(index)


def get_sequence_path(scene_id, index):
    return os.path.join(scene_id, get_sequence_subpath(index))
