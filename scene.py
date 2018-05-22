from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from PIL import Image
import numpy as np
import zipfile
from .path import get_zip_path, get_split_path
from .path import get_sequence_id, get_scene_dir

invalid_depth = 65535


def _load_image_from_zip(zf, subpath):
    from PIL import Image
    return Image.open(zf.open(subpath))


def extract_scene_data(scene_id):
    path = get_zip_path(scene_id)
    folder = os.path.dirname(path)

    print('Extracting scene data: "%s"' % scene_id)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(folder)
    print('Done!')


class Scene(object):
    def __init__(self, scene_id):
        self._scene_id = scene_id
        self._folder = get_scene_dir(scene_id)
        if not os.path.isdir(self._folder):
            extract_scene_data(scene_id)

    @property
    def scene_id(self):
        return self._scene_id

    @property
    def folder(self):
        return self._folder

    @property
    def n_sequences(self):
        return len(os.listdir(self.folder)) - 3

    def get_image(self):
        path = os.path.join(self.folder, '%s.png' % self.scene_id)
        return Image.open(path)

    def get_sequence(self, index):
        return Sequence(self, index)

    def get_sequences(self, mode):
        return tuple(self.get_sequence(i) for i in self.get_split(mode))

    def get_all_sequences(self):
        return tuple(self.get_sequence(i) for i in range(self.n_sequences))

    def get_split(self, mode):
        with open(get_split_path(self.folder, mode), 'r') as fp:
            for line in fp.readlines():
                line = line.rstrip()
                if len(line) > 0:
                    yield (int(line[8:])) - 1


class Sequence(object):
    def __init__(self, scene, index):
        self._scene = scene
        self._index = index
        self._zip = None
        self._sequence_id = get_sequence_id(self._index)

    @property
    def scene(self):
        return self._scene

    @property
    def scene_id(self):
        return self._scene._scene_id

    @property
    def sequence_id(self):
        return self._sequence_id

    @property
    def index(self):
        return self._index

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def open(self):
        self._zip = zipfile.ZipFile(os.path.join(
            self.scene.folder, '%s.zip' % self.sequence_id), 'r')

    def close(self):
        self._zip.close()

    @property
    def n_frames(self):
        return (len(self._zip.namelist()) - 1) // 3

    def get_pose(self, frame_index, dtype=np.float32):
        subpath = os.path.join(
            self.sequence_id, 'frame-%06d.pose.txt' % frame_index)
        with self._zip.open(subpath, 'r') as fp:
            data = tuple(float(f) for f in fp.read().split())
        return np.array(data, dtype=dtype).reshape(4, 4)

    def get_depth(self, frame_index):
        return _load_image_from_zip(
            self._zip,
            '%s/frame-%06d.depth.png' % (self.sequence_id, frame_index))

    def get_rgb(self, frame_index):
        return _load_image_from_zip(
            self._zip,
            '%s/frame-%06d.color.png' % (self.sequence_id, frame_index))

    def get_frame(self, frame_index):
        return Frame(self, frame_index)


class Frame(object):
    def __init__(self, sequence, index):
        self._sequence = sequence
        self._index = index

    @property
    def index(self):
        return self._index

    @property
    def sequence(self):
        return self._sequence

    def rgb(self):
        return self._sequence.get_rgb(self._index)

    def depth(self):
        return self._sequence.get_depth(self._index)

    def pose(self):
        return self._sequence.get_pose(self._index)


get_scene = Scene
