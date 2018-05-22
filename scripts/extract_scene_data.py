#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from seven_scenes.path import get_scene_ids
from seven_scenes.scene import extract_scene_data
from seven_scenes.tsdf import extract_tsdf

for scene_id in get_scene_ids():
    extract_scene_data(scene_id)

extract_tsdf()
