from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from .path import get_tsdf_dir
from .scene import invalid_depth


def extract_tsdf(zip_path=None):
    import zipfile
    if zip_path is None:
        zip_path = '%s.zip' % get_tsdf_dir()
    if not os.path.isfile(zip_path):
        raise IOError('No tsdf data found at %s' % zip_path)
    print('Extracting %s...' % zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        base = os.path.dirname(zip_path)
        zf.extractall(base)
    print('Done!')


_parse_fns = {
    'NDims': int,
    'DimSize': lambda x: tuple(int(k) for k in x.split()),
    'Offset': lambda x: tuple(int(k) for k in x.split()),
    'ElementSpacing': lambda x: tuple(float(k) for k in x.split()),
}

_dtypes = {
    'MET_SHORT': np.int16
}


class Tsdf(object):
    def __init__(self, offset, spacing, data):
        self.offset = np.array(offset)
        self.spacing = np.array(spacing)
        # data[data == 0] = -16384
        self.data = data
        self.shape = data.shape

    def get_signed_distance(self, xyz, out=None):
        data = self.data
        if out is None:
            out = np.empty(shape=xyz.shape[:-1], dtype=np.int32)

        ijk = self.xyz_to_ijk(xyz).astype(np.int32)
        valid = np.all([ijk >= 0, ijk < self.shape], axis=(0, -1))
        i, j, k = ijk[valid].T
        out[valid] = data[i, j, k]
        invalid = np.logical_not(valid)
        i, j, k = ijk[invalid].T
        out[invalid] = invalid_depth
        return out

    def xyz_to_ijk(self, xyz):
        xyz = xyz * 1000  # m to mm
        return (xyz + self.offset) / self.spacing

    def ijk_to_xyz(self, ijk):
        xyz = ijk * self.spacing - self.offset
        xyz /= 1000  # mm to m
        return xyz


class TsdfMeta(object):
    def __init__(self, n_dims, shape, offset, spacing, dtype, data_filename):
        self.n_dims = n_dims
        self.shape = shape
        self.offset = np.array(offset, dtype=np.float32)
        self.spacing = np.array(spacing, dtype=np.float32)
        self.dtype = dtype
        self.data_filename = data_filename

    @staticmethod
    def from_file(path):
        with open(path, 'r') as fp:
            data = dict()
            for line in fp.readlines():
                line = line.rstrip()
                if len(line) > 0:
                    key, value = line.split(' = ')
                    data[key] = value
        return TsdfMeta.from_raw(**data)

    @staticmethod
    def from_raw(
            NDims, DimSize, Offset, ElementSpacing, ElementType,
            ElementDataFile):
        return TsdfMeta(
            n_dims=int(NDims),
            shape=tuple(int(k) for k in DimSize.split()),
            offset=tuple(int(k) for k in Offset.split()),
            spacing=tuple(float(k) for k in ElementSpacing.split()),
            dtype=_dtypes[ElementType],
            data_filename=ElementDataFile)


def load_tsdf(scene_id):
    folder = get_tsdf_dir()
    if not os.path.isdir(folder):
        extract_tsdf()
    meta = TsdfMeta.from_file(os.path.join(folder, '%s.mhd' % scene_id))
    data = np.fromfile(
        os.path.join(folder, meta.data_filename), dtype=meta.dtype)
    return Tsdf(meta.offset, meta.spacing, data.reshape(meta.shape))
    # return meta, data.reshape(meta.shape)
