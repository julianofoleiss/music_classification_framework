# -*- coding: utf-8 -*-
from feature_filter import FeatureFilter
from utils import texture
import numpy as np

class LinspaceDownsampleFilter(FeatureFilter):

    def __init__(self, params=None):
        self.params = params

    def _filter(self, matrix):
        return linspace_downsample(matrix, self.params['linspace_downsample_filter']['hop_size'])
        
def linspace_downsample(feature_matrix, hop_size):
    n_texture_windows = feature_matrix.shape[0] / hop_size
    tidx = np.floor(np.linspace(0, feature_matrix.shape[0], n_texture_windows, endpoint=False)).astype(int)
    return feature_matrix[tidx]
