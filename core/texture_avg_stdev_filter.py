# -*- coding: utf-8 -*-
from feature_filter import FeatureFilter
from utils import texture

class TextureAvgStdevFilter(FeatureFilter):

    def __init__(self, params=None):
        self.params = params

    def _filter(self, matrix):
        return texture.to_texture(matrix, self.params['texture_avg_stdev_filter']['window_size'], 
            mean=self.params['texture_avg_stdev_filter']['mean'],
            variance=self.params['texture_avg_stdev_filter']['variance'])
        