# -*- coding: utf-8 -*-
from feature_filter import FeatureFilter
import numpy as np

class IVectorFilter(FeatureFilter):

    def __init__(self, params=None):
        self.params = params

    def _filter(self, matrix):
        return np.average(matrix, axis=0).reshape((1,-1))
        