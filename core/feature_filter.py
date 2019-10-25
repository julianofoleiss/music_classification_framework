# -*- coding: utf-8 -*-
import six

from data.predicted_features import PredictedFeatures

class FeatureFilter(object):

    def __init__(self, params=None):
        pass

    def filter(self, ipt, transpose=False, **kwargs):
        import dill
        from core.data.predicted_features import PredictedFeatures

        feats = ipt

        if isinstance(ipt, six.string_types):
            feats = dill.load(open(ipt))

        assert isinstance(feats, PredictedFeatures), "input features must be in PredictedFeatures format!"

        m = feats.estimator_output.T if transpose else feats.estimator_output

        textured = self._filter(m, **kwargs)

        out = PredictedFeatures(feats.track_filenames, [textured.shape[0]], textured, feats.label_idx_dict, feats.feature_filenames, feats.labels)

        return out        

    def _filter(self, matrix, **kwargs):
        raise NotImplementedError
