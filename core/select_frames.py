# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import six

class FrameSelector(object):
    def __init__(self, params=None):
        self.params = params

    @staticmethod
    def get_selector(selector, params):
        if selector == 'kmeansc':
            return KMeansClusterSelector(params)

        if selector == 'kmeansf':
            return KMeansFrameSelector(params)

        if selector == 'linspace':
            return LinSpaceFrameSelector(params)

        if selector == 'centerk':
            return CenterKFrameSelector(params)

        if selector == 'random':
            return RandomFrameSelector(params)

        return None

    def _select_frames(self, features, n_frames, normalize=False):
        raise NotImplementedError

    def select_frames(self, ipt):
        """
        Selects frames using the instantiated selector.

        :param ipt: path to input PredictedFeatures pickle or PredictedFeatures instance.
        :returns: returns a new instance of PredictedFeatures containing only the selected features.

        :type ipt: str
        :rtype: PredictedFeatures
        """        
        import dill
        from core.data.predicted_features import PredictedFeatures

        if self.params is not None:
            normalize = self.params['frame_selector']['normalize']
            n_frames = self.params['frame_selector']['n_frames']
        else:
            normalize = True
            n_frames = 20

        feats = ipt

        if isinstance(ipt, six.string_types):
            feats = dill.load(open(ipt))
        
        assert isinstance(feats, PredictedFeatures), "input features must be in PredictedFeatures format!"

        frames = self._select_frames(feats.estimator_output, n_frames, normalize)

        if self.params['frame_selector']['subset_size'] is not None:
            frames = frames[:self.params['frame_selector']['subset_size']]

        out = PredictedFeatures(feats.track_filenames, [frames.shape[0]], frames, feats.label_idx_dict, feats.feature_filenames, feats.labels)

        return out

    def _norm_features(self, features):
        ss = StandardScaler()
        ss.fit(features)

        return ss.transform(features, copy=True)

class KMeansClusterSelector(FrameSelector):
    def __init__(self, params=None):
        super(KMeansClusterSelector, self).__init__(params)

    def _select_frames(self, features, n_frames, normalize=False):
        X = features

        if normalize:
            X = self._norm_features(features)

        km = self._fit_kmeans(X, n_frames)

        if self.params['kmeansc_selector']['sort_centroids_by_common'] is not None:
            dist = km.transform(X)
            sort_idxs = np.argsort(np.bincount(np.argmin(dist, axis=1)))
            if self.params['kmeansc_selector']['sort_centroids_by_common'] == 'desc':
                sort_idxs = np.flip(sort_idxs, -1)
        else:
            sort_idxs = np.arange(km.cluster_centers_.shape[0])

        return km.cluster_centers_[sort_idxs]

    def _fit_kmeans(self, features, n_frames):
        km = KMeans(n_clusters=n_frames)
        km.fit(features)

        return km

class KMeansFrameSelector(KMeansClusterSelector):
    def __init__(self, params=None):
        super(KMeansFrameSelector, self).__init__(params)
    
    def _select_frames(self, features, n_frames, normalize=False):
        X = features

        if normalize:
            X = self._norm_features(features)

        km = self._fit_kmeans(X, n_frames)
        d = km.transform(X)
        frames = X[ np.argmin(d, axis=0) ]

        return frames

class LinSpaceFrameSelector(FrameSelector):

    def __init__(self, params=None):
        super(LinSpaceFrameSelector, self).__init__(params)

    def _select_frames(self, features, n_frames, normalize=False):
        X = features

        if normalize:
            X = self._norm_features(features)

        frame_idxs = np.linspace(0, features.shape[0]-1, n_frames, dtype=np.int32)

        if n_frames == 1:
            if self.params['linspace_selector']['single_is_center_frame']:
                frame_idxs = np.array([(features.shape[0]-1) / 2])

        frames = X[ frame_idxs ]

        return frames

class CenterKFrameSelector(FrameSelector):

    def __init__(self, params=None):
        super(CenterKFrameSelector, self).__init__(params)

    def _select_frames(self, features, n_frames, normalize=False):
        #X = features

        #if normalize:
        #    X = self._norm_features(features)

        n_frames = float(n_frames)

        le = int(math.floor(n_frames/2))
        re = int(math.ceil(n_frames/2))

        start = (features.shape[0]//2) - le
        end = (features.shape[0]//2) + re
        
        if start < 0:
            
            print('Warning: CenterKFrameSelector: padding needed. Pad? %s. Number of Frames: %s.' % 
                (self.params['centerk_selector']['pad'], features.shape[0]))

            if self.params['centerk_selector']['pad']:
                padl = np.abs(start)
                padr = int(n_frames - end)
                features = np.pad(features,[(padl,padr),(0,0)], 'constant', constant_values=0)
                end=int(np.floor(n_frames))
            else:
                end  = features.shape[0]

            start = 0

            print('len of padded vector ', len(np.arange(features.shape[0])[start:end]))

        frame_idxs = np.arange(features.shape[0])[start:end]

        frames = features[ frame_idxs ]

        return frames

class RandomFrameSelector(FrameSelector):
    def __init__(self, params=None):
        super(RandomFrameSelector, self).__init__(params)

    def _select_frames(self, features, n_frames, normalize=False):
        X = features

        if normalize:
            X = self._norm_features(features)

        rs = np.random.RandomState(self.params['random_selector']['seed'])

        selected_idxs = rs.choice(features.shape[0], n_frames, replace=False)

        return features[selected_idxs]

if __name__ == "__main__":
    import dill
    import yaml
    import matplotlib.pyplot as plt
    import librosa.display
    from sklearn.metrics.pairwise import cosine_distances

    feats = dill.load(open('rock_00000.feats'))

    params = yaml.load( open('parameters.yaml').read() )

    fs = FrameSelector.get_selector('kmeansf', params)
    frames = fs._select_frames(feats.estimator_output, 10, True)

    f2 = FrameSelector.get_selector('kmeansc', params)
    frames2 = f2._select_frames(feats.estimator_output, 10, True)

    plt.figure()
    plt.subplot(1,2,1)
    librosa.display.specshow(frames.T)

    plt.subplot(1,2,2)
    librosa.display.specshow(frames2.T)

    plt.figure()
    grayscale_map = plt.get_cmap('gray')
    plt.matshow( cosine_distances(frames, frames2) / 2, cmap=grayscale_map )
    plt.colorbar()
    plt.show()

    dill.dump(f2.select_frames('rock_00000.feats'), open('rock_00000_red.feats', 'w'))
    

