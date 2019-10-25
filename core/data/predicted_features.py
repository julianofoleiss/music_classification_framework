from predicted import Predicted

class PredictedFeatures(Predicted):

    def __init__(self, track_filenames, track_nframes, estimator_output, label_idx_dict, feature_filenames=None, labels=None):
        super(PredictedFeatures, self).__init__(track_filenames, track_nframes, estimator_output, label_idx_dict)

        self.feature_filenames = feature_filenames

        if hasattr(labels, 'ndim'):
            if labels.ndim < 2:
                labels = labels.reshape((-1, labels.shape[0]))

        self.labels = labels
    
    def get_track_label(self, filename):
        return self.labels[ self.track_filenames_frame_idx_dict[filename][1]]
