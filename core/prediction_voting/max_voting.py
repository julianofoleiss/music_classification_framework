import numpy as np
from prediction_voting import PredictionVoting

class MaxVoting(PredictionVoting):

    def __init__(self, params):
        super(MaxVoting, self).__init__(params)

    def _vote(self, predictions, track_nframes, p=None, **kwargs):
        
        predicted_classes = p.max_prediction()

        final = []
        for i in p.get_track_idxs():
            decision = np.argmax(np.bincount(predicted_classes[i]))
            final.append(p.label_idx_dict[decision] if p.label_idx_dict is not None else decision)

        return final
