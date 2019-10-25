import numpy as np
from prediction_voting import PredictionVoting

class SumVoting(PredictionVoting):

    def __init__(self, params):
        super(SumVoting, self).__init__(params)

    def _vote(self, predictions, track_nframes, p=None, **kwargs):

        final = []
        for i in p.get_track_idxs():
            decision = np.argmax(np.sum(p.estimator_output[i], axis=0))
            final.append(p.label_idx_dict[decision] if p.label_idx_dict is not None else decision)

        return final
