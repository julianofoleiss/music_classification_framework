import six
from core.data.predicted import Predicted

class PredictionVoting(object):

    def __init__(self, params):
        self.params = params

    def vote(self, predictions, output_fmt='text', **kwargs):

        if isinstance(predictions, six.string_types):
            predictions = dill.load(open(predictions))

        assert isinstance(predictions, Predicted), "pf must be in Predicted format!"

        voted = self._vote(predictions.estimator_output, predictions.track_nframes, p=predictions)

        if output_fmt=='text':
            lines = [ "%s\t%s" % (predictions.track_filenames[i], voted[i]) for i in xrange(len(voted))]
            out = '\n'.join( lines )

        if output_fmt=='predicted':
            out = Predicted(predictions.track_filenames, [1] * len(voted), voted, predictions.label_idx_dict)

        return out

    def _vote(self, predictions, track_nframes, p=None, **kwargs):
        raise NotImplementedError
