import numpy as np

class DataBatchGenerator(object):
    def __init__():
        pass

    def next(batch_size=None, shuffle=False):
        raise NotImplementedError

class TensorBatches(DataBatchGenerator):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def next(self, batch_size=None, shuffle=False):
        if self.targets is not None:
            assert len(self.inputs) == len(self.targets)
        if shuffle:
            indices = np.arange(len(self.inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(self.inputs), batch_size):
            if shuffle:
                excerpt = indices[start_idx: min(start_idx + batch_size, len(self.inputs))]
            else:
                excerpt = slice(start_idx, min(start_idx + batch_size, len(self.inputs)))
            if self.targets is not None:
                yield self.inputs[excerpt], self.targets[excerpt]
            else:
                yield self.inputs[excerpt], None
                
class TensorAutoEncoderBatches(TensorBatches):

    def __init__(self, inputs):
        super(TensorAutoEncoderBatches, self).__init__(inputs, None)

    def next(self, batch_size=None, shuffle=False):

        for X, _ in super(TensorAutoEncoderBatches, self).next(batch_size, shuffle):
            yield X, X