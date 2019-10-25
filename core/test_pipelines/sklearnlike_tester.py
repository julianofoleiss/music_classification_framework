from model_tester import ModelTester
from core.utils.encodings import inv_dict, to_label_matrix
import numpy as np

class SklearnLike(ModelTester):

    def __init__(self, params):
        super(SklearnLike, self).__init__(params)

    def _predict(self, model, test_features, track_idxs=None, pf=None, **kwargs):
        
        if (hasattr(model, 'fit') == False) or (hasattr(model, 'predict') == False):
            raise ValueError, 'model must be a Sklearn-like classifier, i.e. it must have a fit and a predict method!'

        #print('test features in sklearnlike _predict', test_features.shape)
            
        Y = model.predict(test_features)

        #label_dict = ...
        exec(open(self.params['general']['label_dict_file']).read())

        #Check if the prediction returned class ids instead of strings.
        #This is also a shameless hack to accomodate thundersvm.
        thundersvm=False
        if hasattr(Y, 'dtype'):
            if Y.dtype.char != 'S':
                #print('Y is floating')
                if pf is not None:
                    #print('pf is not None!')
                    Y = Y.astype(int)
                    Y = [pf.label_idx_dict[i] for i in Y]
                    thundersvm = True

        Y_M = to_label_matrix(Y, label_dict)

        if thundersvm:
            del self.model
            del model

        return Y_M

        